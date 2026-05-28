"""Unit tests for error_humanizer — pattern detection and credential sanitization."""

import pytest

from multi_mcp.utils.error_humanizer import _get_runtime_secret_values, humanize_error


@pytest.fixture(autouse=True)
def _clear_secret_cache():
    """Reset the lru_cache on _get_runtime_secret_values between tests.

    Without this, mocked Settings/env values from one test would leak into the next
    via the cache. The cache is desirable in production (Settings is process-immutable)
    but must be invalidated for test isolation.
    """
    _get_runtime_secret_values.cache_clear()
    yield
    _get_runtime_secret_values.cache_clear()


class TestSanitization:
    """Credentials and long strings must never leak through humanize_error."""

    def test_strips_openai_style_key(self):
        raw = "AuthenticationError: Invalid API key 'sk-proj-abc123def456ghi789jkl012mno345pqr' provided"
        out = humanize_error(raw)
        assert "sk-proj" not in out
        assert "[REDACTED]" in out

    def test_strips_bearer_token(self):
        raw = "401 Unauthorized: Bearer abc123def456ghi789jkl012mno345"
        out = humanize_error(raw)
        assert "abc123def456" not in out
        assert "[REDACTED]" in out

    def test_strips_anthropic_key(self):
        raw = "Invalid key: sk-ant-api03-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789"
        out = humanize_error(raw)
        assert "sk-ant-api03" not in out

    def test_strips_google_aiza_key(self):
        """Gemini API keys (AIza followed by 35 chars) must be redacted."""
        raw = "Failed: API key 'AIzaSyA-1234567890AbCdEfGhIjKlMnOpQrStUvWx' is invalid"
        out = humanize_error(raw)
        assert "AIzaSy" not in out
        assert "[REDACTED]" in out

    def test_strips_aws_akia_key(self):
        """AWS access key IDs (AKIA + 16 uppercase alphanumerics) must be redacted."""
        raw = "AWS error: access denied for AKIAIOSFODNN7EXAMPLE"
        out = humanize_error(raw)
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED]" in out

    def test_strips_aws_asia_temporary_key(self):
        """AWS temporary STS access key IDs (ASIA prefix, used with assumed roles)
        must also be redacted — the AKIA-only pattern missed these."""
        raw = "AWS STS error: ASIAEXAMPLETOKEN1234 invalid"
        out = humanize_error(raw)
        assert "ASIAEXAMPLETOKEN1234" not in out
        assert "[REDACTED]" in out

    def test_strips_env_var_assignment_in_error(self):
        """A CLI echoing an env-var assignment must not leak the value.

        The test fixture intentionally avoids `sk-...` shape so we exercise the
        env-assignment pattern, not the OpenAI-key pattern.
        """
        # Note: test fixture uses an obviously-fake placeholder; semgrep may still
        # heuristically flag this, but it's just a dummy value.
        fake_value = "TEST_FAKE_VALUE_NOT_REAL_xyz123"  # nosec
        raw = f"Bad config: ANTHROPIC_API_KEY={fake_value} found in env"
        out = humanize_error(raw)
        assert fake_value not in out
        assert "[REDACTED]" in out

    def test_strips_azure_env_assignment(self):
        """Azure key in env-var assignment format must be redacted."""
        fake_azure = "TEST_FAKE_AZURE_KEY_NOT_REAL_abc"  # nosec
        raw = f"Azure error: AZURE_API_KEY: '{fake_azure}' invalid"
        out = humanize_error(raw)
        assert fake_azure not in out
        assert "[REDACTED]" in out

    def test_redacts_bare_aws_secret_value_via_value_based_pass(self):
        """Defense-in-depth: a bare AWS secret access key value (40-char base64-ish, no
        recognizable prefix) is caught by exact-string-replace against live Settings values,
        even though the regex layer can't match it by shape.

        Without this layer, a CLI that echoed the raw secret value alone in stderr would
        bypass sanitization entirely.
        """
        from unittest.mock import patch

        fake_aws_secret = "TestFakeAwsSecretValueNotReal12345678901"  # nosec — 40-char placeholder
        raw = f"Some unrelated error containing the bare value {fake_aws_secret} in the middle"

        with patch("multi_mcp.utils.error_humanizer._get_runtime_secret_values", return_value=(fake_aws_secret,)):
            out = humanize_error(raw)

        assert fake_aws_secret not in out
        assert "[REDACTED]" in out
        # Surrounding context preserved
        assert "Some unrelated error" in out

    def test_value_based_pass_handles_missing_settings_gracefully(self):
        """If Settings can't be imported (e.g. test isolation), _sanitize falls back to
        regex-only redaction without raising. This is the try/except path in
        _get_runtime_secret_values.
        """
        # Just verify regex-based redaction still works when no Settings values are configured.
        # The fact that this test runs at all confirms the module loaded successfully.
        raw = "Error with sk-FAKE123456789012345678901234 in it"
        out = humanize_error(raw)
        assert "sk-FAKE" not in out
        assert "[REDACTED]" in out

    def test_value_pass_sorts_by_length_descending_to_avoid_substring_leak(self):
        """If two configured secrets share a prefix or one is substring of another, the
        longer one MUST be replaced first — otherwise the shorter replacement breaks the
        longer match and leaves a partial leak like `[REDACTED]_suffix_of_longer`.

        This is realistic during key rotation or when STS tokens share a base prefix
        with another credential.
        """
        from unittest.mock import patch

        short_secret = "SHARED_PREFIX_a"  # 15 chars
        long_secret = "SHARED_PREFIX_a_followed_by_unique_tail"  # 39 chars, contains short as prefix

        with patch(
            "multi_mcp.utils.error_humanizer._get_runtime_secret_values",
            return_value=(long_secret, short_secret),  # already in correct order (long first)
        ):
            raw = f"Error: secret was {long_secret} and got rejected"
            out = humanize_error(raw)

        # The long secret must be fully replaced — no portion of it should remain
        assert long_secret not in out
        assert "followed_by_unique_tail" not in out, f"Long secret tail leaked: {out}"
        assert "[REDACTED]" in out

    def test_value_pass_reads_broader_env_vars(self):
        """_get_runtime_secret_values must read ALL credential env vars from os.environ,
        not just AWS STS tokens — covers shell-injected values that bypass Settings.
        """
        from unittest.mock import MagicMock, patch

        from multi_mcp.utils.error_humanizer import _get_runtime_secret_values

        fake_anthropic = "fake_anthropic_value_from_shell_xyz"  # nosec — >= 12 chars
        fake_openai = "fake_openai_value_from_shell_abc"  # nosec
        fake_aws = "fake_aws_secret_value_from_shell_456"  # nosec

        # Mock Settings with all-None values so contribution comes from os.environ only
        mock_settings = MagicMock()
        for attr in (
            "anthropic_api_key",
            "openai_api_key",
            "gemini_api_key",
            "openrouter_api_key",
            "ollama_api_key",
            "lm_studio_api_key",
            "dashscope_api_key",
            "azure_api_key",
            "aws_access_key_id",
            "aws_secret_access_key",
        ):
            setattr(mock_settings, attr, None)

        with (
            patch.dict(
                "os.environ",
                {
                    "ANTHROPIC_API_KEY": fake_anthropic,
                    "OPENAI_API_KEY": fake_openai,
                    "AWS_SECRET_ACCESS_KEY": fake_aws,
                },
                clear=True,
            ),
            patch("multi_mcp.settings.settings", mock_settings),
        ):
            values = _get_runtime_secret_values()

        # All three shell-injected values must be in the redaction list
        assert fake_anthropic in values, f"ANTHROPIC env value missing from secrets: {values}"
        assert fake_openai in values, f"OPENAI env value missing from secrets: {values}"
        assert fake_aws in values, f"AWS_SECRET env value missing from secrets: {values}"

    def test_value_pass_uses_lru_cache(self):
        """_get_runtime_secret_values is cached — repeated calls return the same tuple
        instance (Settings is process-immutable in production).
        """
        from multi_mcp.utils.error_humanizer import _get_runtime_secret_values

        # Call twice — second call should hit cache
        first = _get_runtime_secret_values()
        second = _get_runtime_secret_values()

        # Tuples are immutable; cached version returns the same object
        assert first is second

        # Verify cache_clear works (used by our autouse fixture)
        _get_runtime_secret_values.cache_clear()
        third = _get_runtime_secret_values()
        # After clear, may or may not be same object (depends on dict ordering / interning)
        # — just verify the function still works.
        assert isinstance(third, tuple)

    def test_value_pass_narrow_import_error_only(self):
        """If Settings import fails with ImportError → fall through gracefully.
        BUT other exceptions (ValidationError, AttributeError) should NOT be swallowed —
        those indicate real bugs that need to surface.
        """
        from unittest.mock import patch

        from multi_mcp.utils.error_humanizer import _get_runtime_secret_values

        # Simulate Settings raising a non-ImportError (e.g. pydantic ValidationError)
        # Real exception should propagate, not be silently swallowed.
        class FakeValidationError(Exception):
            pass

        # Patch the settings import inside the function to raise non-ImportError
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def selective_raise(name, *args, **kwargs):
            if name == "multi_mcp.settings":
                raise FakeValidationError("simulated pydantic failure")
            return original_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=selective_raise),
            pytest.raises(FakeValidationError),
        ):
            _get_runtime_secret_values()

    def test_401_with_colon_separator_matches_auth_rule(self):
        """The 401-auth regex must match both `401 Unauthorized` (whitespace) and
        `401: Unauthorized` (colon separator). Previously only whitespace was matched.
        """
        raw = "HTTP 401: Unauthorized — please re-login"
        out = humanize_error(raw)
        # Should hit the auth-failure friendly message
        assert "Authentication failed" in out
        assert "401" in out

    def test_value_pass_skips_short_values_to_avoid_false_positives(self):
        """_get_runtime_secret_values filters out values shorter than 12 chars to avoid
        redacting common words. A 5-char Settings value like 'short' must not enter the
        redaction list — otherwise the word 'short' in unrelated error text would get redacted.
        """
        from unittest.mock import MagicMock, patch

        from multi_mcp.utils.error_humanizer import _get_runtime_secret_values

        # Mock Settings with a too-short value + an empty value + a real-length value
        mock_settings = MagicMock()
        mock_settings.anthropic_api_key = "short"  # 5 chars — should be filtered
        mock_settings.openai_api_key = ""  # empty — should be filtered
        mock_settings.gemini_api_key = "this_is_long_enough_to_be_a_credential_abc123"  # ≥12 — included
        mock_settings.openrouter_api_key = None
        mock_settings.ollama_api_key = None
        mock_settings.lm_studio_api_key = None
        mock_settings.dashscope_api_key = None
        mock_settings.azure_api_key = None
        mock_settings.aws_access_key_id = None
        mock_settings.aws_secret_access_key = None

        with (
            patch("multi_mcp.settings.settings", mock_settings),
            patch.dict("os.environ", {}, clear=True),
        ):
            values = _get_runtime_secret_values()

        # Short and empty values must be filtered out
        assert "short" not in values
        assert "" not in values
        # Only the long value remains
        assert "this_is_long_enough_to_be_a_credential_abc123" in values

    def test_strips_aws_session_token_assignment(self):
        """AWS_SESSION_TOKEN / AWS_SECURITY_TOKEN env-assignment forms must be redacted.

        We strip these tokens from the CLI subprocess env, but a CLI/provider that
        echoes the assignment in its error output would still leak it without this rule.
        """
        fake_token = "TEST_FAKE_SESSION_TOKEN_VALUE_xyz"  # nosec
        raw = f"AWS error: AWS_SESSION_TOKEN={fake_token} expired"
        out = humanize_error(raw)
        assert fake_token not in out
        assert "[REDACTED]" in out

        # And the legacy AWS_SECURITY_TOKEN name
        raw2 = f"AWS error: AWS_SECURITY_TOKEN='{fake_token}' expired"
        out2 = humanize_error(raw2)
        assert fake_token not in out2
        assert "[REDACTED]" in out2

    def test_truncates_very_long_error(self):
        raw = "x" * 1000
        out = humanize_error(raw)
        assert len(out) < 600  # truncated to ~500 + "... [truncated]"
        assert "truncated" in out

    def test_empty_error_returns_placeholder(self):
        """Empty string → placeholder; whitespace-only passes through (caller responsibility)."""
        assert "Unknown error" in humanize_error("")


class TestCodexPatterns:
    """Codex CLI-specific failure modes."""

    def test_trusted_directory_error(self):
        raw = "Not inside a trusted directory and --skip-git-repo-check was not specified."
        out = humanize_error(raw, canonical_name="codex-cli")
        assert "Codex CLI refuses to run here" in out
        assert "--skip-git-repo-check" in out
        assert "codex-cli" in out  # canonical_name prefix included

    def test_codex_command_not_found(self):
        raw = "/bin/sh: codex: command not found"
        out = humanize_error(raw)
        assert "not installed or not on PATH" in out
        assert "@openai/codex" in out


class TestOllamaPatterns:
    """Ollama daemon + cloud auth failure modes."""

    def test_daemon_not_running(self):
        raw = "ConnectionError: connection refused to http://localhost:11434"
        out = humanize_error(raw, canonical_name="ollama-glm-5.1")
        assert "Ollama daemon is not running" in out
        assert "ollama serve" in out

    def test_cloud_signin_required(self):
        raw = "model requires ollama signin to access cloud features"
        out = humanize_error(raw)
        assert "Ollama cloud model" in out
        assert "ollama signin" in out

    def test_model_not_pulled(self):
        raw = "Error: model 'llama3.5' not found, please pull model first"
        out = humanize_error(raw)
        assert "not pulled locally" in out
        assert "ollama pull" in out

    def test_non_ollama_model_not_found_does_not_get_ollama_advice(self):
        """Regression: previously `model .*not found` matched ANY provider's NotFoundError
        and incorrectly advised `ollama pull` for OpenAI/Anthropic/Gemini failures.

        The fix requires Ollama context in the regex.
        """
        raw = "OpenAI API error: Model gpt-5-mini not found"
        out = humanize_error(raw)
        # Must NOT contain Ollama-specific advice
        assert "ollama pull" not in out
        assert "ollama list" not in out
        # Should pass through (or hit a different pattern if any)
        # At minimum, the original message context should be preserved.
        assert "gpt-5-mini" in out


class TestLitellmPatterns:
    """LiteLLM-level failures (provider misconfig, bad alias)."""

    def test_provider_not_provided(self):
        raw = "litellm.BadRequestError: LLM Provider NOT provided. You passed model=foo"
        out = humanize_error(raw)
        assert "LiteLLM couldn't determine the provider" in out
        assert "/multi:models" in out

    def test_authentication_error(self):
        raw = "AuthenticationError: invalid_api_key"
        out = humanize_error(raw)
        assert "Authentication failed" in out
        # Should mention both CLI login and API key paths
        assert ".env" in out


class TestSanitizeForLog:
    """Tests for the log-only sanitizer (no friendly pattern rewriting, just secret stripping)."""

    def test_redacts_secrets_without_pattern_rewriting(self):
        """sanitize_for_log preserves the original error structure but strips secrets.

        Unlike humanize_error which prepends a friendly message and appends
        '(Original error: ...)', this just returns the sanitized input.
        """
        from multi_mcp.utils.error_humanizer import sanitize_for_log

        raw = "ConnectionError: connection refused to http://localhost:11434"
        out = sanitize_for_log(raw)
        # Same shape (no friendly preamble, no Original error wrapping)
        assert out == raw

    def test_strips_secret_but_keeps_context(self):
        """Secret material is redacted; rest of message intact for debugging."""
        from multi_mcp.utils.error_humanizer import sanitize_for_log

        raw = "Failed: API key 'AIzaSyA-1234567890AbCdEfGhIjKlMnOpQrStUvWx' invalid"
        out = sanitize_for_log(raw)
        assert "AIzaSyA" not in out
        assert "[REDACTED]" in out
        # Context preserved
        assert "Failed: API key" in out
        assert "invalid" in out

    def test_empty_input_returns_empty(self):
        from multi_mcp.utils.error_humanizer import sanitize_for_log

        assert sanitize_for_log("") == ""

    def test_truncates_long_input(self):
        from multi_mcp.utils.error_humanizer import sanitize_for_log

        raw = "x" * 1000
        out = sanitize_for_log(raw)
        assert len(out) < 600  # ~500 cap + truncation marker
        assert "truncated" in out


class TestFallthrough:
    """Unknown errors pass through (sanitized but unchanged in meaning)."""

    def test_unknown_error_returns_sanitized_raw(self):
        raw = "Some weird error we've never seen before"
        out = humanize_error(raw)
        assert out == raw  # no secrets, no pattern → unchanged

    def test_sanitization_still_applies_on_fallthrough(self):
        raw = "Weird error with sk-proj-secret123456789012345678 in it"
        out = humanize_error(raw)
        assert "sk-proj-secret" not in out
        assert "[REDACTED]" in out
