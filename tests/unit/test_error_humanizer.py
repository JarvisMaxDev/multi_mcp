"""Unit tests for error_humanizer — pattern detection and credential sanitization."""

from multi_mcp.utils.error_humanizer import humanize_error


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
