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
