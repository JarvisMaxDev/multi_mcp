"""Unit tests for CLI executor."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from multi_mcp.models.cli_executor import CLIExecutor
from multi_mcp.models.config import ModelConfig
from multi_mcp.schemas.base import ModelResponse


class TestCLIExecutor:
    """Tests for CLIExecutor class."""

    @pytest.fixture
    def cli_executor(self):
        """Create CLI executor instance."""
        return CLIExecutor()

    @pytest.fixture
    def cli_model_config(self):
        """Create sample CLI model config."""
        return ModelConfig(
            provider="cli",
            cli_command="gemini",
            cli_args=["chat"],
            cli_parser="json",
            cli_env={},
        )

    @pytest.fixture
    def mock_subprocess_success(self):
        """Create mock successful subprocess."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"response": "Test response from CLI"}', b""))
        return mock_process

    @pytest.fixture
    def mock_subprocess_failure(self):
        """Create mock failed subprocess."""
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error: something went wrong"))
        return mock_process

    @pytest.mark.asyncio
    async def test_execute_success(self, cli_executor, cli_model_config, mock_subprocess_success):
        """Test successful CLI execution."""
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_subprocess_success

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test prompt"}],
            )

            assert isinstance(result, ModelResponse)
            assert result.status == "success"
            assert result.content == "Test response from CLI"
            assert result.metadata.model == "gemini-cli"
            assert result.metadata.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_command_not_found(self, cli_executor, cli_model_config):
        """Test CLI command not found in PATH."""
        with patch("shutil.which", return_value=None):
            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.status == "error"
            assert "not found in PATH" in result.error
            assert "Install via" in result.error or "Ensure" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_cli_command(self, cli_executor):
        """Test error when cli_command is not configured."""
        config = ModelConfig(provider="cli")

        result = await cli_executor.execute(
            canonical_name="test-cli",
            model_config=config,
            messages=[{"role": "user", "content": "Test"}],
        )

        assert result.status == "error"
        assert "no cli_command configured" in result.error

    @pytest.mark.asyncio
    async def test_execute_timeout(self, cli_executor, cli_model_config):
        """Test CLI execution timeout handling."""
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
        ):
            mock_settings.model_timeout_seconds = 1

            # Create mock process that will timeout
            mock_process = MagicMock()
            mock_process.returncode = None  # Still running
            mock_process.kill = MagicMock()
            mock_process.communicate = AsyncMock(side_effect=TimeoutError())

            mock_exec.return_value = mock_process

            # First call to communicate() will timeout
            async def slow_communicate(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return (b"", b"")

            mock_process.communicate.side_effect = None
            mock_process.communicate = slow_communicate

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.status == "error"
            assert "timed out" in result.error

    @pytest.mark.asyncio
    async def test_execute_non_zero_exit(self, cli_executor, cli_model_config, mock_subprocess_failure):
        """Test CLI execution with non-zero exit code."""
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.return_value = mock_subprocess_failure

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.status == "error"
            assert "failed with exit code 1" in result.error
            assert "Error: something went wrong" in result.error

    @pytest.mark.asyncio
    async def test_execute_exception_handling(self, cli_executor, cli_model_config):
        """Test CLI execution exception handling."""
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
        ):
            mock_exec.side_effect = Exception("Unexpected error")

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test"}],
            )

            assert result.status == "error"
            assert "CLI execution failed" in result.error
            assert "Unexpected error" in result.error

    def test_parse_output_json(self, cli_executor):
        """Test JSON output parsing."""
        stdout = '{"response": "Test response"}'
        result = cli_executor._parse_output(stdout, "json")
        assert result == "Test response"

    def test_parse_output_json_claude_format(self, cli_executor):
        """Test JSON output parsing with Claude CLI format."""
        stdout = '{"result": "Test response", "is_error": false}'
        result = cli_executor._parse_output(stdout, "json")
        assert result == "Test response"

    def test_parse_output_json_error(self, cli_executor):
        """Test JSON output parsing with Claude CLI error."""
        stdout = '{"result": "Error message", "is_error": true}'
        with pytest.raises(ValueError, match="Claude CLI error"):
            cli_executor._parse_output(stdout, "json")

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_empty_cli_output(self, cli_executor, cli_model_config):
        """CLI that exits 0 with empty stdout must produce status:error, not silent success.

        Parity with LiteLLMClient: downstream consumers (codereview/debate) cannot
        distinguish between "model returned successful empty answer" and "model failed
        but exited 0", so we surface this as an error.
        """
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"", b""))  # empty stdout AND stderr

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            assert "empty output" in result.error

    @pytest.mark.asyncio
    async def test_execute_returns_error_on_whitespace_only_output(self, cli_executor, cli_model_config):
        """Whitespace-only stdout also counts as empty (would break consumers expecting real text)."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        # JSON parser will fall back to text and return the whitespace
        mock_process.communicate = AsyncMock(return_value=(b"  \n  \t  ", b""))

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            assert "empty output" in result.error

    @pytest.mark.asyncio
    async def test_execute_humanizes_codex_trusted_dir_error(self, cli_executor):
        """When codex CLI reports the trusted-directory failure, the user must see actionable advice,
        not just the raw stderr line. Verifies the humanize_error integration in the exit-code branch.
        """
        codex_config = ModelConfig(
            provider="cli",
            cli_command="codex",
            cli_args=["exec"],
            cli_parser="jsonl",
            cli_env={},
        )
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(
            return_value=(b"", b"Error: Not inside a trusted directory and --skip-git-repo-check was not specified.\n")
        )

        with (
            patch("shutil.which", return_value="/usr/bin/codex"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="codex-cli",
                model_config=codex_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            # The humanized message should appear, not just the raw stderr
            assert "Codex CLI refuses to run here" in result.error
            assert "--skip-git-repo-check" in result.error

    @pytest.mark.asyncio
    async def test_execute_sanitizes_cli_args_in_log_interaction(self, cli_executor):
        """If a user puts a secret-shaped value in cli_args (e.g. `--api-key=sk-...`),
        it must be sanitized before being passed to log_llm_interaction.

        Defense-in-depth: our default configs use cli_env (which is also sanitized) for
        credentials, but a custom config could legitimately put a token in cli_args.
        """
        fake_secret_in_args = "sk-FAKE-IN-ARGS-1234567890abcdef"  # nosec
        config = ModelConfig(
            provider="cli",
            cli_command="claude",
            cli_args=[f"--api-key={fake_secret_in_args}"],
            cli_parser="json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result":"ok","is_error":false}', b""))

        captured_request_data = {}

        def capture_log(**kwargs):
            captured_request_data.update(kwargs.get("request_data", {}))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction", side_effect=capture_log),
        ):
            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            command = captured_request_data.get("command", [])
            command_str = " ".join(command)
            # The fake secret must not appear in the logged command
            assert fake_secret_in_args not in command_str, f"Secret leaked in command log: {command_str}"
            # Redaction marker should be present
            assert "[REDACTED]" in command_str

    @pytest.mark.asyncio
    async def test_execute_sanitizes_unknown_cli_failure_stderr(self, cli_executor, cli_model_config):
        """Regression test: when stderr contains an unknown failure pattern AND a secret,
        the secret must NOT appear in the user-visible error (only the install-hint
        fallback path is used, but the stderr preview going into that fallback must be sanitized).

        Found by claude+codex in round-6 review — the previous code returned raw `error_preview`
        when no pattern matched, leaking any secrets echoed by the CLI into stderr.
        """
        # Note: the test fixture uses a fake "secret"-shaped value to exercise the sk-* pattern.
        fake_secret = "sk-FAKE-TEST-VALUE-1234567890abcdef"  # nosec
        stderr_with_secret = f"Random unique error that matches no humanize rule. config={fake_secret}"
        mock_process = MagicMock()
        mock_process.returncode = 137
        mock_process.communicate = AsyncMock(return_value=(b"", stderr_with_secret.encode()))

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            # The secret value must NOT appear in the user-visible error message
            assert fake_secret not in result.error, f"Secret leaked: {result.error}"
            # The redaction marker should be present
            assert "[REDACTED]" in result.error
            # The install-hint format is preserved
            assert "failed with exit code 137" in result.error
            assert "Troubleshooting:" in result.error
            # The context around the secret is preserved (sanitized, not replaced)
            assert "Random unique error" in result.error

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_install_hint_for_unknown_cli_failure(self, cli_executor, cli_model_config):
        """For unrecognized failure modes (no pattern match), preserve the original
        'failed with exit code N + Troubleshooting:' format with install hint.

        This is the bug fixed in humanize_error fallback: previously the string-equality
        check broke when sanitize() truncated the input or when input was empty.
        """
        mock_process = MagicMock()
        mock_process.returncode = 137
        mock_process.communicate = AsyncMock(return_value=(b"", b"Some unique error nobody has ever seen before"))

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            # Unrecognized error → keep the rich exit-code format
            assert "failed with exit code 137" in result.error
            assert "Troubleshooting:" in result.error
            assert "Some unique error nobody has ever seen before" in result.error

    @pytest.mark.asyncio
    async def test_execute_handles_empty_stderr_with_install_hint(self, cli_executor, cli_model_config):
        """Empty stderr/stdout still produces a useful error with install hint, not a
        placeholder. Was the regression caused by humanize_error("") returning the
        "Unknown error" placeholder, which broke the previous equality-based fallback.
        """
        mock_process = MagicMock()
        mock_process.returncode = 127
        mock_process.communicate = AsyncMock(return_value=(b"", b""))  # both empty

        # The empty-output guard fires BEFORE the exit-code branch can claim it.
        # That's correct behavior — empty stdout with exit 0 OR exit nonzero both
        # produce an error; here we verify that the exit-code branch handles the
        # empty-stderr case cleanly when it does run (exit code nonzero, but no
        # stderr to humanize).
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            # Must surface exit code and install hint even with empty stderr
            assert "failed with exit code 127" in result.error
            assert "Troubleshooting:" in result.error
            assert "(no output)" in result.error  # fallback for empty stderr

    @pytest.mark.asyncio
    async def test_execute_surfaces_claude_application_error_cleanly(self, cli_executor, cli_model_config):
        """When Claude CLI reports is_error=true, execute() must return a clean error
        response — not a misleading 'CLI execution failed: ValueError: ...' wrapper.

        Regression: previously _parse_output raised ValueError, which was caught by the
        generic Exception handler and surfaced to users as if the subprocess crashed.
        The CLI actually succeeded (exit code 0); only the model reported a problem.
        """
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result": "rate limit exceeded", "is_error": true}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            # Error must contain the actual Claude message, not a Python exception trace
            assert "rate limit exceeded" in result.error
            assert "Claude CLI error" in result.error
            # Must NOT be wrapped in "CLI execution failed: ValueError: ..."
            assert "CLI execution failed" not in result.error
            assert "ValueError" not in result.error
            assert result.metadata.model == "claude-cli"

    def test_parse_output_json_malformed(self, cli_executor):
        """Test JSON parsing fallback for malformed JSON."""
        stdout = "not valid json"
        result = cli_executor._parse_output(stdout, "json")
        # Should fall back to text parsing
        assert result == "not valid json"

    def test_parse_output_jsonl(self, cli_executor):
        """Test JSONL output parsing."""
        stdout = """{"type": "text", "text": "Hello"}
{"type": "text", "text": "World"}"""
        result = cli_executor._parse_output(stdout, "jsonl")
        assert result == "Hello\nWorld"

    def test_parse_output_jsonl_codex_format(self, cli_executor):
        """Test JSONL output parsing with Codex format."""
        stdout = '{"type": "item.completed", "item": {"type": "agent_message", "text": "Test response"}}'
        result = cli_executor._parse_output(stdout, "jsonl")
        assert result == "Test response"

    def test_parse_output_jsonl_empty_lines(self, cli_executor):
        """Test JSONL parsing skips empty lines."""
        stdout = """{"type": "text", "text": "Hello"}

{"type": "text", "text": "World"}"""
        result = cli_executor._parse_output(stdout, "jsonl")
        assert result == "Hello\nWorld"

    def test_parse_output_text(self, cli_executor):
        """Test text output parsing."""
        stdout = "  Simple text response  \n"
        result = cli_executor._parse_output(stdout, "text")
        assert result == "Simple text response"

    def test_get_install_hint_gemini(self, cli_executor):
        """Install hint for gemini CLI must point at the real published package."""
        hint = cli_executor.get_install_hint("gemini")
        assert "npm install" in hint
        assert "@google/gemini-cli" in hint

    def test_get_install_hint_codex(self, cli_executor):
        """Install hint for codex CLI must point at OpenAI's real package (not Anthropic's)."""
        hint = cli_executor.get_install_hint("codex")
        assert "npm install" in hint
        assert "@openai/codex" in hint

    def test_get_install_hint_claude(self, cli_executor):
        """Install hint for Claude Code CLI must use npm (not pip — it's a Node package)."""
        hint = cli_executor.get_install_hint("claude")
        assert "npm install" in hint
        assert "@anthropic-ai/claude-code" in hint

    def test_get_install_hint_unknown(self, cli_executor):
        """Test install hint for unknown CLI."""
        hint = cli_executor.get_install_hint("unknown-cli")
        assert "Ensure 'unknown-cli' is installed" in hint

    @pytest.mark.asyncio
    async def test_execute_injects_allowed_api_key_for_first_party_cli(self, cli_executor):
        """First-party CLIs on the allowlist (claude/codex/gemini/qwen) receive their key.

        cli_env-driven ${VAR} expansion must also work for the allowed key.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="claude",  # on the allowlist for ANTHROPIC_API_KEY
            cli_args=[],
            cli_parser="text",
            cli_env={"API_KEY": "${ANTHROPIC_API_KEY}"},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Success", b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True),
        ):
            mock_settings.anthropic_api_key = "test-key"
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process

            result = await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=config,
                messages=[{"role": "user", "content": "Test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Anthropic key is the allowed one for claude — must be present
            assert call_env.get("ANTHROPIC_API_KEY") == "test-key"
            # And API_KEY (from cli_env) should be expanded to the same value
            assert call_env.get("API_KEY") == "test-key"
            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execute_strips_unrelated_keys_per_cli(self, cli_executor):
        """Each first-party CLI sees ONLY its own provider key, not others.

        Defense against supply-chain attacks: a compromised claude-code binary
        can't exfiltrate OPENAI_API_KEY or GEMINI_API_KEY because they were
        never in its environment.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="claude",
            cli_args=[],
            cli_parser="json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result":"ok","is_error":false}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            # Simulate the realistic threat: ALL provider keys present in
            # the parent shell (loaded from .env files by Settings).
            patch.dict(
                "os.environ",
                {
                    "PATH": "/usr/bin",
                    "ANTHROPIC_API_KEY": "shell-anthropic",
                    "OPENAI_API_KEY": "shell-openai",
                    "GEMINI_API_KEY": "shell-gemini",
                    "OPENROUTER_API_KEY": "shell-openrouter",
                },
                clear=True,
            ),
        ):
            mock_settings.anthropic_api_key = "settings-anthropic"
            mock_settings.openai_api_key = "settings-openai"
            mock_settings.gemini_api_key = "settings-gemini"
            mock_settings.openrouter_api_key = "settings-openrouter"
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Allowed: claude gets ANTHROPIC_API_KEY (from Settings, not shell)
            assert call_env.get("ANTHROPIC_API_KEY") == "settings-anthropic"
            # NOT allowed: claude must NOT see other providers' keys, even though
            # they were in the inherited shell env.
            assert "OPENAI_API_KEY" not in call_env, f"OPENAI_API_KEY leaked: {call_env.get('OPENAI_API_KEY')!r}"
            assert "GEMINI_API_KEY" not in call_env, f"GEMINI_API_KEY leaked: {call_env.get('GEMINI_API_KEY')!r}"
            assert "OPENROUTER_API_KEY" not in call_env, f"OPENROUTER_API_KEY leaked: {call_env.get('OPENROUTER_API_KEY')!r}"

    @pytest.mark.asyncio
    async def test_execute_custom_cli_gets_no_keys_by_default(self, cli_executor):
        """A custom CLI (not on the allowlist) receives NO provider keys by default.

        Users must opt in explicitly via cli_env in config.yaml. This is the
        defense-in-depth contract: unknown CLIs are untrusted until explicitly
        granted credentials.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="my-custom-cli",  # NOT on allowlist
            cli_args=[],
            cli_parser="text",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"ok", b""))

        with (
            patch("shutil.which", return_value="/usr/bin/my-custom-cli"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            patch.dict("os.environ", {"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "shell-key"}, clear=True),
        ):
            mock_settings.anthropic_api_key = "settings-key"
            mock_settings.openai_api_key = "settings-openai"
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="my-custom-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Custom CLI without explicit cli_env opt-in → NO provider keys at all
            assert "ANTHROPIC_API_KEY" not in call_env
            assert "OPENAI_API_KEY" not in call_env

    @pytest.mark.asyncio
    async def test_execute_strips_aws_session_and_region(self, cli_executor):
        """AWS_SESSION_TOKEN/AWS_SECURITY_TOKEN (STS temporary creds) and AWS_REGION_NAME
        (not secret but discloses deployment region) must also be stripped from every CLI subprocess.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="claude",
            cli_args=[],
            cli_parser="json",
            cli_env={},
        )
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result":"ok","is_error":false}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            patch.dict(
                "os.environ",
                {
                    "PATH": "/usr/bin",
                    # Temporary STS credentials in parent env (assumed role)
                    "AWS_SESSION_TOKEN": "FQoDYXdzE...long-session-token...",  # nosec
                    "AWS_SECURITY_TOKEN": "legacy-name-for-session-token",  # nosec
                    "AWS_REGION_NAME": "us-west-2",
                    "AZURE_API_VERSION": "2025-04-01-preview",
                },
                clear=True,
            ),
        ):
            mock_settings.anthropic_api_key = "claude-key"
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.azure_api_key = None
            mock_settings.azure_api_base = None
            mock_settings.azure_api_version = "2025-04-01-preview"
            mock_settings.aws_access_key_id = None
            mock_settings.aws_secret_access_key = None
            mock_settings.aws_region_name = "us-east-1"
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # All AWS session-related and region keys, plus Azure version, must be ABSENT
            assert "AWS_SESSION_TOKEN" not in call_env, f"AWS_SESSION_TOKEN leaked: {call_env.get('AWS_SESSION_TOKEN')!r}"
            assert "AWS_SECURITY_TOKEN" not in call_env, f"AWS_SECURITY_TOKEN leaked: {call_env.get('AWS_SECURITY_TOKEN')!r}"
            assert "AWS_REGION_NAME" not in call_env, f"AWS_REGION_NAME leaked: {call_env.get('AWS_REGION_NAME')!r}"
            assert "AZURE_API_VERSION" not in call_env, f"AZURE_API_VERSION leaked: {call_env.get('AZURE_API_VERSION')!r}"

    @pytest.mark.asyncio
    async def test_execute_strips_azure_and_aws_credentials(self, cli_executor):
        """Azure and AWS credentials (API-only providers with no CLI consumer) must be
        stripped from every CLI subprocess. They are written to os.environ by
        Settings.set_provider_env_vars at startup, so without explicit stripping they
        would inherit into any CLI subprocess.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="claude",  # any first-party CLI
            cli_args=[],
            cli_parser="json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result":"ok","is_error":false}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            # Simulate Azure/AWS keys present in the parent env (written there by
            # Settings.set_provider_env_vars or via shell rc).
            patch.dict(
                "os.environ",
                {
                    "PATH": "/usr/bin",
                    "AZURE_API_KEY": "azure-secret",
                    "AZURE_API_BASE": "https://internal.openai.azure.com/",
                    "AWS_ACCESS_KEY_ID": "AKIAIOSFODNN7EXAMPLE",
                    "AWS_SECRET_ACCESS_KEY": "aws-secret-key-value",
                },
                clear=True,
            ),
        ):
            mock_settings.anthropic_api_key = "claude-key"
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.azure_api_key = "azure-from-settings"
            mock_settings.azure_api_base = "https://from-settings.azure.com/"
            mock_settings.aws_access_key_id = "AKIAFROMSETTINGS01"
            mock_settings.aws_secret_access_key = "aws-from-settings"
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Allowed: claude gets its anthropic key
            assert call_env.get("ANTHROPIC_API_KEY") == "claude-key"
            # NOT allowed: Azure and AWS credentials must be ABSENT, both inherited
            # from shell env AND Settings-injected values.
            assert "AZURE_API_KEY" not in call_env, f"AZURE_API_KEY leaked: {call_env.get('AZURE_API_KEY')!r}"
            assert "AZURE_API_BASE" not in call_env, f"AZURE_API_BASE leaked: {call_env.get('AZURE_API_BASE')!r}"
            assert "AWS_ACCESS_KEY_ID" not in call_env, f"AWS_ACCESS_KEY_ID leaked: {call_env.get('AWS_ACCESS_KEY_ID')!r}"
            assert "AWS_SECRET_ACCESS_KEY" not in call_env, f"AWS_SECRET_ACCESS_KEY leaked: {call_env.get('AWS_SECRET_ACCESS_KEY')!r}"

    @pytest.mark.asyncio
    async def test_execute_custom_cli_cli_env_explicit_opt_in_still_works(self, cli_executor):
        """Custom CLI can request keys explicitly via cli_env (escape hatch).

        The ${ANTHROPIC_API_KEY} expansion must resolve from Settings, even
        though the executor stripped it from the inherited env first.
        """
        config = ModelConfig(
            provider="cli",
            cli_command="my-custom-claude-wrapper",  # NOT on allowlist
            cli_args=[],
            cli_parser="text",
            cli_env={"ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}"},  # explicit opt-in
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"ok", b""))

        with (
            patch("shutil.which", return_value="/usr/bin/my-custom-claude-wrapper"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True),
        ):
            mock_settings.anthropic_api_key = "user-anthropic-key"
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = None
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="my-custom-claude-wrapper",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # The explicit opt-in via cli_env brings the key in, expanded from Settings.
            assert call_env.get("ANTHROPIC_API_KEY") == "user-anthropic-key"

    @pytest.mark.asyncio
    async def test_execute_qwen_keys_only_forwarded_to_qwen(self, cli_executor):
        """Qwen-only credentials (OLLAMA/LM_STUDIO/DASHSCOPE) MUST NOT leak into non-qwen CLI envs.

        Principle of least privilege: claude/gemini/codex/custom subprocess environments
        shouldn't see Ollama/LM Studio/DashScope keys they have no use for.
        """
        non_qwen_config = ModelConfig(
            provider="cli",
            cli_command="claude",
            cli_args=[],
            cli_parser="json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"result":"ok","is_error":false}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            # Critical: simulate the realistic threat — qwen-only keys ALREADY
            # in os.environ (loaded by Settings from .env files). Without the
            # inherited-env strip in cli_executor these would leak to claude.
            # `clear=True` removes everything else so we test in isolation.
            patch.dict(
                "os.environ",
                {
                    "PATH": "/usr/bin",
                    "OLLAMA_API_KEY": "inherited-ollama-secret",
                    "LM_STUDIO_API_KEY": "inherited-lm-secret",
                    "DASHSCOPE_API_KEY": "inherited-ds-secret",
                },
                clear=True,
            ),
        ):
            mock_settings.anthropic_api_key = "claude-key"
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = "settings-ollama"
            mock_settings.lm_studio_api_key = "settings-lm-studio"
            mock_settings.dashscope_api_key = "settings-dashscope"
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="claude-cli",
                model_config=non_qwen_config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Anthropic key SHOULD be present (claude-cli's normal credential)
            assert call_env.get("ANTHROPIC_API_KEY") == "claude-key"
            # Qwen-only keys MUST be ABSENT — neither inherited from parent env
            # nor injected by Settings should reach a non-qwen subprocess.
            assert "OLLAMA_API_KEY" not in call_env, f"OLLAMA_API_KEY leaked into claude-cli env: {call_env.get('OLLAMA_API_KEY')!r}"
            assert "LM_STUDIO_API_KEY" not in call_env, (
                f"LM_STUDIO_API_KEY leaked into claude-cli env: {call_env.get('LM_STUDIO_API_KEY')!r}"
            )
            assert "DASHSCOPE_API_KEY" not in call_env, (
                f"DASHSCOPE_API_KEY leaked into claude-cli env: {call_env.get('DASHSCOPE_API_KEY')!r}"
            )

    @pytest.mark.asyncio
    async def test_execute_qwen_keys_forwarded_to_qwen(self, cli_executor):
        """Qwen-only credentials MUST be forwarded when cli_command == 'qwen' (positive case)."""
        qwen_config = ModelConfig(
            provider="cli",
            cli_command="qwen",
            cli_args=[],
            cli_parser="qwen-json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'[{"type":"result","is_error":false,"result":"ok"}]', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/qwen"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            # Isolate from real os.environ so we measure ONLY the Settings injection
            patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True),
        ):
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = "ollama-key"
            mock_settings.lm_studio_api_key = "lm-key"
            mock_settings.dashscope_api_key = "ds-key"
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="qwen-cli",
                model_config=qwen_config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            assert call_env["OLLAMA_API_KEY"] == "ollama-key"
            assert call_env["LM_STUDIO_API_KEY"] == "lm-key"
            assert call_env["DASHSCOPE_API_KEY"] == "ds-key"

    @pytest.mark.asyncio
    async def test_execute_qwen_basename_matching_for_absolute_paths(self, cli_executor):
        """Credential gating must work when cli_command is an absolute path or wrapper.

        Users may legitimately set `cli_command: /opt/homebrew/bin/qwen` or use a
        wrapper script. Exact-match `cli_command == "qwen"` would silently drop
        credentials for those configurations.
        """
        qwen_abspath_config = ModelConfig(
            provider="cli",
            cli_command="/opt/homebrew/bin/qwen",  # absolute path
            cli_args=[],
            cli_parser="qwen-json",
            cli_env={},
        )

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'[{"type":"result","is_error":false,"result":"ok"}]', b""))

        with (
            patch("shutil.which", return_value="/opt/homebrew/bin/qwen"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.settings") as mock_settings,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
            patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True),
        ):
            mock_settings.anthropic_api_key = None
            mock_settings.openai_api_key = None
            mock_settings.gemini_api_key = None
            mock_settings.openrouter_api_key = None
            mock_settings.ollama_api_key = "ollama-key"
            mock_settings.lm_studio_api_key = None
            mock_settings.dashscope_api_key = None
            mock_settings.model_timeout_seconds = 120

            mock_exec.return_value = mock_process
            await cli_executor.execute(
                canonical_name="qwen-cli",
                model_config=qwen_abspath_config,
                messages=[{"role": "user", "content": "test"}],
            )

            call_env = mock_exec.call_args[1]["env"]
            # Even with absolute path, basename "qwen" should match and inject
            assert call_env["OLLAMA_API_KEY"] == "ollama-key"

    @pytest.mark.asyncio
    async def test_execute_serializes_full_message_history(self, cli_executor, cli_model_config):
        """All messages (system + history + new user turn) should reach the CLI as a labeled transcript.

        Previously the executor sent only `messages[-1]["content"]`, silently dropping
        the system prompt and any prior turns. The fix serializes the full conversation
        with role markers so CLI agents receive the same context as API models.
        """
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b'{"response": "Answer"}', b""))

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=messages,
            )

            communicate_call = mock_process.communicate.call_args
            stdin_data = communicate_call[1]["input"].decode("utf-8")

            # All four messages must appear in stdin, in order, with role markers
            assert "[SYSTEM]\nSystem prompt" in stdin_data
            assert "[USER]\nFirst question" in stdin_data
            assert "[ASSISTANT]\nFirst answer" in stdin_data
            assert "[USER]\nSecond question" in stdin_data
            # Order check: system first, last user message last
            assert stdin_data.index("[SYSTEM]") < stdin_data.index("[USER]\nFirst question")
            assert stdin_data.index("[USER]\nFirst question") < stdin_data.index("[ASSISTANT]")
            assert stdin_data.index("[ASSISTANT]") < stdin_data.index("[USER]\nSecond question")

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_empty(self, cli_executor):
        """Empty message list serializes to empty string (no crash)."""
        assert cli_executor._format_messages_as_prompt([]) == ""

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_single_user(self, cli_executor):
        """Single user message serializes with role marker."""
        result = cli_executor._format_messages_as_prompt([{"role": "user", "content": "Hello"}])
        assert result == "[USER]\nHello"

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_missing_role_defaults_to_user(self, cli_executor):
        """Messages without an explicit role default to user (defensive)."""
        result = cli_executor._format_messages_as_prompt([{"content": "no role here"}])
        assert result == "[USER]\nno role here"

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_none_content(self, cli_executor):
        """None content must become an empty string, not the literal 'None'."""
        result = cli_executor._format_messages_as_prompt([{"role": "user", "content": None}])
        assert result == "[USER]\n"
        assert "None" not in result  # the literal word "None" must NOT appear

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_missing_content_key(self, cli_executor):
        """Missing content key must behave like empty content."""
        result = cli_executor._format_messages_as_prompt([{"role": "user"}])
        assert result == "[USER]\n"

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_list_content_text_parts(self, cli_executor):
        """Anthropic/OpenAI style list-of-parts content: keep text parts, drop others."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "source": {"type": "base64", "data": "..."}},
                    {"type": "text", "text": "world"},
                ],
            }
        ]
        result = cli_executor._format_messages_as_prompt(messages)
        assert result == "[USER]\nhello\nworld"
        assert "base64" not in result  # image part must be dropped

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_list_with_raw_strings(self, cli_executor):
        """List content that contains raw strings should keep them too."""
        messages = [{"role": "user", "content": ["first", "second"]}]
        result = cli_executor._format_messages_as_prompt(messages)
        assert result == "[USER]\nfirst\nsecond"

    @pytest.mark.asyncio
    async def test_format_messages_as_prompt_non_string_content_uses_json(self, cli_executor):
        """Non-string scalar content falls back to JSON (not Python repr)."""
        messages = [{"role": "user", "content": {"key": "value"}}]
        result = cli_executor._format_messages_as_prompt(messages)
        # Must use double quotes (json) not single quotes (Python repr)
        assert '"key"' in result
        assert '"value"' in result
        assert "'key'" not in result  # single-quoted repr must NOT appear

    @pytest.mark.asyncio
    async def test_parse_output_json_non_dict_returns_valid_json(self, cli_executor):
        """Fallthrough for non-dict JSON (e.g. bare list) must return re-parseable JSON, not Python repr."""
        stdout = '[{"a": 1}, {"b": 2}]'
        result = cli_executor._parse_output(stdout, "json")
        # Result must be valid JSON (double-quoted), not Python repr
        import json as _json

        reparsed = _json.loads(result)
        assert reparsed == [{"a": 1}, {"b": 2}]

    def test_parse_output_json_response_nested_object_reserialized(self, cli_executor):
        """If a CLI wraps a nested object under `response`, we must re-serialize it
        as JSON so the return type stays `str` and downstream consumers don't hit
        Python repr with single quotes."""
        stdout = '{"response": {"answer": 42, "ok": true}}'
        result = cli_executor._parse_output(stdout, "json")
        # Must be a string (return type contract)
        assert isinstance(result, str)
        # Must be re-parseable JSON (not Python repr)
        import json as _json

        reparsed = _json.loads(result)
        assert reparsed == {"answer": 42, "ok": True}
        # Defensive: no single-quoted Python dict repr should leak through
        assert "'" not in result or '"' in result  # double quotes present

    def test_parse_output_json_result_nested_object_reserialized(self, cli_executor):
        """Same contract for Claude-style {\"result\": <nested obj>} — must return valid JSON string."""
        stdout = '{"result": [1, 2, 3], "is_error": false}'
        result = cli_executor._parse_output(stdout, "json")
        assert isinstance(result, str)
        import json as _json

        reparsed = _json.loads(result)
        assert reparsed == [1, 2, 3]

    # ---- Qwen CLI: --output-format json emits a heterogeneous event array ---

    def test_parse_output_json_qwen_array_format(self, cli_executor):
        """Qwen array: final answer lives in the last `type:'result'` event's `result` field."""
        stdout = (
            '[{"type":"system","subtype":"session_start","model":"qwen3"},'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"thinking..."}]}},'
            '{"type":"result","subtype":"success","is_error":false,"result":"Qwen final answer"}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "Qwen final answer"

    def test_parse_output_json_qwen_error_in_result_field(self, cli_executor):
        """Qwen `is_error: true` with message in `result` → ValueError using that message."""
        stdout = '[{"type":"result","is_error":true,"result":"auth failed: invalid api key"}]'
        with pytest.raises(ValueError, match="Qwen CLI error: auth failed"):
            cli_executor._parse_output(stdout, "qwen-json")

    def test_parse_output_json_qwen_error_in_error_field(self, cli_executor):
        """Qwen `is_error: true` without `result` but with `error` → ValueError uses `error`."""
        stdout = '[{"type":"result","is_error":true,"error":"tool exec failed","subtype":"tool_error"}]'
        with pytest.raises(ValueError, match="Qwen CLI error: tool exec failed"):
            cli_executor._parse_output(stdout, "qwen-json")

    def test_parse_output_json_qwen_error_no_message(self, cli_executor):
        """Qwen `is_error: true` with no string fields → ValueError with 'unknown failure'."""
        stdout = '[{"type":"result","is_error":true}]'
        with pytest.raises(ValueError, match="Qwen CLI error: unknown failure"):
            cli_executor._parse_output(stdout, "qwen-json")

    def test_parse_output_json_qwen_assistant_fallback(self, cli_executor):
        """Empty/missing `result` field → fall back to last assistant `content[].text`."""
        stdout = (
            '[{"type":"system","subtype":"session_start"},'
            '{"type":"assistant","message":{"content":['
            '{"type":"text","text":"first chunk"},'
            '{"type":"text","text":"second chunk"}'
            "]}},"
            '{"type":"result","subtype":"success","is_error":false,"result":""}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "first chunk\nsecond chunk"

    def test_parse_output_json_qwen_assistant_fallback_no_result_event(self, cli_executor):
        """No `type:'result'` event at all → assistant text fallback still works."""
        stdout = (
            '[{"type":"system","subtype":"session_start"},'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"only assistant"}]}}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "only assistant"

    def test_parse_output_json_qwen_picks_last_result(self, cli_executor):
        """Multiple `type:'result'` events → use the last (final) one."""
        stdout = (
            '[{"type":"result","is_error":false,"result":"intermediate"},'
            '{"type":"assistant","message":{"content":[{"type":"text","text":"more"}]}},'
            '{"type":"result","is_error":false,"result":"final"}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "final"

    def test_parse_output_json_qwen_nested_result(self, cli_executor):
        """Qwen `result` as object/list (defensive) → re-serialize as JSON string."""
        stdout = '[{"type":"result","is_error":false,"result":{"answer":42,"ok":true}}]'
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert isinstance(result, str)
        import json as _json

        reparsed = _json.loads(result)
        assert reparsed == {"answer": 42, "ok": True}

    def test_parse_output_json_qwen_no_usable_content(self, cli_executor):
        """No result event and no assistant text → empty string so the empty-content guard
        in execute() catches it and returns a proper error. Raw events still in logs.

        Previously returned `json.dumps(events)` which slipped through the empty-content
        guard as meaningless `[{...}]` content — failure was indistinguishable from a
        real model response. Now returns "" so execute() rejects it cleanly.
        """
        stdout = '[{"type":"system","subtype":"session_start"},{"type":"tool_call","name":"foo"}]'
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == ""

    def test_parse_output_json_array_not_treated_as_qwen(self, cli_executor):
        """REGRESSION: `cli_parser: json` (generic) with a JSON array must NOT trigger qwen
        event parsing. Custom user CLIs declaring `cli_parser: json` and returning JSON
        arrays should get the array re-serialized as a JSON string (preserved data), not
        collapsed to the last 'result' field.
        """
        stdout = '[{"type":"result","result":"would be wrong to collapse this"}]'
        result = cli_executor._parse_output(stdout, "json")
        # Must be re-parseable JSON of the original array — NOT the inner 'result' string
        import json as _json

        reparsed = _json.loads(result)
        assert isinstance(reparsed, list)
        assert reparsed[0]["type"] == "result"
        # Specifically: NOT the qwen-extracted value
        assert result != "would be wrong to collapse this"

    def test_parse_output_qwen_json_missing_result_key_uses_assistant_fallback(self, cli_executor):
        """`is_error: false` with no `result` key at all (not just empty string) → assistant fallback."""
        stdout = (
            '[{"type":"assistant","message":{"content":[{"type":"text","text":"fallback text"}]}},'
            '{"type":"result","subtype":"success","is_error":false}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "fallback text"

    def test_parse_output_qwen_json_non_list_input(self, cli_executor):
        """Defensive: qwen-json parser given a single object (contract violation) re-serializes."""
        stdout = '{"type":"result","result":"single object"}'
        result = cli_executor._parse_output(stdout, "qwen-json")
        # Falls through to JSON re-serialization, not the dict-format handling
        import json as _json

        reparsed = _json.loads(result)
        assert reparsed == {"type": "result", "result": "single object"}

    def test_parse_output_qwen_json_malformed(self, cli_executor):
        """qwen-json parser falls back to text on malformed JSON."""
        stdout = "not valid json"
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "not valid json"

    def test_parse_output_qwen_json_text_part_non_string_ignored(self, cli_executor):
        """Defensive: assistant content[].text must be a string; non-string entries are skipped."""
        stdout = (
            '[{"type":"assistant","message":{"content":['
            '{"type":"text","text":{"nested":"object"}},'
            '{"type":"text","text":"valid string"}'
            "]}},"
            '{"type":"result","is_error":false,"result":""}]'
        )
        result = cli_executor._parse_output(stdout, "qwen-json")
        assert result == "valid string"

    def test_get_install_hint_qwen(self, cli_executor):
        """Install hint for Qwen Code CLI must reference its npm package."""
        hint = cli_executor.get_install_hint("qwen")
        assert "npm install" in hint
        assert "@qwen-code/qwen-code" in hint

    @pytest.mark.asyncio
    async def test_execute_qwen_cli_basic_execution(self, cli_executor):
        """End-to-end execute() with a qwen-shaped config and array JSON stdout."""
        qwen_config = ModelConfig(
            provider="cli",
            cli_command="qwen",
            cli_args=["--output-format", "json", "--approval-mode", "auto-edit"],
            cli_parser="qwen-json",
            cli_env={},
        )
        qwen_output = (
            b'[{"type":"system","subtype":"session_start","model":"glm-5.1:cloud"},'
            b'{"type":"assistant","message":{"content":[{"type":"text","text":"hi"}]}},'
            b'{"type":"result","subtype":"success","is_error":false,"result":"Hello from qwen"}]'
        )
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(qwen_output, b""))

        with (
            patch("shutil.which", return_value="/opt/homebrew/bin/qwen"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process
            result = await cli_executor.execute(
                canonical_name="qwen-cli",
                model_config=qwen_config,
                messages=[{"role": "user", "content": "say hi"}],
            )

            assert result.status == "success"
            assert result.content == "Hello from qwen"
            assert result.metadata.model == "qwen-cli"

    @pytest.mark.asyncio
    async def test_execute_qwen_cli_surfaces_application_error_cleanly(self, cli_executor):
        """Qwen `is_error: true` must surface as a clean error response, not 'CLI execution failed'.

        Parity with the Claude CLI is_error contract — the subprocess succeeded
        (exit 0), only the model reported a problem, so wrap it as application
        error rather than runtime crash.
        """
        qwen_config = ModelConfig(
            provider="cli",
            cli_command="qwen",
            cli_args=[],
            cli_parser="qwen-json",
            cli_env={},
        )
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(
            return_value=(
                b'[{"type":"result","subtype":"error","is_error":true,"result":"model overloaded"}]',
                b"",
            )
        )

        with (
            patch("shutil.which", return_value="/opt/homebrew/bin/qwen"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process
            result = await cli_executor.execute(
                canonical_name="qwen-cli",
                model_config=qwen_config,
                messages=[{"role": "user", "content": "test"}],
            )

            assert result.status == "error"
            assert "model overloaded" in result.error
            assert "Qwen CLI error" in result.error
            assert "CLI execution failed" not in result.error
            assert "ValueError" not in result.error

    def test_expand_env_vars_resolves_nested_references(self, cli_executor):
        """_expand_env_vars should resolve multi-level ${VAR} references by iterating."""
        env = {
            "FOO": "${BAR}",
            "BAR": "${BAZ}",
            "BAZ": "final",
        }
        # value references FOO, which references BAR, which references BAZ
        result = cli_executor._expand_env_vars("${FOO}", env)
        assert result == "final"

    def test_expand_env_vars_stops_on_cycle(self, cli_executor):
        """Reference cycles must not cause infinite loops — bounded iteration."""
        env = {
            "A": "${B}",
            "B": "${A}",
        }
        # Should not hang; exact output doesn't matter as long as it returns
        result = cli_executor._expand_env_vars("${A}", env)
        assert isinstance(result, str)  # terminates cleanly

    def test_expand_env_vars_single_level_still_works(self, cli_executor):
        """Single-level substitution (the common case) must still work."""
        env = {"KEY": "secret123"}
        result = cli_executor._expand_env_vars("${KEY}", env)
        assert result == "secret123"

    def test_expand_env_vars_missing_var_left_as_is(self, cli_executor):
        """Unknown variables should be left as ${NAME} literal (unchanged contract)."""
        env = {}
        result = cli_executor._expand_env_vars("${MISSING}", env)
        assert result == "${MISSING}"

    @pytest.mark.asyncio
    async def test_execute_cli_env_expansion_is_order_independent(self, cli_executor, mock_subprocess_success):
        """Regression: codex round-3 finding. cli_env expansion used to iterate
        in YAML insertion order, so `A=${B}` declared before `B=value` would
        leave A unresolved. Build a stable lookup map upfront so cross-key
        references work regardless of order."""
        # A references B, but A is declared FIRST (would have failed under old code)
        config = ModelConfig(
            provider="cli",
            cli_command="gemini",
            cli_args=["chat"],
            cli_parser="json",
            cli_env={"A": "${B}", "B": "resolved_value"},
        )

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_subprocess_success
            await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=config,
                messages=[{"role": "user", "content": "test"}],
            )
            call_kwargs = mock_exec.call_args.kwargs
            env = call_kwargs["env"]
            # A must be the resolved value of B, not the literal "${B}"
            assert env["A"] == "resolved_value"
            assert env["B"] == "resolved_value"

    @pytest.mark.asyncio
    async def test_execute_tolerates_process_lookup_error_on_cleanup(self, cli_executor, cli_model_config):
        """Regression: codex round-3 finding. After process.kill(), the child
        may have already exited (race window between returncode check and kill),
        so kill() can raise ProcessLookupError. The cleanup helper must catch it
        rather than letting it escape and mask the original error response."""
        mock_process = MagicMock()
        mock_process.returncode = None  # appears alive at the check
        mock_process.kill = MagicMock(side_effect=ProcessLookupError("already gone"))
        # communicate raises a generic error to enter the cleanup branch
        mock_process.communicate = AsyncMock(side_effect=RuntimeError("subprocess crash"))

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process
            result = await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "test"}],
            )
            # Must return a structured error, NOT crash with ProcessLookupError
            assert result.status == "error"
            assert "ProcessLookupError" not in result.error  # not leaked
            mock_process.kill.assert_called_once()  # cleanup was attempted

    @pytest.mark.asyncio
    async def test_execute_preflight_honors_cli_env_path_override(self, cli_executor, mock_subprocess_success):
        """Round-4 finding (codex medium): the shutil.which preflight used to
        consult os.environ PATH only, ignoring any PATH override declared in
        cli_env. So a CLI binary that's only reachable via a custom PATH would
        be wrongly rejected before launch. Now env is built first and shutil
        .which receives `path=env.get("PATH")`."""
        config = ModelConfig(
            provider="cli",
            cli_command="custom-bin",
            cli_args=["chat"],
            cli_parser="json",
            cli_env={"PATH": "/opt/custom/bin:/usr/bin"},
        )

        captured: dict[str, str | None] = {"path": None}

        def fake_which(cmd, path=None):
            captured["path"] = path
            return f"/opt/custom/bin/{cmd}"  # pretend the binary exists there

        with (
            patch("shutil.which", side_effect=fake_which) as mock_which,
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_subprocess_success
            result = await cli_executor.execute(
                canonical_name="custom-cli",
                model_config=config,
                messages=[{"role": "user", "content": "hi"}],
            )

            # Must have been called with explicit path= from the cli_env override
            assert mock_which.called
            assert captured["path"] is not None
            assert "/opt/custom/bin" in captured["path"]
            # And the call must succeed (not blocked by preflight)
            assert result.status == "success"

    @pytest.mark.asyncio
    async def test_execute_cancellation_kills_subprocess_and_propagates(self, cli_executor, cli_model_config):
        """Regression: gemini round-3 finding (CRITICAL). When the asyncio task
        is cancelled (client disconnect, parent task cancel), the subprocess
        must be killed so it doesn't leak as an orphan, AND the CancelledError
        must propagate so cancellation actually works."""
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.kill = MagicMock()
        # communicate raises CancelledError to simulate the task being cancelled
        mock_process.communicate = AsyncMock(side_effect=asyncio.CancelledError())

        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction"),
        ):
            mock_exec.return_value = mock_process

            # CancelledError must propagate, NOT be swallowed into a ModelResponse
            with pytest.raises(asyncio.CancelledError):
                await cli_executor.execute(
                    canonical_name="gemini-cli",
                    model_config=cli_model_config,
                    messages=[{"role": "user", "content": "test"}],
                )
            # And the subprocess must have been killed in the cleanup path
            mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_logs_interaction(self, cli_executor, cli_model_config, mock_subprocess_success):
        """Test that CLI interactions are logged."""
        with (
            patch("shutil.which", return_value="/usr/bin/gemini"),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec,
            patch("multi_mcp.models.cli_executor.log_llm_interaction") as mock_log,
        ):
            mock_exec.return_value = mock_subprocess_success

            await cli_executor.execute(
                canonical_name="gemini-cli",
                model_config=cli_model_config,
                messages=[{"role": "user", "content": "Test"}],
            )

            mock_log.assert_called_once()
            call_args = mock_log.call_args[1]
            assert call_args["request_data"]["model"] == "gemini-cli"
            assert call_args["request_data"]["cli"] is True
            assert "command" in call_args["request_data"]
            assert call_args["response_data"]["content"] == "Test response from CLI"
            assert call_args["response_data"]["status"] == "success"
