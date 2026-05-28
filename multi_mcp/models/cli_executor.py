"""CLI model execution via subprocess."""

import asyncio
import json
import logging
import os
import re
import shutil
import time

from multi_mcp.constants import DEBUG_LOG_MAX_LENGTH, ERROR_PREVIEW_MAX_LENGTH
from multi_mcp.models.config import ModelConfig
from multi_mcp.schemas.base import ModelResponse, ModelResponseMetadata
from multi_mcp.settings import settings
from multi_mcp.utils.error_humanizer import humanize_error, sanitize_for_log
from multi_mcp.utils.json_parser import parse_llm_json
from multi_mcp.utils.request_logger import log_llm_interaction

logger = logging.getLogger(__name__)


class CLIExecutor:
    """Executes CLI models via subprocess."""

    async def execute(
        self,
        canonical_name: str,
        model_config: ModelConfig,
        messages: list[dict],
        enable_web_search: bool = False,
    ) -> ModelResponse:
        """Execute CLI model via subprocess.

        Args:
            canonical_name: Canonical model name
            model_config: Model configuration
            messages: List of message dicts
            enable_web_search: Enable web search (ignored for CLI models)

        Returns:
            ModelResponse with CLI output
        """
        # Note: enable_web_search is ignored for CLI models (not supported)

        # Validate CLI command is set
        if not model_config.cli_command:
            error_msg = f"CLI model '{canonical_name}' has no cli_command configured"
            logger.error(f"[CLI_CALL] {error_msg}")
            return ModelResponse.error_response(
                error=error_msg,
                model=canonical_name,
            )

        # Narrow type for type checker - we know cli_command is str here
        cli_command: str = model_config.cli_command
        # Match by basename so absolute paths (`/opt/homebrew/bin/qwen`) and
        # wrapper scripts still resolve to "qwen" for credential gating below.
        cli_executable = os.path.basename(cli_command)

        # Build the subprocess environment FIRST so the cli_env PATH override
        # (if any) is honored by the shutil.which preflight below. Otherwise a
        # CLI that's only reachable via a custom PATH set in cli_env would be
        # rejected before we even try to launch it.
        env = os.environ.copy()

        # Per-CLI credential gating (defense-in-depth against supply-chain attacks).
        #
        # Threat model: a compromised CLI binary (e.g. malicious npm postinstall in
        # @anthropic-ai/claude-code or @openai/codex) can trivially exfiltrate
        # `process.env` via a fetch() — no shell access needed. By stripping ALL
        # known provider keys from the inherited env and re-injecting only the one
        # each CLI legitimately needs, we limit the blast radius to that one provider.
        #
        # This generalizes the existing qwen-only gating pattern to all CLIs.
        # Custom or wrapper CLIs (basename not in CLI_ALLOWED_KEYS) receive NO
        # provider credentials by default — users must opt them in explicitly via
        # `cli_env: {VAR: "${VAR}"}` in config.yaml.
        #
        # We also strip Azure and AWS credentials. These are API-only providers
        # (Bedrock, Azure OpenAI) with no CLI consumer in the current allowlist,
        # but Settings.set_provider_env_vars writes them to os.environ at startup,
        # so they would otherwise leak to every CLI subprocess.
        ALL_PROVIDER_KEYS = (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "OLLAMA_API_KEY",
            "LM_STUDIO_API_KEY",
            "DASHSCOPE_API_KEY",
            # Azure OpenAI (API-only, no CLI consumer today).
            # AZURE_API_VERSION and AZURE_API_BASE aren't credentials but they disclose
            # internal deployment shape (deployment URL, contract version) — strip them too.
            "AZURE_API_KEY",
            "AZURE_API_BASE",
            "AZURE_API_VERSION",
            # AWS Bedrock (API-only, no CLI consumer today). AWS_SESSION_TOKEN /
            # AWS_SECURITY_TOKEN are temporary STS credentials used with assumed roles —
            # useless without the access key pair we already strip, but defense-in-depth.
            # AWS_REGION_NAME isn't secret but discloses deployment region.
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_SECURITY_TOKEN",
            "AWS_REGION_NAME",
        )
        for key in ALL_PROVIDER_KEYS:
            env.pop(key, None)

        # Snapshot configured credentials. Pulled from Settings (the canonical
        # source) rather than re-reading the just-stripped env. Used both for
        # allowlist re-injection below AND for ${VAR} expansion in cli_env even
        # when a custom CLI is not on the allowlist.
        settings_secrets: dict[str, str] = {}
        if settings.anthropic_api_key:
            settings_secrets["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.openai_api_key:
            settings_secrets["OPENAI_API_KEY"] = settings.openai_api_key
        if settings.gemini_api_key:
            settings_secrets["GEMINI_API_KEY"] = settings.gemini_api_key
        if settings.openrouter_api_key:
            settings_secrets["OPENROUTER_API_KEY"] = settings.openrouter_api_key
        if settings.ollama_api_key:
            settings_secrets["OLLAMA_API_KEY"] = settings.ollama_api_key
        if settings.lm_studio_api_key:
            settings_secrets["LM_STUDIO_API_KEY"] = settings.lm_studio_api_key
        if settings.dashscope_api_key:
            settings_secrets["DASHSCOPE_API_KEY"] = settings.dashscope_api_key
        # Azure and AWS — only exposed if a user explicitly opts in via cli_env
        # (no CLI is on the allowlist for these today, but the values must be
        # in settings_secrets so ${AZURE_API_KEY} etc. can expand if requested).
        if settings.azure_api_key:
            settings_secrets["AZURE_API_KEY"] = settings.azure_api_key
        if settings.azure_api_base:
            settings_secrets["AZURE_API_BASE"] = settings.azure_api_base
        if settings.azure_api_version:
            settings_secrets["AZURE_API_VERSION"] = settings.azure_api_version
        if settings.aws_access_key_id:
            settings_secrets["AWS_ACCESS_KEY_ID"] = settings.aws_access_key_id
        if settings.aws_secret_access_key:
            settings_secrets["AWS_SECRET_ACCESS_KEY"] = settings.aws_secret_access_key
        if settings.aws_region_name:
            settings_secrets["AWS_REGION_NAME"] = settings.aws_region_name

        # Allowlist: which provider key(s) each first-party CLI legitimately needs.
        # Keep this list narrow — extending it without a real consumer use case
        # weakens the supply-chain defense.
        CLI_ALLOWED_KEYS: dict[str, tuple[str, ...]] = {
            "claude": ("ANTHROPIC_API_KEY",),
            "codex": ("OPENAI_API_KEY",),
            "gemini": ("GEMINI_API_KEY",),
            "qwen": ("OLLAMA_API_KEY", "LM_STUDIO_API_KEY", "DASHSCOPE_API_KEY"),
        }
        for allowed_key in CLI_ALLOWED_KEYS.get(cli_executable, ()):
            if allowed_key in settings_secrets:
                env[allowed_key] = settings_secrets[allowed_key]

        # Now expand variables in cli_env (e.g., ${ANTHROPIC_API_KEY}).
        # Build a stable lookup map that contains all cli_env keys upfront so
        # cross-key references resolve regardless of YAML insertion order.
        # Same-name variables in os.environ are still respected (process env
        # takes precedence on conflicts via dict merge order).
        #
        # Critical: include settings_secrets in the lookup so ${ANTHROPIC_API_KEY}
        # still expands even for a custom CLI that isn't on the allowlist — the
        # user opting in via cli_env should get the canonical Settings value, not
        # whatever happens to be in the just-stripped env.
        cli_env_lookup: dict[str, str] = {**model_config.cli_env, **settings_secrets, **env}
        for key, value in model_config.cli_env.items():
            expanded = self._expand_env_vars(value, cli_env_lookup)
            env[key] = expanded
            cli_env_lookup[key] = expanded

        # Check if CLI command exists — using the env we just built so any
        # PATH override from cli_env is taken into account.
        if not shutil.which(cli_command, path=env.get("PATH")):
            install_hint = self.get_install_hint(cli_command)
            error_msg = f"CLI command '{cli_command}' not found in PATH. {install_hint}"
            logger.error(f"[CLI_CALL] {error_msg}")
            return ModelResponse.error_response(
                error=error_msg,
                model=canonical_name,
            )

        # Serialize all messages with role markers so the system prompt and
        # conversation history reach the CLI agent. Previously this took only
        # `messages[-1]["content"]`, which silently dropped the system prompt
        # (e.g. codereview.md, chat.md) and any prior turns — meaning CLI
        # models were running without their tool-specific instructions and
        # multi-turn chat was effectively broken for CLI providers.
        prompt = self._format_messages_as_prompt(messages)

        # Build command
        command = [cli_command, *model_config.cli_args]

        # Use config timeout or fall back to settings
        timeout = settings.model_timeout_seconds

        # Sanitize command for logs — if a user puts a secret-shaped value in cli_args
        # (e.g. `--api-key=sk-...`), we don't want it persisted to logs/*.llm.json
        # or debug output. cli_command itself is just the executable name (safe).
        safe_command = [sanitize_for_log(arg) for arg in command]
        logger.info(f"[CLI_CALL] model={canonical_name} command={cli_command} parser={model_config.cli_parser}")
        logger.debug(f"[CLI_CALL] full_command={' '.join(safe_command)}")

        start_time = time.perf_counter()
        process: asyncio.subprocess.Process | None = None

        try:
            # Execute CLI subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(input=prompt.encode("utf-8")),
                timeout=timeout,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            if process.returncode != 0:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                stdout = stdout_bytes.decode("utf-8", errors="replace")

                # Use stderr if available, otherwise use stdout (some CLIs write errors to stdout)
                error_output = stderr if stderr else stdout
                error_preview = error_output[:ERROR_PREVIEW_MAX_LENGTH] if error_output else "(no output)"

                install_hint = self.get_install_hint(cli_command)
                logger.error(f"[CLI_CALL] {canonical_name} failed with exit code {process.returncode}")
                # Sanitize raw stderr/stdout before logging — a CLI in debug mode might
                # echo provider keys, Bearer tokens, or env-var assignments. logs/*.llm.json
                # would otherwise persist secrets even though the caller gets a humanized error.
                logger.debug(f"[CLI_CALL] stderr: {sanitize_for_log(stderr[:DEBUG_LOG_MAX_LENGTH])}")
                logger.debug(f"[CLI_CALL] stdout: {sanitize_for_log(stdout[:DEBUG_LOG_MAX_LENGTH])}")
                # humanize_error pattern-matches known CLI failures (e.g. codex "trusted dir",
                # missing CLI on PATH, auth errors) and returns an actionable message.
                # Falls through to the raw error preview + install hint when nothing matches.
                # We detect a pattern match by looking for the humanizer's marker string
                # "(Original error:" — equality check would fail on truncation/sanitization/empty input.
                humanized = humanize_error(error_preview or "", canonical_name=canonical_name)
                pattern_matched = "(Original error:" in humanized
                if not pattern_matched:
                    # CRITICAL: when no pattern matched, humanize_error returned the SANITIZED
                    # raw error (with secrets stripped). Use that — NEVER the raw error_preview,
                    # which may contain provider API keys, Bearer tokens, etc. if a misbehaving
                    # CLI echoes them in its stderr/stdout. Earlier versions of this code used
                    # error_preview here and would leak those secrets to the MCP caller.
                    # Special-case empty input: humanize_error returns "Unknown error..." but
                    # we prefer the cleaner "(no output)" wording in the failure message.
                    sanitized_preview = humanized if error_preview else "(no output)"
                    error_msg = (
                        f"CLI '{cli_command}' failed with exit code {process.returncode}. "
                        f"Error: {sanitized_preview}\n\n"
                        f"Troubleshooting: {install_hint}"
                    )
                else:
                    error_msg = humanized
                return ModelResponse.error_response(
                    error=error_msg,
                    model=canonical_name,
                    latency_ms=latency_ms,
                )

            # Parse output
            stdout = stdout_bytes.decode("utf-8", errors="replace")
            content = self._parse_output(stdout, model_config.cli_parser)

            # Treat empty/whitespace-only content as an error — parity with LiteLLMClient.
            # CLI tools that exit 0 but produce no useful output (filtered JSONL, empty
            # stdout, etc.) should not silently succeed with blank content, because
            # downstream consumers (codereview/debate) cannot distinguish "successful
            # empty answer" from "model failed to respond".
            if not content.strip():
                error_msg = f"CLI '{cli_command}' returned empty output"
                logger.error(f"[CLI_CALL] {error_msg}")
                return ModelResponse.error_response(
                    error=error_msg,
                    model=canonical_name,
                    latency_ms=latency_ms,
                )

            metadata = ModelResponseMetadata(
                model=canonical_name,
                total_tokens=0,
                latency_ms=latency_ms,
            )

            response = ModelResponse(
                content=content,
                status="success",
                metadata=metadata,
            )

            log_llm_interaction(
                request_data={
                    "model": canonical_name,
                    "cli": True,
                    "command": safe_command,  # sanitized version — see safe_command construction above
                    "prompt_length": len(prompt),
                },
                response_data=response.model_dump(),
            )

            return response

        except TimeoutError:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Clean up timed-out subprocess. We bound the drain with a 5s secondary
            # timeout because a child that ignores SIGKILL (or whose pipes never
            # drain) would otherwise hang here indefinitely — silently defeating
            # MODEL_TIMEOUT_SECONDS and tying up the worker.
            await self._terminate_subprocess(process, "timed-out")

            logger.error(f"[CLI_CALL] {canonical_name} timed out after {timeout}s")
            return ModelResponse.error_response(
                error=f"CLI '{cli_command}' timed out after {timeout}s. "
                f"The command took longer than expected. "
                f"Consider using a faster model or increasing MODEL_TIMEOUT_SECONDS in config.",
                model=canonical_name,
                latency_ms=latency_ms,
            )

        except FileNotFoundError as e:
            # This shouldn't happen due to shutil.which() check, but handle it anyway
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            install_hint = self.get_install_hint(cli_command)
            logger.error(f"[CLI_CALL] {canonical_name} command not found: {e}")
            return ModelResponse.error_response(
                error=f"CLI command '{cli_command}' not found. {install_hint}",
                model=canonical_name,
                latency_ms=latency_ms,
            )

        except ValueError as e:
            # _parse_output() raises ValueError for application-level errors that
            # the CLI reports inside its successful subprocess output (e.g. Claude
            # CLI returning {"is_error": true, "result": "..."}). Without this
            # explicit branch the generic Exception handler below would mask it
            # as "CLI execution failed: ValueError: Claude CLI error: ..." which
            # confusingly suggests a subprocess crash. The CLI actually ran fine —
            # it just reported a model-side error — so surface that cleanly.
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"[CLI_CALL] {canonical_name} returned application error: {sanitize_for_log(str(e))}")
            # Sanitize the error message — application errors may embed tokens
            # (e.g. a CLI echoing back the request that included a key).
            response = ModelResponse.error_response(
                error=humanize_error(str(e), canonical_name=canonical_name),
                model=canonical_name,
                latency_ms=latency_ms,
            )
            # Log the interaction so application errors are visible in logs/*.llm.json
            # alongside successful calls — parity with the success path above.
            log_llm_interaction(
                request_data={
                    "model": canonical_name,
                    "cli": True,
                    "command": safe_command,  # sanitized version — see safe_command construction above
                    "prompt_length": len(prompt),
                },
                response_data=response.model_dump(),
            )
            return response

        except asyncio.CancelledError:
            # Higher-level cancellation (client disconnect, parent task cancel,
            # etc.) — kill the child so we don't leave an orphaned subprocess
            # consuming CPU/memory after the parent task is gone, then re-raise
            # so the cancellation actually propagates.
            await self._terminate_subprocess(process, "cancelled")
            raise

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Clean up failed subprocess with a bounded drain (same rationale as
            # the TimeoutError branch — don't let kill()+communicate hang forever).
            await self._terminate_subprocess(process, "failed")

            logger.error(f"[CLI_CALL] {canonical_name} failed with exception: {type(e).__name__}: {sanitize_for_log(str(e))}")
            # Note: exc_info=True logs the full traceback including local variables in some
            # Python configurations. We accept this trade-off for debuggability — the traceback
            # itself rarely contains user secrets (those would be in subprocess output, sanitized above).
            logger.debug("[CLI_CALL] Full error details", exc_info=True)
            # Sanitize through humanize_error so any embedded secrets get redacted
            # and known failure patterns produce actionable messages.
            safe_error = humanize_error(f"{type(e).__name__}: {e!s}", canonical_name=canonical_name)
            return ModelResponse.error_response(
                error=f"CLI execution failed: {safe_error}",
                model=canonical_name,
                latency_ms=latency_ms,
            )

    @staticmethod
    def get_install_hint(cli_command: str) -> str:
        """Get installation hint for common CLI tools.

        Args:
            cli_command: CLI command name

        Returns:
            Installation hint string
        """
        # Verified package names (April 2026):
        # - gemini: https://www.npmjs.com/package/@google/gemini-cli
        # - codex:  https://github.com/openai/codex (npm @openai/codex)
        # - claude: https://github.com/anthropics/claude-code (npm @anthropic-ai/claude-code)
        # The previous hints were factually wrong and misled users.
        hints = {
            "gemini": "Install via: npm install -g @google/gemini-cli",
            "codex": "Install via: npm install -g @openai/codex",
            "claude": "Install via: npm install -g @anthropic-ai/claude-code",
            "qwen": "Install via: npm install -g @qwen-code/qwen-code@latest (or: brew install qwen-code)",
        }
        return hints.get(cli_command, f"Ensure '{cli_command}' is installed and in PATH")

    def _parse_output(self, stdout: str, parser_type: str) -> str:
        """Parse CLI output based on parser type.

        Args:
            stdout: Raw CLI stdout
            parser_type: "json", "jsonl", or "text"

        Returns:
            Parsed content string
        """
        if parser_type == "qwen-json":
            # Qwen Code CLI emits an array of message events ending in a
            # `type:"result"` event. Scoped to its own parser_type so generic
            # `cli_parser: json` (used by gemini, claude, and any custom user
            # CLI returning JSON arrays) keeps its original re-serialize
            # behavior unchanged.
            parsed = parse_llm_json(stdout)
            if parsed is not None and isinstance(parsed, list):
                return self._parse_qwen_event_array(parsed)
            if parsed is not None:
                # Defensive: qwen-json should always be a list. If not, fall
                # through to JSON serialization rather than crashing.
                logger.warning(
                    "[CLI_PARSE] qwen-json parser got non-list (%s); re-serializing",
                    type(parsed).__name__,
                )
                return json.dumps(parsed, ensure_ascii=False)
            logger.warning("[CLI_PARSE] qwen-json parse failed, falling back to text")
            return stdout.strip()

        if parser_type == "json":
            # Use existing robust JSON parser (handles malformed JSON)
            parsed = parse_llm_json(stdout)
            if parsed is not None:
                # Extract content based on CLI format
                if isinstance(parsed, dict):
                    # Claude CLI format: check for errors first
                    # {"type":"result","is_error":true/false,"result":"content"}
                    if parsed.get("is_error"):
                        # Claude CLI returned an error
                        error_msg = parsed.get("result", "Unknown error from Claude CLI")
                        raise ValueError(f"Claude CLI error: {error_msg}")

                    # Gemini CLI format: {"response": "content"}
                    # Claude CLI format: {"result": "content"}
                    # Both fields are normally strings, but we defensively handle
                    # nested objects/arrays by re-serializing as JSON so the
                    # return type stays `str` and downstream consumers don't get
                    # a surprise dict/list (which would then hit Python repr).
                    for key in ("response", "result"):
                        if key in parsed:
                            val = parsed[key]
                            if isinstance(val, str):
                                return val
                            return json.dumps(val, ensure_ascii=False)
                # Non-dict or unknown-shape JSON (lists, scalars, dict without
                # known content keys): re-serialize as JSON rather than using
                # Python repr via str(). str(parsed) produces single-quoted
                # output that downstream JSON consumers cannot re-parse.
                return json.dumps(parsed, ensure_ascii=False)
            else:
                logger.warning("[CLI_PARSE] JSON parse failed, falling back to text")
                return stdout.strip()

        elif parser_type == "jsonl":
            # Parse JSONL (one JSON per line, extract text from events)
            lines = stdout.strip().split("\n")
            messages = []
            for line in lines:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    # Handle different event types
                    if event.get("type") == "text":
                        text = event.get("text", "")
                        if text:  # Skip empty text fields
                            messages.append(text)
                    elif event.get("type") == "item.completed":
                        # Codex format: extract text from item
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            text = item.get("text", "")
                            if text:
                                messages.append(text)
                except json.JSONDecodeError:
                    continue
            return "\n".join(messages) if messages else stdout.strip()

        else:  # "text" or fallback
            return stdout.strip()

    @staticmethod
    def _parse_qwen_event_array(events: list) -> str:
        """Extract the final answer from a Qwen Code `--output-format json` array.

        Qwen emits a heterogeneous event list (session_start, assistant turns,
        tool calls, ..., then a terminal `type:"result"` event). The contract
        is: pick the last `type:"result"` and read its `result` field; on
        `is_error: true` raise so the caller surfaces it as an application
        error (parity with the Claude `is_error` branch in the dict path).

        If the result event is missing or empty (defensive — docs around
        partial-content events are inconsistent across versions), fall back to
        the last assistant message's text parts. Re-serialize the whole array
        as the final fallback so the raw output remains debuggable.
        """
        # Pass 1: find the terminal result event and try to use it
        result_event: dict | None = None
        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "result":
                result_event = event
                break

        if result_event is not None:
            if result_event.get("is_error"):
                # Error messages may surface under several keys depending on the
                # failure mode (auth, tool, model). Try them in priority order
                # so users see something specific rather than "Unknown error".
                for key in ("result", "error", "message", "subtype"):
                    val = result_event.get(key)
                    if isinstance(val, str) and val:
                        raise ValueError(f"Qwen CLI error: {val}")
                raise ValueError("Qwen CLI error: unknown failure (no error message in result event)")

            result_val = result_event.get("result")
            if isinstance(result_val, str) and result_val:
                return result_val
            if result_val is not None and not isinstance(result_val, str):
                # Defensive: result is a dict/list (shouldn't happen per docs)
                return json.dumps(result_val, ensure_ascii=False)
            # result is empty/None — fall through to assistant fallback

        # Pass 2: last assistant message text (in case result event is missing
        # or has an empty result — e.g., some headless modes only emit assistant
        # turns without a terminal result event).
        for event in reversed(events):
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, list):
                text_parts = [
                    part["text"]
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                ]
                joined = "\n".join(t for t in text_parts if t)
                if joined:
                    return joined

        # Last resort: return empty so the empty-content guard in execute()
        # catches it and produces a proper error response. The raw events are
        # still visible in server logs via the warning below for diagnosis.
        # (Earlier versions returned json.dumps(events) here, which would slip
        # through the empty-content check as "successful" content equal to "[...]"
        # — meaningless to downstream consumers and indistinguishable from a real answer.)
        logger.warning(
            "[CLI_PARSE] Qwen event array had no usable result or assistant text; raw events: %s",
            json.dumps(events, ensure_ascii=False)[:DEBUG_LOG_MAX_LENGTH],
        )
        return ""

    @staticmethod
    async def _terminate_subprocess(process: asyncio.subprocess.Process | None, reason: str) -> None:
        """Best-effort kill + bounded drain of a subprocess pipe.

        Used by all execute() exception branches (timeout, cancellation, generic
        failure). Guards against:
        - exit-after-check race: process may exit between the returncode check
          and kill(), causing ProcessLookupError — caught and logged
        - hanging communicate(): wrapped in asyncio.wait_for(timeout=5) so a
          child that ignores SIGKILL or whose pipes never drain cannot defeat
          MODEL_TIMEOUT_SECONDS

        Never raises — cleanup must not mask the exception that triggered it.
        """
        if process is None or process.returncode is not None:
            return
        try:
            process.kill()
        except ProcessLookupError:
            logger.debug(f"[CLI_CALL] {reason} CLI process exited before kill()")
        except Exception:
            logger.debug(f"[CLI_CALL] Error sending kill() to {reason} CLI process", exc_info=True)
        try:
            await asyncio.wait_for(process.communicate(), timeout=5.0)
        except TimeoutError:
            logger.warning(f"[CLI_CALL] {reason} CLI process did not exit cleanly within 5s after kill()")
        except Exception:
            logger.debug(f"[CLI_CALL] Error while draining {reason} CLI process", exc_info=True)

    @staticmethod
    def _format_messages_as_prompt(messages: list[dict]) -> str:
        """Serialize a list of role/content message dicts into a single prompt string.

        CLI agents (claude, codex, gemini) consume their prompt via stdin as a single
        string — they have no native concept of a multi-message conversation array
        the way the OpenAI/Anthropic chat APIs do. We therefore flatten the message
        list into a labeled transcript so that the system prompt and earlier turns
        are preserved end-to-end.

        Format:
            [SYSTEM]
            <system content>

            [USER]
            <user content>

            [ASSISTANT]
            <assistant content>

            ...

        Content normalization (parity with LiteLLM/Anthropic message formats):
            - None                     → ""
            - str                      → as-is
            - list of parts            → concatenated `text` fields of dict parts
              (multimodal messages drop image/tool parts — CLI agents can't render them)
            - anything else            → json.dumps(..., ensure_ascii=False, default=str)
              (produces valid JSON rather than Python repr with single quotes)

        Empty messages list yields an empty string.
        """
        if not messages:
            return ""
        parts: list[str] = []
        for msg in messages:
            role_raw = msg.get("role")
            role = str(role_raw if role_raw else "user").upper()
            content = CLIExecutor._normalize_content(msg.get("content"))
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_content(content: object) -> str:
        """Coerce heterogeneous message content into a plain string.

        Mirrors the shapes that LiteLLM / Anthropic / OpenAI APIs produce so that
        CLI agents see equivalent input to API models. See _format_messages_as_prompt
        for the normalization contract.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            # Multimodal / tool-use: keep only text parts, drop images/tool calls
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                # Anthropic / OpenAI style: {"type": "text", "text": "..."}
                elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return "\n".join(text_parts)
        # Fallback: serialize as JSON (not Python repr — json.dumps uses double quotes
        # and is re-parseable downstream). default=str handles datetimes, Paths, etc.
        try:
            return json.dumps(content, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            # Last-resort fallback for truly unserializable content
            return str(content)

    def _expand_env_vars(self, value: str, env: dict[str, str]) -> str:
        """Expand environment variables in a string using provided env dict.

        Handles ${VAR_NAME} syntax. Uses the provided env dict instead of os.environ
        so that variables injected from settings are properly expanded.

        Supports nested references (${FOO} where FOO=${BAR}) by iterating until
        the substitution is stable, bounded to 5 passes to avoid infinite loops
        on reference cycles.

        Args:
            value: String that may contain ${VAR_NAME} patterns
            env: Environment dict to use for variable lookup

        Returns:
            String with variables expanded
        """

        def replacer(match: re.Match[str]) -> str:
            var_name = match.group(1)
            result = env.get(var_name)
            # If variable not found in env, return original ${VAR} syntax
            return result if result is not None else match.group(0)

        for _ in range(5):
            new_value = re.sub(r"\$\{([^}]+)\}", replacer, value)
            if new_value == value:
                break
            value = new_value
        return value
