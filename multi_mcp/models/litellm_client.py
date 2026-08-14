"""LiteLLM client wrapper with config-based model resolution."""

import asyncio
import logging
import time
from typing import Any

import litellm
from litellm.exceptions import Timeout as LiteLLMTimeout

from multi_mcp.constants import DEFAULT_MAX_TOKENS
from multi_mcp.models.config import PROVIDERS, ModelConfig
from multi_mcp.models.resolver import ModelResolver
from multi_mcp.schemas.base import ModelResponse, ModelResponseMetadata
from multi_mcp.settings import settings
from multi_mcp.utils.error_humanizer import humanize_error, sanitize_for_log
from multi_mcp.utils.request_logger import log_llm_interaction

logger = logging.getLogger(__name__)

litellm.drop_params = True


def _extract_content_from_responses_api(response) -> str:
    """Extract text from responses API output array.

    Handles both OpenAI/Azure ([reasoning/web_search_call, message]) and Anthropic/Gemini ([message]).
    Supports both object and dict formats (LiteLLM's responses API can return either depending on
    provider/version). Earlier versions of this function used hasattr() to detect the output array,
    which silently fails for dicts (dicts have keys, not attributes) — fixed to use isinstance check.
    """
    # Extract output array — supports both object (response.output) and dict (response["output"]) shapes
    output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    if not output:
        logger.warning("[RESPONSE_PARSE] Response has no output or empty output array")
        return ""

    for item in output:
        # LiteLLM's responses API can return items as dicts or objects depending on provider/version
        # Handle both formats for 'type' and 'content' extraction
        item_type = item.get("type") if isinstance(item, dict) else getattr(item, "type", None)

        if item_type == "message":
            # Extract content (supports both dict and object formats)
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)

            # Handle None content
            if content is None:
                logger.debug("[RESPONSE_PARSE] Message item has None content, skipping")
                continue

            # Handle content as list of text items.
            # Use `or ""` to coerce both missing-key and None-value cases — dict.get("text", "")
            # returns None when the key exists with a None value, which would crash "".join().
            if isinstance(content, list):
                return "".join((c.get("text") or "") if isinstance(c, dict) else (getattr(c, "text", "") or "") for c in content if c)
            # Handle content as string (fallback)
            elif isinstance(content, str):
                return content
            else:
                # Unexpected content type
                logger.debug(f"[RESPONSE_PARSE] Unexpected content type '{type(content).__name__}' in message item")

    # No message item found in output
    logger.warning("[RESPONSE_PARSE] No message item found in response.output, returning empty string")
    return ""


def _extract_content_from_chat_completion(response) -> str:
    """Extract text from Chat Completions API response (used by Ollama and other non-Responses providers).

    Standard OpenAI chat-completion shape: response.choices[0].message.content
    Handles both object and dict formats (LiteLLM normalizes provider responses).
    Normalizes structured content (list of parts) to a string to satisfy the
    ModelResponse(content: str) contract; unexpected shapes log a debug message
    and return "".
    """
    choices = response.get("choices") if isinstance(response, dict) else getattr(response, "choices", None)
    if not choices:
        logger.warning("[RESPONSE_PARSE] Chat completion response has no choices")
        return ""

    first = choices[0]
    message = first.get("message") if isinstance(first, dict) else getattr(first, "message", None)
    if message is None:
        logger.warning("[RESPONSE_PARSE] Chat completion first choice has no message")
        return ""

    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Some providers return structured content as a list of parts (e.g. [{"type": "text", "text": "..."}]).
    # `or ""` defends against `{"text": null}` which would otherwise pass None into "".join() and crash.
    if isinstance(content, list):
        return "".join((c.get("text") or "") if isinstance(c, dict) else (getattr(c, "text", "") or "") for c in content if c)
    logger.debug(f"[RESPONSE_PARSE] Unexpected chat completion content type '{type(content).__name__}'")
    return ""


# Provider prefixes that must use Chat Completions API (litellm.acompletion).
# LiteLLM's Responses API (litellm.aresponses) only routes for providers that natively
# support it: OpenAI, Azure OpenAI, Anthropic, Gemini. Ollama (and others) only
# expose chat-completions, so calling aresponses() raises BadRequestError.
_CHAT_COMPLETION_PROVIDERS: frozenset[str] = frozenset({"ollama", "ollama_chat"})


class LiteLLMClient:
    """Wrapper for LiteLLM model calls with config-based resolution."""

    def __init__(self, resolver: ModelResolver | None = None):
        """Initialize LiteLLM client.

        Args:
            resolver: Optional ModelResolver instance. Creates one if not provided.
        """
        self._resolver: ModelResolver | None = resolver

    @property
    def resolver(self) -> ModelResolver:
        """Lazy-load resolver to avoid import-time config loading."""
        if self._resolver is None:
            self._resolver = ModelResolver()
        return self._resolver

    def _validate_provider_credentials(self, litellm_model: str) -> str | None:
        """Validate that required provider credentials are configured.

        Args:
            litellm_model: LiteLLM model string (e.g., "azure/gpt-5-mini", "gemini/gemini-2.5-flash")

        Returns:
            Error message if credentials missing, None if valid
        """
        # Extract provider from litellm_model (format: "provider/model-name")
        provider = litellm_model.split("/")[0].lower() if "/" in litellm_model else None
        if not provider:
            return None

        provider_config = PROVIDERS.get(provider)
        if not provider_config:
            return None

        missing = []
        present = []

        for attr, env_var in provider_config.credentials:
            if not getattr(settings, attr, None):
                missing.append(env_var)
            else:
                present.append(env_var)

        if not missing:
            return None

        msg = f"{provider_config.name} models require {' and '.join(missing)} to be set in environment or .env file"
        if present:
            msg += f" (already set: {', '.join(present)})"
        return msg

    def validate_model_credentials(self, litellm_model: str) -> str | None:
        """Public wrapper for credential validation (safe to call from other modules).

        Args:
            litellm_model: LiteLLM model string (e.g., "azure/gpt-5-mini", "gemini/gemini-2.5-flash")

        Returns:
            Error message if credentials missing, None if valid
        """
        return self._validate_provider_credentials(litellm_model)

    async def execute(
        self,
        canonical_name: str,
        model_config: ModelConfig,
        messages: list[dict],
        enable_web_search: bool = False,
    ) -> ModelResponse:
        """Execute LiteLLM API call (API models only).

        Args:
            canonical_name: Canonical model name (pre-resolved)
            model_config: Model configuration (pre-resolved)
            messages: List of message dicts with role and content
            enable_web_search: Enable provider-native web search if supported

        Returns:
            ModelResponse with status, content, metadata, error

        Note:
            Model should already be resolved by the caller.
            CLI models should be routed through CLIExecutor.
        """
        # Start time captured BEFORE the try block so exception handlers can compute
        # latency_ms for observability (parity with cli_executor.py error paths).
        start_time = time.perf_counter()
        try:
            # Reject CLI models - they should be routed elsewhere
            if model_config.is_cli_model():
                error_msg = f"Model '{canonical_name}' is a CLI model. Use CLIExecutor for CLI models."
                logger.error(f"[MODEL_CALL] {error_msg}")
                return ModelResponse.error_response(
                    error=error_msg,
                    model=canonical_name,
                )

            # API model execution (existing logic)
            timeout = settings.model_timeout_seconds
            litellm_model = model_config.litellm_model

            # Validate we have a litellm_model for API calls
            if not litellm_model:
                error_msg = f"Model '{canonical_name}' has no litellm_model configured"
                logger.error(f"[MODEL_CALL] {error_msg}")
                return ModelResponse.error_response(
                    error=error_msg,
                    model=canonical_name,
                )

            # Validate provider credentials before making API call
            credential_error = self._validate_provider_credentials(litellm_model)
            if credential_error:
                logger.error(f"[MODEL_CALL] Credential validation failed for {litellm_model}: {credential_error}")
                return ModelResponse.error_response(
                    error=credential_error,
                    model=canonical_name,
                )

            # Apply temperature (config constraint > default)
            temp = settings.default_temperature
            if model_config.constraints and model_config.constraints.temperature is not None:
                temp = model_config.constraints.temperature

            # Detect provider to choose API path.
            # Ollama (and similar providers) only support Chat Completions, not Responses API.
            provider_prefix = litellm_model.split("/", 1)[0].lower() if "/" in litellm_model else ""
            use_chat_completion = provider_prefix in _CHAT_COMPLETION_PROVIDERS
            api_path = "acompletion" if use_chat_completion else "aresponses"

            logger.info(f"[MODEL_CALL] canonical={canonical_name} litellm={litellm_model} temp={temp} api={api_path}")

            # Build kwargs starting with generic params from config.
            # Key difference: Chat Completions uses "messages", Responses API uses "input".
            kwargs: dict[str, Any] = {
                **model_config.params,
                "model": litellm_model,
                "temperature": temp,
                "num_retries": settings.max_retries,
                "timeout": timeout,
            }
            if use_chat_completion:
                kwargs["messages"] = messages
                # Explicit api_base if user configured it. LiteLLM also reads OLLAMA_API_BASE
                # from env (set in settings.set_provider_env_vars), but explicit beats implicit.
                if provider_prefix in {"ollama", "ollama_chat"} and settings.ollama_api_base:
                    kwargs["api_base"] = settings.ollama_api_base
            else:
                kwargs["input"] = messages

            # Set output token limit: config value > sensible default.
            # Per LiteLLM docs, the Responses API uses `max_output_tokens` while Chat Completions
            # uses `max_tokens`. Passing `max_tokens` to aresponses() risks silent drop because
            # `litellm.drop_params=True` is set globally, which would leave output uncapped.
            max_output = model_config.max_tokens if model_config.max_tokens is not None else DEFAULT_MAX_TOKENS
            max_tokens_param = "max_tokens" if use_chat_completion else "max_output_tokens"
            kwargs[max_tokens_param] = max_output
            logger.debug(
                f"[MODEL_CALL] Using {max_tokens_param}={max_output} ({'config' if model_config.max_tokens is not None else 'default'})"
            )

            # Enable provider-native web search if requested and supported.
            # Only Responses-API providers (OpenAI, Azure, Anthropic, Gemini) support the unified
            # web_search tool; Chat-Completions providers (Ollama, etc.) don't have this concept.
            # Final gate is model_config.has_provider_web_search() — per-model opt-in via YAML.
            if not use_chat_completion and enable_web_search and model_config.has_provider_web_search():
                kwargs["tools"] = [{"type": "web_search"}]
                logger.info(f"[WEB_SEARCH] Enabled for model: {canonical_name}")

            logger.debug(f"[MODEL_REQUEST] litellm_model={litellm_model} num_messages={len(messages)} api={api_path}")

            # Call LiteLLM via the appropriate API for the provider.
            # `raw_response` is the LiteLLM ModelResponse object; we keep it distinct from
            # our own `model_response` constructed below to avoid the shadowing trap where
            # log_llm_interaction would log the wrong shape if reordered.
            # Note: start_time is captured at method entry (before try) so exception handlers
            # can compute latency_ms — re-capturing here would lose pre-API setup time
            # (credential validation, kwargs building) from the success-path latency, which
            # is a minor observability tradeoff we accept for parity with exception handlers.
            if use_chat_completion:
                raw_response = await asyncio.wait_for(litellm.acompletion(**kwargs), timeout=timeout)
                content = _extract_content_from_chat_completion(raw_response)
            else:
                raw_response = await asyncio.wait_for(litellm.aresponses(**kwargs), timeout=timeout)
                content = _extract_content_from_responses_api(raw_response)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Treat empty content as an error rather than a silent success.
            # Both extractors return "" defensively for malformed responses (missing output/choices,
            # message item not found, etc.); without this guard, callers would receive
            # status="success" with no content and silently produce bad reviews/empty answers.
            if not content.strip():
                error_msg = f"Model '{canonical_name}' returned empty content via {api_path}"
                logger.error(f"[MODEL_CALL] {error_msg}")
                return ModelResponse.error_response(error=error_msg, model=canonical_name)

            # Extract usage stats. Both APIs expose response.usage.total_tokens
            # (chat completions also has prompt/completion split, but we only need the total).
            # LiteLLM can return response as either an object or a raw dict depending on provider,
            # so handle both shapes. Also fall back to prompt+completion sum when total isn't set,
            # which some chat-completion providers do.
            total_tokens = 0
            usage = raw_response.get("usage") if isinstance(raw_response, dict) else getattr(raw_response, "usage", None)
            if usage:
                if isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens") or ((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
                else:
                    total_tokens = getattr(usage, "total_tokens", None) or (
                        (getattr(usage, "prompt_tokens", 0) or 0) + (getattr(usage, "completion_tokens", 0) or 0)
                    )

            metadata = ModelResponseMetadata(
                model=canonical_name,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
            )
            model_response = ModelResponse(
                content=content,
                status="success",
                metadata=metadata,
            )

            log_llm_interaction(
                request_data={**kwargs},
                response_data=model_response.model_dump(),
            )
            return model_response

        except (TimeoutError, LiteLLMTimeout):
            # litellm.Timeout fires from LiteLLM's internal HTTP layer; TimeoutError fires
            # from our asyncio.wait_for wrapper. Either way, surface a consistent timeout message.
            # latency_ms tells operators how long the timeout actually took (parity with CLIExecutor).
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error(f"[MODEL_CALL] Model {canonical_name} timed out after {timeout}s")
            return ModelResponse.error_response(
                error=f"Request timed out after {timeout}s",
                model=canonical_name,
                latency_ms=latency_ms,
            )
        except Exception as e:
            # logger.exception captures the full traceback — critical for diagnosing
            # non-trivial LiteLLM failures (auth errors, malformed responses, network glitches).
            # Sanitize the message portion to redact any API keys LiteLLM may have echoed
            # in the exception text. The traceback itself is logged separately by logger.exception.
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            logger.exception(f"[MODEL_CALL] Model {canonical_name} failed: {sanitize_for_log(str(e))}")
            # humanize_error sanitizes (strips API keys, caps length) and converts known
            # failure patterns (Ollama auth, missing models, etc.) into actionable messages.
            return ModelResponse.error_response(
                error=humanize_error(str(e), canonical_name=canonical_name),
                model=canonical_name,
                latency_ms=latency_ms,
            )
