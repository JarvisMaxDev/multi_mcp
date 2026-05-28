"""Convert raw provider/CLI errors into actionable messages.

When a model call fails because the user isn't authenticated (or trusted, or signed in),
the raw error from LiteLLM / a CLI tool is usually cryptic — e.g. `BadRequestError: 401`
or `Not inside a trusted directory and --skip-git-repo-check was not specified`.

This module pattern-matches known failure modes and replaces them with a short
"here's what to do" message so the user doesn't have to interpret stack traces.

Patterns are intentionally narrow: we only humanize errors we recognize. Unknown errors
pass through unchanged so we don't accidentally mask new failure modes.
"""

import re
from functools import lru_cache
from typing import Final

# All credential env vars we know about — read from os.environ in addition to Settings
# so the value-based redaction layer catches secrets injected after Settings init or
# directly via shell rc files that bypass Pydantic Settings.
_RUNTIME_SECRET_ENV_VARS: Final[tuple[str, ...]] = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "OLLAMA_API_KEY",
    "LM_STUDIO_API_KEY",
    "DASHSCOPE_API_KEY",
    "AZURE_API_KEY",
    "AZURE_API_BASE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN",
)

# Sanitization: cap message length and strip common secret patterns before surfacing.
# These never need to leak to the AI assistant caller.
_MAX_ERROR_LENGTH: Final[int] = 500
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    # OpenAI-style keys: sk-..., sk-proj-...
    # Also covers sk-ant-... (Anthropic) since sk-ant-* is a strict subset of sk-*.
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}"),
    # Generic Bearer tokens (Authorization: Bearer ...)
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]{20,}", re.IGNORECASE),
    # Google API keys (Gemini, Maps, etc.): AIza followed by 35 chars
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    # AWS access key IDs: AKIA (permanent) or ASIA (temporary STS) + 16 uppercase alphanumerics
    re.compile(r"(?:AKIA|ASIA)[0-9A-Z]{16}"),
    # Provider env-var assignments in error messages — catches strings like
    # `ANTHROPIC_API_KEY=sk-...`, `AZURE_API_KEY: '...'`, etc. for any of our
    # known provider env vars. Matches up to the next whitespace, quote, or
    # punctuation that typically delimits an assignment value.
    re.compile(
        r"['\"]?(?:OPENAI|ANTHROPIC|GEMINI|OPENROUTER|OLLAMA|LM_STUDIO|"
        r"DASHSCOPE|AZURE|AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|"
        r"AWS_SESSION_TOKEN|AWS_SECURITY_TOKEN)"
        r"(?:_API_KEY|_API_BASE)?['\"]?\s*[:=]\s*['\"]?[^'\"\s,;}]+",
        re.IGNORECASE,
    ),
)


@lru_cache(maxsize=1)
def _get_runtime_secret_values() -> tuple[str, ...]:
    """Return all live secret values that should be redacted by exact-match string replace.

    The regex patterns in _SECRET_PATTERNS catch secrets by SHAPE (sk-*, Bearer, AIza, AKIA/ASIA)
    or by ASSIGNMENT FORM (KEY=value). But AWS secret access keys (40 base64 chars) and AWS STS
    session tokens (long base64) have no recognizable prefix — if a CLI echoes just the bare
    value, the regex layer misses it. This function returns the live configured values from
    Settings + os.environ so _sanitize can do an exact string replace as a defense-in-depth
    second pass. Empty/short values are filtered (avoids false-positive redactions on common words).

    Reads BOTH os.environ (catches shell-injected values, post-Settings-init mutations) and
    Settings (canonical source for Pydantic-managed credentials). Returns values sorted
    longest-first so a shorter secret that is a substring of another can't shred the longer
    match before it gets a chance to replace.

    Cached via @lru_cache(maxsize=1) — Settings is process-lifetime immutable, so a single
    snapshot is correct. Tests that need to mutate Settings/env must call
    _get_runtime_secret_values.cache_clear() in setup.

    Lazy import of Settings to avoid circular import (settings.py is loaded by every consumer
    of this module). Narrow ImportError catch — real Settings validation errors should surface.
    """
    import os

    # Read all known credential env vars from os.environ first (broader coverage than just STS).
    values: list[str | None] = [os.environ.get(key) for key in _RUNTIME_SECRET_ENV_VARS]

    try:
        from multi_mcp.settings import settings

        # Include ALL configured provider secrets — even those covered by regex patterns,
        # because a CLI could echo a partial slice or unusual format that misses the regex.
        values.extend(
            [
                settings.anthropic_api_key,
                settings.openai_api_key,
                settings.gemini_api_key,
                settings.openrouter_api_key,
                settings.ollama_api_key,
                settings.lm_studio_api_key,
                settings.dashscope_api_key,
                settings.azure_api_key,
                settings.aws_access_key_id,
                settings.aws_secret_access_key,
            ]
        )
    except ImportError:
        # Settings module unreachable (e.g. during partial test import) — regex layer still applies.
        # Narrow catch: real Settings validation errors (Pydantic ValidationError, AttributeError
        # from renamed fields, etc.) should surface as bugs, not be silently swallowed.
        pass

    # Filter: drop None/empty + values too short to be credentials (avoids redacting common words).
    # Dedupe via dict to preserve insertion order (but final sort discards order anyway).
    # Sort longest-first: prevents a shorter secret that's a substring of a longer one from
    # shredding the longer match (e.g. rotated keys sharing a prefix → partial leakage).
    unique = {v: None for v in values if v and len(v) >= 12}.keys()
    return tuple(sorted(unique, key=len, reverse=True))


def _sanitize(error: str) -> str:
    """Remove credentials and cap length so we never leak secrets into error messages.

    Two-pass redaction:
    1. Exact-match replace of any live configured secret value (covers bare unprefixed values
       like AWS secret keys / STS tokens which the regex layer can't catch by shape).
       Values are pre-sorted longest-first by _get_runtime_secret_values to prevent
       substring overlap from leaving partial leakage.
    2. Regex pattern replace (covers structured forms: sk-*, Bearer, AIza, AKIA/ASIA, env-assigns)
    """
    for secret_value in _get_runtime_secret_values():
        error = error.replace(secret_value, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        error = pattern.sub("[REDACTED]", error)
    if len(error) > _MAX_ERROR_LENGTH:
        error = error[:_MAX_ERROR_LENGTH] + "... [truncated]"
    return error


def sanitize_for_log(raw: str) -> str:
    """Redact secrets and cap length without applying friendly pattern rewrites.

    Intended for use in `logger.debug(...)`, `logger.error(...)`, etc. where we
    want the original error context for debugging but must not persist credentials.
    Unlike `humanize_error`, this never adds "(Original error: ...)" wrapping —
    just strips secrets and truncates.
    """
    if not raw:
        return ""
    return _sanitize(raw)


# Each rule: (pattern → friendly message generator).
# The generator receives the canonical model name and the original (sanitized) error.
# Patterns match against the lower-cased error message for robustness.
_AuthRule = tuple[re.Pattern[str], str]


_HUMANIZE_RULES: Final[tuple[_AuthRule, ...]] = (
    # --- Codex CLI ---
    (
        re.compile(r"not inside a trusted directory", re.IGNORECASE),
        (
            "Codex CLI refuses to run here because the current directory is not a trusted git repo. "
            "Either (a) cd into a git-tracked project, (b) trust this dir via `codex auth trust`, "
            "or (c) add `--skip-git-repo-check` to the codex-cli `cli_args` in ~/.multi_mcp/config.yaml. "
            "(multi-mcp already runs codex in `--sandbox read-only` mode so skipping the git check is safe.)"
        ),
    ),
    (
        re.compile(r"codex(:\s*command|.*not found|.*: no such file)", re.IGNORECASE),
        "Codex CLI is not installed or not on PATH. Install with `npm install -g @openai/codex` and verify with `which codex`.",
    ),
    # --- Claude Code CLI ---
    (
        re.compile(
            # Tightened from a bare `401` (which matched any string with "401" — port
            # numbers, request IDs, etc.) to require auth context: 401 followed by
            # Unauthorized/Authentication (with whitespace OR colon separator like
            # "401: Unauthorized"), or explicit auth-error wording.
            r"(invalid api key|api[_ ]?key.*not.*found|authentication[_ ]?error|"
            r"\b401[\s:]+(unauthorized|authentication)|unauthorized.*401)",
            re.IGNORECASE,
        ),
        (
            "Authentication failed. For CLI tools (claude/codex/gemini): run the tool's login command "
            "(e.g. `claude login`, `codex auth login`, `gemini auth login`). "
            "For API providers: check the corresponding *_API_KEY in your .env."
        ),
    ),
    (
        re.compile(r"claude.*command not found|claude.*: no such file", re.IGNORECASE),
        "Claude Code CLI is not installed or not on PATH. Install with `npm install -g @anthropic-ai/claude-code`.",
    ),
    # --- Ollama (local server + cloud models) ---
    (
        re.compile(r"(connection refused|connection error).*(11434|localhost|ollama)", re.IGNORECASE),
        (
            "Ollama daemon is not running. Start it with `ollama serve` (or check OLLAMA_API_BASE). "
            "Verify with `curl http://localhost:11434/api/tags`."
        ),
    ),
    (
        re.compile(r"(ollama.*signin|model.*requires.*signin|cloud.*signin)", re.IGNORECASE),
        (
            "Ollama cloud model requires authentication. Run `ollama signin` in a terminal "
            "to link your local daemon to your Ollama cloud account, then retry."
        ),
    ),
    (
        # Require Ollama context — otherwise the regex would match any provider's
        # NotFoundError (e.g. OpenAI "Model gpt-5-mini not found") and give wrong
        # "ollama pull" advice. `pull model first` and `no such model` are Ollama-specific.
        re.compile(r"(ollama.*model .*not found|model .*not found.*ollama|pull model first|no such model)", re.IGNORECASE),
        (
            "Ollama model is not pulled locally. Run `ollama pull <model-name>` "
            "(use the exact name from `ollama list`), then retry. "
            "Cloud models (`:cloud` suffix) require `ollama signin` instead of pull."
        ),
    ),
    # --- Gemini CLI ---
    (
        re.compile(r"gemini.*command not found|gemini.*: no such file", re.IGNORECASE),
        "Gemini CLI is not installed or not on PATH. Install via https://github.com/google-gemini/gemini-cli.",
    ),
    # --- Generic LiteLLM provider misconfig ---
    (
        re.compile(r"LLM Provider NOT provided", re.IGNORECASE),
        (
            "LiteLLM couldn't determine the provider for this model. The model alias likely doesn't resolve "
            "to a known config entry. Check `~/.multi_mcp/config.yaml` and verify the alias is listed. "
            "Run `/multi:models` to see available aliases."
        ),
    ),
)


def humanize_error(raw_error: str, canonical_name: str | None = None) -> str:
    """Return an actionable error message for known failure patterns, else the sanitized raw error.

    Args:
        raw_error: The original error string from LiteLLM / a CLI subprocess.
        canonical_name: Resolved model name (used to enrich messages if provided).

    Returns:
        Either a friendly multi-line "what to do" message, or the sanitized raw error
        if no pattern matched. Always safe to surface to the caller — no credentials.
    """
    if not raw_error:
        return "Unknown error (empty error message from provider)"

    sanitized = _sanitize(raw_error)

    for pattern, friendly in _HUMANIZE_RULES:
        if pattern.search(sanitized):
            prefix = f"[{canonical_name}] " if canonical_name else ""
            return f"{prefix}{friendly}\n\n(Original error: {sanitized[:200]})"

    return sanitized
