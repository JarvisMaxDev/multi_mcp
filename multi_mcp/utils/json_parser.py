"""Robust JSON extraction and parsing for LLM responses.

This module provides utilities to extract and parse JSON from LLM responses
that may contain markdown formatting, comments, or other non-standard JSON.
"""

import json
import re
from typing import Any

# Greedy match (`*` not `*?`) so this handles nested code fences inside the
# JSON content (e.g. a "fix" field that contains ```python ... ```). The
# greedy version locks onto the OUTERMOST opening + closing pair instead of
# truncating at the first inner `````. The fallback in _strip_code_fences
# uses this for the trailing-text-after-fence case where the
# end-of-string-anchored greedy pattern doesn't match.
_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*([\s\S]*)\s*```", re.IGNORECASE)

_ANALYSIS_BLOCK_RE = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)

_COMMENT_RE = re.compile(
    r"""
    //.*?$           |   # line comments
    /\*[\s\S]*?\*/       # block comments
    """,
    re.MULTILINE | re.VERBOSE,
)

_UNQUOTED_KEY_RE = re.compile(r"(?P<prefix>[{\[,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_\-]*)(?P<suffix>\s*:)")

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

_STRING_RE = re.compile(r'"(?:[^"\\]|\\.)*"')

_SMART_QUOTES = {
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',  # double quotes
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",  # single quotes
}


def _strip_code_fences(s: str) -> str:
    """Strip markdown code fences from string.

    Handles both complete fences (```json ... ```) and unclosed fences
    where the LLM was cut off mid-response (```json ...).

    IMPORTANT: Uses greedy matching to handle nested code fences inside JSON strings.
    For example, if the JSON contains a "fix" field with code snippets that have their
    own ```python ... ``` fences, we want to match the OUTERMOST fences, not the first
    closing ``` we encounter.
    """
    # Primary: greedy match anchored at end-of-string (`\s*$`). When the
    # closing ``` is the very last thing in the input, this locks onto the
    # OUTERMOST fence pair without being tricked by any inner ```python```
    # fences the JSON content may carry in its string fields.
    greedy_pattern = re.compile(r"```(?:json|JSON)?\s*([\s\S]*)\s*```\s*$", re.IGNORECASE)
    m = greedy_pattern.search(s)
    if m:
        return m.group(1)

    # Fallback: greedy match WITHOUT the end-of-string anchor. Used when the
    # LLM added trailing prose after the closing fence. Still greedy (see
    # _CODE_FENCE_RE definition — round-4 made it greedy because the previous
    # non-greedy version truncated at the first inner ```python``` fence).
    m = _CODE_FENCE_RE.search(s)
    if m:
        return m.group(1)

    # Check for unclosed fence (opening ``` but no closing ```)
    unclosed_match = re.search(r"```(?:json|JSON)?\s*([\s\S]+)", s, re.IGNORECASE)
    if unclosed_match:
        return unclosed_match.group(1)

    return s


def _strip_analysis_blocks(s: str) -> str:
    """Remove <analysis>...</analysis> blocks from string."""
    return re.sub(_ANALYSIS_BLOCK_RE, "", s)


def _strip_comments(s: str) -> str:
    """Remove line and block comments from string."""
    return re.sub(_COMMENT_RE, "", s)


def _normalize_quotes(s: str) -> str:
    """Replace smart quotes with standard quotes."""
    for k, v in _SMART_QUOTES.items():
        s = s.replace(k, v)
    return s


def _convert_single_to_double_quotes(s: str) -> str:
    """Convert single-quoted strings to double-quoted strings.

    JSON requires double quotes for strings. This handles LLM responses
    that use single quotes instead.

    Approach:
    - Track whether we're currently inside a valid double-quoted string; if so,
      copy apostrophes verbatim so `"it's fine"` is preserved intact.
    - Outside double-quoted context, single quotes start a single-quoted string
      scan that collects content, escapes any unescaped double quotes, and
      re-wraps the result in double quotes.
    """
    result = []
    i = 0
    length = len(s)
    in_double_quotes = False
    dq_escaped = False

    while i < length:
        char = s[i]

        # Inside a double-quoted string: preserve everything verbatim (including
        # apostrophes and backslash escapes). This is the fix for `"it's fine"`.
        if in_double_quotes:
            result.append(char)
            if dq_escaped:
                dq_escaped = False
            elif char == "\\":
                dq_escaped = True
            elif char == '"':
                in_double_quotes = False
            i += 1
            continue

        # Outside double-quoted context:
        if char == '"':
            # Entering a double-quoted string
            result.append(char)
            in_double_quotes = True
            i += 1
            continue

        # Check if we're at the start of a single-quoted string
        if char == "'":
            # Collect the string content
            string_start = i
            i += 1
            string_content = []
            escaped = False

            while i < length:
                if escaped:
                    # After escape, add the char literally
                    string_content.append(s[i])
                    escaped = False
                elif s[i] == "\\":
                    # Escape sequence - keep the backslash
                    string_content.append("\\")
                    escaped = True
                elif s[i] == "'":
                    # End of single-quoted string
                    # Escape any unescaped double quotes in the content
                    content_str = "".join(string_content)
                    # Replace unescaped " with \"
                    content_str = content_str.replace('"', '\\"')
                    # Build double-quoted string
                    result.append('"' + content_str + '"')
                    i += 1
                    break
                else:
                    string_content.append(s[i])
                i += 1
            else:
                # Unclosed single quote - just keep original
                result.append(s[string_start:i])
        else:
            result.append(char)
            i += 1

    return "".join(result)


def _mask_strings(s: str) -> tuple[str, dict[str, str]]:
    """Mask string literals with placeholders to protect during repairs.

    This prevents regex-based repairs from corrupting content inside JSON strings.
    For example, URLs with '//' won't be treated as comments, and literal words
    like 'None' won't be replaced with 'null'.

    Returns:
        tuple of (masked_string, placeholder_map)
    """
    strings = {}
    counter = 0

    def replace_string(match):
        nonlocal counter
        placeholder = f"@STR_{counter}@"
        strings[placeholder] = match.group(0)
        counter += 1
        return placeholder

    masked = _STRING_RE.sub(replace_string, s)
    return masked, strings


_PLACEHOLDER_RE = re.compile(r"@STR_\d+@")


def _unmask_strings(s: str, strings: dict[str, str]) -> str:
    """Restore masked string literals.

    Uses a single-pass ``re.sub`` instead of a loop of ``s.replace()`` calls:
    sequential replacement would corrupt data if a restored string happened to
    contain a substring matching a placeholder we haven't processed yet (e.g.
    the restored string contains the literal text ``@STR_2@`` because an LLM
    echoed an earlier debug message back). Single-pass scanning walks the
    string once and only substitutes real placeholders, never touching content
    that was just restored.

    Args:
        s: String with placeholders
        strings: Map of placeholders to original strings

    Returns:
        String with original string literals restored
    """
    if not strings:
        return s
    return _PLACEHOLDER_RE.sub(lambda m: strings.get(m.group(0), m.group(0)), s)


def _scan_balanced_block(s: str, start: int) -> str | None:
    """Return the balanced bracket block starting at position `start`.

    Assumes `s[start]` is `{` or `[`. Walks forward, tracking depth and
    string-literal state (so brackets inside strings don't affect depth),
    and returns the substring covering the opening bracket through the
    matching closing bracket. Returns None if the string ends before the
    block is closed.
    """
    opener = s[start]
    closer = "}" if opener == "{" else "]"

    depth = 0
    in_str = False
    esc = False
    quote_char = ""

    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote_char:
                in_str = False
            continue
        if c in ('"', "'"):
            in_str = True
            quote_char = c
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _try_parse_block(block: str) -> tuple[bool, object]:
    """Try to parse `block` as JSON, with a repair fallback.

    Returns (True, parsed_value) on success or (False, None) on failure.
    Catches broad `Exception` because deeply nested pathological input can
    raise `RecursionError` from `json.loads` — we want graceful degradation
    to None, not a propagated crash.
    """
    try:
        return True, json.loads(block)
    except Exception:
        try:
            return True, json.loads(_repair_json(block))
        except Exception:
            return False, None


def _extract_first_json_block(s: str) -> str | None:
    """Extract the first non-empty parseable JSON object or array from a string.

    Iterates over every `{` and `[` position in the string, extracts the
    balanced block starting there via ``_scan_balanced_block``, and returns
    the FIRST block that both (a) parses as JSON (or parses after repair)
    AND (b) is not an empty container. If only empty blocks parse, returns
    the first empty `{}` / `[]` block instead. Returns None only if nothing
    parses at all.

    Why "first non-empty" (vs "first parseable" or "longest parseable"):
    - "First parseable" is too greedy: prose tokens like ``messages[].content``
      form accidental balanced empty containers that parse as valid empty
      JSON, masking real payloads later in the string (round-5 original bug).
    - "Longest parseable" (round-5 fix) broke the envelope-then-example case:
      if an LLM returns a status envelope first and includes a longer JSON
      example later in its prose, "longest" would pick the example and lose
      the envelope (round-5 re-review regression, flagged by codex-cli).
    - "First non-empty" resolves both cases: skips accidental empty
      containers AND picks the envelope over any later example.
    """
    first_empty: str | None = None

    for start in range(len(s)):
        if s[start] not in "{[":
            continue
        block = _scan_balanced_block(s, start)
        if block is None:
            continue
        parsed_ok, parsed = _try_parse_block(block)
        if not parsed_ok:
            continue
        if parsed == {} or parsed == []:
            if first_empty is None:
                first_empty = block
        else:
            return block

    return first_empty


def _repair_json(s: str) -> str:
    """Repair common JSON formatting issues.

    Handles:
    - Comments
    - Smart quotes
    - Single-quoted strings
    - Python/JavaScript literals
    - Unquoted keys
    - Trailing commas
    - Invalid escape sequences

    Uses string masking to protect string literal content from corruption
    during regex-based repairs.
    """
    # Basic cleanup
    s = s.strip()
    s = s.lstrip("\ufeff")

    # Normalize smart quotes before masking (safe to run on everything)
    s = _normalize_quotes(s)

    # Convert single-quoted strings to double-quoted strings. The scanner is
    # state-aware — it tracks whether we're inside a double-quoted string and
    # copies apostrophes verbatim there, so `"it's fine"` is preserved
    # correctly, while genuine single-quoted literals like 'msg' still get
    # converted.
    s = _convert_single_to_double_quotes(s)

    # Mask string literals to protect their content during repairs
    masked, string_map = _mask_strings(s)

    # Apply repairs on masked content (won't corrupt string internals)
    masked = _strip_comments(masked)

    # Convert Python/JS literals to JSON (only outside strings)
    masked = re.sub(r"\bNone\b", "null", masked)
    masked = re.sub(r"\bTrue\b", "true", masked)
    masked = re.sub(r"\bFalse\b", "false", masked)
    masked = re.sub(r"\bundefined\b", "null", masked)
    masked = re.sub(r"\bNaN\b", "null", masked)
    # `\bInfinity\b` matches both `Infinity` and `-Infinity` because the `-`
    # is treated as a non-word boundary; the previous attempt at a separate
    # `\b-Infinity\b` substitution was dead code (the `\b` between `[`/`,`
    # and `-` doesn't fire). The single replacement turns `-Infinity` into
    # `-1e9999`, which json.loads parses as -inf. Good enough.
    masked = re.sub(r"\bInfinity\b", "1e9999", masked)

    # Fix unquoted keys and trailing commas (structural repairs)
    masked = re.sub(_UNQUOTED_KEY_RE, r'\g<prefix>"\g<key>"\g<suffix>', masked)
    masked = re.sub(_TRAILING_COMMA_RE, r"\1", masked)

    # Restore original string literals
    s = _unmask_strings(masked, string_map)

    # Fix invalid escape sequences (including inside strings).
    # JSON only allows: \" \\ \/ \b \f \n \r \t \uXXXX
    # We walk escapes PAIR-BY-PAIR (each `\x` consumes both chars) instead of
    # a single char-class regex, because the char-class approach had a subtle
    # data corruption bug: for input `\\W` (literal backslash + W — a common
    # case after repairing paths like `C:\\Windows`), the engine would skip
    # position 0 (the `\\` pair with excluded `\`), then match position 1
    # (`\W` where W is not excluded), stripping the second backslash and
    # corrupting the valid `\\` escape into `\W`. Pair-by-pair consumption
    # prevents the scanner from ever "sliding into" the middle of a valid
    # escape.
    s = re.sub(
        r"\\.",
        lambda m: m.group(0) if m.group(0)[1] in '"\\/bfnrtu' else m.group(0)[1],
        s,
    )

    return s


def parse_llm_json(text: str) -> Any | None:
    """Parse JSON from LLM response with robust error handling.

    Tries to extract and parse JSON from text that may contain:
    - Markdown code fences
    - Surrounding text
    - Comments
    - Non-standard JSON formatting

    Args:
        text: Raw text from LLM response

    Returns:
        Parsed JSON object/array, or None if parsing fails
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Fast path: if the input is already a clean JSON document, parse it directly.
    # This avoids a subtle bug where the markdown-fence stripper would otherwise
    # reach INTO a JSON-encoded string field (e.g. CLI wrappers like
    # `{"session_id": "...", "response": "### text\n```json\n[...]\n```"}`)
    # and incorrectly extract the inner code fence instead of returning the outer
    # object. We only fall through to fence-stripping when the raw parse fails.
    # We catch broad Exception here (not just JSONDecodeError) because deeply
    # nested pathological input can raise RecursionError from json.loads, which
    # would otherwise escape and crash the caller. LLM output can be adversarial
    # or accidentally malformed — better to gracefully degrade to the unwrapping
    # pipeline (and ultimately None) than to propagate a crash.
    stripped_input = text.strip()
    if stripped_input.startswith(("{", "[")):
        try:
            return json.loads(stripped_input)
        except Exception:
            pass
        # Try the repair pipeline on the OUTER document before any destructive
        # fence stripping. This rescues "almost-valid" wrapper JSON (e.g. with a
        # trailing comma) that contains nested ```json``` markdown inside one of
        # its string fields — without this step, _strip_code_fences below would
        # reach into the string and pull out the inner block.
        try:
            return json.loads(_repair_json(stripped_input))
        except Exception:
            pass  # fall through to the unwrapping pipeline below

    # First attempt: parse WITHOUT stripping analysis blocks. The analysis
    # stripper is designed to handle LLM outputs that wrap reasoning in
    # `<analysis>...</analysis>` tags preceding the real JSON, but it's a
    # text-level regex — it doesn't know whether `<analysis>` is a wrapper
    # OR legitimate string content inside a valid JSON field (e.g.
    # `{"fix": "<analysis>keep</analysis>"}`). Running it before parsing
    # destroys that legitimate content. So: try the full parse pipeline on
    # the untouched text first, and only fall back to analysis-stripping
    # if nothing else works.
    parsed = _try_parse_candidate(text)
    if parsed is not None:
        return parsed

    # Fallback: the LLM probably wrapped its JSON in `<analysis>...</analysis>`
    # reasoning tags. Strip them and re-run the pipeline. Only reached when
    # the unmodified text genuinely couldn't be parsed — which means the
    # analysis blocks were structural wrappers, not string content.
    analysis_stripped = _strip_analysis_blocks(text)
    if analysis_stripped != text:
        return _try_parse_candidate(analysis_stripped)

    return None


def _try_parse_candidate(text: str) -> Any | None:
    """Run the full fence-strip + direct-parse + repair + block-extract pipeline.

    Shared by ``parse_llm_json`` so the main function can run this pipeline
    twice — once on the raw text (to preserve legitimate `<analysis>` string
    content) and once on analysis-stripped text (to rescue outputs where
    `<analysis>` was actually a wrapper).
    """
    raw = _strip_code_fences(text)
    candidate = raw.strip()

    # Try direct parse on the unwrapped content first.
    try:
        return json.loads(candidate)
    except Exception:
        pass

    # Try repair pipeline on the unwrapped content.
    repaired = _repair_json(candidate)
    try:
        return json.loads(repaired)
    except Exception:
        pass

    # Last resort: extract a JSON block from anywhere in the (repaired) text.
    # This handles cases like `Here is the result: {"ok": true} (timestamp: ...)`
    # where the JSON is embedded in surrounding prose, OR cases where the
    # input has a leading non-JSON bracket fragment like `[tag] {real json}`.
    # We run this even if the candidate starts with `{`/`[` because the direct
    # parse may have failed due to trailing garbage.
    #
    # We pass `repaired` (not `candidate`) because the repair pipeline already
    # normalized obvious issues like comments, single quotes and trailing
    # commas. Running the block extractor on the raw candidate would hand the
    # naive brace counter a more hostile input (e.g. unmatched braces hidden
    # inside `// comment` lines).
    block = _extract_first_json_block(repaired)
    if block is not None:
        try:
            return json.loads(block)
        except Exception:
            try:
                return json.loads(_repair_json(block))
            except Exception:
                pass

    return None
