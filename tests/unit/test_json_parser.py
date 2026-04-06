"""Tests for JSON parser."""

from multi_mcp.utils.json_parser import parse_llm_json


def test_parse_simple_json():
    """Test parsing simple JSON."""
    result = parse_llm_json('{"status": "no_issues_found", "message": "Great"}')
    assert result == {"status": "no_issues_found", "message": "Great"}


def test_parse_outer_json_with_inner_code_fence_in_string():
    """Regression: gemini-cli wraps responses as `{session_id, response, stats}`
    where the `response` field contains markdown text that itself includes a
    ```json ... ``` code block. The previous fence-stripping logic would reach
    INTO that nested string and extract the inner code fence, returning a list
    instead of the outer dict — which then caused codereview.py to fail with
    'Failed to parse LLM response as JSON'. We must return the OUTER object."""
    gemini_wrapper = (
        '{"session_id": "abc-123", '
        '"response": "### Analysis\\n1. SQL Injection.\\n```json\\n[{\\"issue\\": \\"sqli\\"}]\\n```", '
        '"stats": {"tokens": 100}}'
    )
    result = parse_llm_json(gemini_wrapper)
    assert isinstance(result, dict), f"expected outer dict, got {type(result).__name__}"
    assert result.get("session_id") == "abc-123"
    assert "response" in result
    assert "### Analysis" in result["response"]
    assert result.get("stats", {}).get("tokens") == 100


def test_parse_clean_json_array_returns_list():
    """A clean JSON array at top level should round-trip as a Python list."""
    result = parse_llm_json('[{"a": 1}, {"b": 2}]')
    assert result == [{"a": 1}, {"b": 2}]


def test_parse_preserves_apostrophes_inside_double_quoted_strings():
    """Regression: _convert_single_to_double_quotes used to run BEFORE string
    masking, so an apostrophe inside a valid double-quoted string (e.g. 'it\\'s
    fine') would trigger a spurious single-quoted-string scan and mangle the
    content. The fix masks double-quoted strings first, then converts only the
    unmasked single-quoted literals.

    Failure mode: {"message": "it's fine",} enters repair mode because of the
    trailing comma, then the repair used to corrupt the apostrophe."""
    result = parse_llm_json('{"message": "it\'s fine",}')
    assert result == {"message": "it's fine"}


def test_parse_preserves_multiple_apostrophes_in_double_quoted_content():
    """Multiple apostrophes in the same double-quoted string must all survive repair."""
    # The outer dict needs a trailing comma to force the repair pipeline
    result = parse_llm_json('{"msg": "it\'s John\'s car, isn\'t it?",}')
    assert result == {"msg": "it's John's car, isn't it?"}


def test_parse_genuine_single_quoted_json_still_converts():
    """The fix must NOT break conversion of genuinely single-quoted JSON literals."""
    result = parse_llm_json("{'status': 'ok'}")
    assert result == {"status": "ok"}


def test_parse_extracts_json_after_leading_bracket_fragment():
    """Regression: codex round-3 finding. _extract_first_json_block used to pick
    the EARLIEST `[` or `{` and give up if that fragment didn't parse, returning
    None. Now it iterates over every bracket position and returns the first one
    that parses successfully — so `prefix [note] {"real": "json"}` correctly
    extracts the trailing object."""
    result = parse_llm_json('prefix [note] {"real": "json"}')
    assert result == {"real": "json"}


def test_parse_extracts_json_with_trailing_garbage():
    """Regression: gemini round-3 finding. parse_llm_json used to skip block
    extraction entirely when input started with `{` or `[`. So a clean JSON
    object followed by trailing prose would fail to parse. Now block extraction
    runs as a final fallback so trailing garbage is tolerated."""
    result = parse_llm_json('{"ok": true} (timestamp: 12345)')
    assert result == {"ok": True}


def test_parse_handles_trailing_text_after_closing_code_fence():
    """Lock-in: gemini round-3 finding #2 was a false positive — the greedy
    fence pattern requires \\s*$ (closing fence at EOL), but there's a fallback
    pattern that handles trailing text. Verify the fallback works."""
    content = '```json\n{"status": "ok"}\n```\nFollow-up commentary here.'
    result = parse_llm_json(content)
    assert result == {"status": "ok"}


def test_parse_nested_fences_with_trailing_text():
    """Round-4 finding (gemini high): nested markdown fences (e.g. a "fix" field
    containing ```python``` blocks) MUST not be truncated by the fallback regex
    even when the outer LLM output has trailing prose. Previously
    `_CODE_FENCE_RE` was non-greedy and stopped at the first inner closing
    ```` ``` ````, mangling the JSON. Now it's greedy and locks onto the
    OUTERMOST opening + closing pair."""
    content = (
        "```json\n"
        '{"status": "success", "fix": "```python\\nprint(42)\\n```"}\n'
        "```\n"
        "Follow-up commentary here."
    )
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "success"
    assert "print(42)" in result["fix"]
    assert "```python" in result["fix"]  # inner fence preserved verbatim


def test_parse_outer_json_with_trailing_comma_and_nested_fence():
    """Round-4 finding (codex medium): if outer JSON is repairable (e.g. trailing
    comma) AND contains a nested ```json``` block in a string field, the parser
    must repair the outer document BEFORE running destructive fence stripping —
    otherwise it would extract the inner block and lose the outer object."""
    # Trailing comma forces the repair pipeline; nested code fence inside a
    # string would otherwise be misinterpreted by _strip_code_fences.
    content = '{"fix": "see ```json\\n{\\"x\\":1}\\n```", "ok": true,}'
    result = parse_llm_json(content)
    assert result is not None
    assert result["ok"] is True
    assert "```json" in result["fix"]


def test_parse_negative_infinity_via_single_substitution():
    """Round-4 finding (gemini low): the previous `\\b-Infinity\\b` substitution
    was dead code (word boundary doesn't fire between `[`/`,`/space and `-`).
    Removed it because the existing `\\bInfinity\\b` substitution already
    converts `Infinity` to `1e9999`, which means `-Infinity` becomes `-1e9999`,
    which json.loads parses as -inf. Lock that behavior in."""
    import math

    result = parse_llm_json('{"a": Infinity, "b": -Infinity}')
    assert result is not None
    assert math.isinf(result["a"]) and result["a"] > 0
    assert math.isinf(result["b"]) and result["b"] < 0


def test_parse_with_markdown_fence():
    """Test parsing JSON in markdown fence."""
    content = """```json
{"status": "no_issues_found", "message": "Excellent code"}
```"""
    result = parse_llm_json(content)
    assert result["status"] == "no_issues_found"
    assert result["message"] == "Excellent code"


def test_parse_with_markdown_fence_no_language():
    """Test parsing JSON in markdown fence without language tag."""
    content = """```
{"status": "no_issues_found", "message": "Good"}
```"""
    result = parse_llm_json(content)
    assert result["status"] == "no_issues_found"


def test_parse_trailing_comma():
    """Test repairing trailing commas."""
    result = parse_llm_json('{"a": 1, "b": 2,}')
    assert result == {"a": 1, "b": 2}


def test_parse_trailing_comma_in_array():
    """Test repairing trailing commas in arrays."""
    result = parse_llm_json('{"arr": [1, 2, 3,]}')
    assert result == {"arr": [1, 2, 3]}


def test_parse_unquoted_keys():
    """Test repairing unquoted keys."""
    result = parse_llm_json('{foo: "bar", baz: 123}')
    assert result == {"foo": "bar", "baz": 123}


def test_parse_python_true():
    """Test converting Python True literal."""
    result = parse_llm_json('{"a": True}')
    assert result == {"a": True}


def test_parse_python_false():
    """Test converting Python False literal."""
    result = parse_llm_json('{"a": False}')
    assert result == {"a": False}


def test_parse_python_none():
    """Test converting Python None literal."""
    result = parse_llm_json('{"a": None}')
    assert result == {"a": None}


def test_parse_python_literals_combined():
    """Test converting multiple Python literals."""
    result = parse_llm_json('{"a": True, "b": False, "c": None}')
    assert result == {"a": True, "b": False, "c": None}


def test_parse_with_line_comments():
    """Test stripping line comments."""
    result = parse_llm_json("""
    {
        "status": "no_issues_found", // This is a comment
        "message": "Good"
    }
    """)
    assert result["status"] == "no_issues_found"


def test_parse_with_block_comments():
    """Test stripping block comments."""
    result = parse_llm_json("""
    {
        "status": "no_issues_found",
        /* Block comment here */
        "message": "Good"
    }
    """)
    assert result["status"] == "no_issues_found"


def test_parse_json_in_text():
    """Test extracting JSON from surrounding text."""
    result = parse_llm_json('Here is the response: {"status": "ok", "value": 123} and more text')
    assert result == {"status": "ok", "value": 123}


def test_parse_json_with_prefix_text():
    """Test extracting JSON when preceded by text."""
    content = """The analysis is complete.

{
  "status": "no_issues_found",
  "message": "Code is excellent"
}"""
    result = parse_llm_json(content)
    assert result["status"] == "no_issues_found"


def test_parse_files_required_response():
    """Test parsing files_required special case."""
    content = """{
  "status": "files_required_to_continue",
  "message": "Need auth module",
  "files_needed": ["auth.py", "models/"]
}"""
    result = parse_llm_json(content)
    assert result["status"] == "files_required_to_continue"
    assert len(result["files_needed"]) == 2
    assert "auth.py" in result["files_needed"]


def test_parse_non_json():
    """Test that non-JSON returns None."""
    result = parse_llm_json("This is just plain text with no JSON")
    assert result is None


def test_parse_empty():
    """Test that empty string returns None."""
    result = parse_llm_json("")
    assert result is None


def test_parse_whitespace_only():
    """Test that whitespace-only string returns None."""
    result = parse_llm_json("   \n\t  ")
    assert result is None


def test_parse_invalid_json():
    """Test that completely broken JSON returns None."""
    result = parse_llm_json("{this is broken json")
    assert result is None


def test_parse_none_input():
    """Test that None input returns None."""
    result = parse_llm_json(None)  # type: ignore
    assert result is None


def test_parse_smart_quotes():
    """Test handling of smart quotes around keys."""
    # Smart quotes around the key
    result = parse_llm_json('{"message": "Code is excellent"}')
    assert result is not None
    assert "message" in result


def test_parse_array():
    """Test parsing JSON array."""
    result = parse_llm_json("[1, 2, 3]")
    assert result == [1, 2, 3]


def test_parse_nested_objects():
    """Test parsing nested JSON objects."""
    result = parse_llm_json('{"outer": {"inner": "value"}}')
    assert result == {"outer": {"inner": "value"}}


def test_parse_combined_issues():
    """Test handling multiple malformations at once."""
    content = """```json
{
  status: "no_issues_found",  // Comment here
  message: "Great code",
}
```"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "no_issues_found"


def test_parse_with_analysis_block():
    """Test stripping <analysis> block before parsing JSON."""
    content = """<analysis>
1. VERIFICATION:
   - Issue "SQL injection": Confirmed. Fix required.
   - Issue "Missing import": False positive. The code handles this in line 45. Discarding.
2. DISCOVERY:
   - Found potential race condition in file.py:12.
</analysis>

```json
{
  "status": "issues_found",
  "issues_found": [
    {"severity": "high", "location": "db.py:23", "description": "SQL injection"}
  ]
}
```"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "issues_found"
    assert len(result["issues_found"]) == 1
    assert result["issues_found"][0]["severity"] == "high"


def test_parse_with_analysis_block_no_fence():
    """Test stripping <analysis> block when JSON is not in a fence."""
    content = """<analysis>
1. VERIFICATION:
   - Issue "X": Confirmed.
</analysis>

{"status": "no_issues_found", "message": "All clear"}"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "no_issues_found"


def test_parse_with_analysis_block_inline():
    """Test stripping <analysis> block that appears inline with JSON."""
    content = """<analysis>Quick check: all good</analysis>{"status": "ok"}"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "ok"


def test_parse_invalid_escape_sequences():
    """Test repairing invalid escape sequences (common in LLM code samples)."""
    # Test \' (Python-style escaped single quote - invalid in JSON)
    content = r'{"code": "print(\'hello\')", "status": "ok"}'
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "ok"
    assert result["code"] == "print('hello')"

    # Test other invalid escapes
    content2 = r'{"pattern": "\\d+", "escape": "\\x41", "status": "ok"}'
    result2 = parse_llm_json(content2)
    assert result2 is not None
    assert result2["status"] == "ok"


def test_parse_complex_code_with_escapes():
    """Test parsing JSON with code samples that have invalid escapes (like real LLM output)."""
    # Simulate what an LLM might return with Python code containing \'
    content = r'{"fix": "f\'Hello {name}\'", "status": "success"}'
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "success"
    assert "Hello" in result["fix"]

    # Another common case: regex patterns with \d, \w, etc
    content2 = '{"pattern": "Match \\digit with \\d", "status": "ok"}'
    result2 = parse_llm_json(content2)
    assert result2 is not None
    assert result2["status"] == "ok"


def test_parse_protected_strings():
    """Test that string masking protects URLs, literals, and patterns inside strings."""
    # Test URL with // (should not be treated as comment)
    content1 = '{"url": "http://example.com", "status": "ok"}'
    result1 = parse_llm_json(content1)
    assert result1 is not None
    assert result1["url"] == "http://example.com"
    assert result1["status"] == "ok"

    # Test literal "None" inside string (should not be replaced with null)
    content2 = '{"message": "Found None in code", "status": "ok"}'
    result2 = parse_llm_json(content2)
    assert result2 is not None
    assert result2["message"] == "Found None in code"

    # Test UNC path with //
    content3 = '{"path": "//server/share/file.txt", "status": "ok"}'
    result3 = parse_llm_json(content3)
    assert result3 is not None
    assert result3["path"] == "//server/share/file.txt"

    # Test unquoted key pattern inside string (should not add quotes)
    content4 = '{"code": "obj = {key: value}", "status": "ok"}'
    result4 = parse_llm_json(content4)
    assert result4 is not None
    assert result4["code"] == "obj = {key: value}"

    # Test multiple issues with unquoted keys outside strings
    content5 = '{url: "http://example.com", status: "ok"}'  # unquoted keys outside
    result5 = parse_llm_json(content5)
    assert result5 is not None
    assert result5["url"] == "http://example.com"  # URL preserved
    assert result5["status"] == "ok"


def test_parse_combined_with_masking():
    """Test string masking combined with other repairs."""
    # URL + comment + unquoted keys
    content = """{
        url: "http://example.com", // This is a URL
        path: "//server/share",
        message: "Value is None here",
        status: "ok"
    }"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["url"] == "http://example.com"
    assert result["path"] == "//server/share"
    assert result["message"] == "Value is None here"
    assert result["status"] == "ok"


def test_parse_single_quoted_strings():
    """Test converting single-quoted strings to double quotes."""
    content = "{'msg': 'hello', 'count': 10}"
    result = parse_llm_json(content)
    assert result is not None
    assert result["msg"] == "hello"
    assert result["count"] == 10


def test_parse_single_quotes_with_trailing_comma():
    """Test single quotes combined with trailing comma."""
    content = "{'msg': 'hello', 'count': 10,}"
    result = parse_llm_json(content)
    assert result is not None
    assert result["msg"] == "hello"
    assert result["count"] == 10


def test_parse_single_quotes_with_escaped_quotes():
    """Test single-quoted strings containing escaped single quotes."""
    content = r"{'msg': 'it\'s working', 'status': 'ok'}"
    result = parse_llm_json(content)
    assert result is not None
    assert result["msg"] == "it's working"
    assert result["status"] == "ok"


def test_parse_single_quotes_with_double_quotes_inside():
    """Test single-quoted strings containing double quotes."""
    content = """{'msg': 'he said "hello"', 'status': 'ok'}"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["msg"] == 'he said "hello"'
    assert result["status"] == "ok"


def test_parse_unclosed_code_fence():
    """Test handling unclosed code fence (LLM cut off mid-response)."""
    content = """```json
{
  "a": 1,
  "b": 2
}"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["a"] == 1
    assert result["b"] == 2


def test_parse_unclosed_fence_with_label():
    """Test unclosed fence with explicit json label."""
    content = """```json
{"status": "ok", "value": 42}"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "ok"
    assert result["value"] == 42


def test_parse_nested_code_fences():
    """Test parsing JSON with nested code fences in string values.

    This reproduces the bug where LLM responses contained code snippets
    with their own ``` fences inside JSON string fields, causing the parser
    to match the first closing ``` instead of the outermost one.
    """
    content = """```json
{
  "status": "success",
  "issues": [
    {
      "description": "Bug found",
      "fix": "```python\\nprint('hello')\\n```"
    }
  ]
}
```"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "success"
    assert isinstance(result["issues"], list)
    assert len(result["issues"]) == 1
    assert "fix" in result["issues"][0]
    assert "```python" in result["issues"][0]["fix"]


def test_parse_multiple_nested_code_fences():
    """Test parsing JSON with multiple nested code fences.

    This simulates a real code review response with multiple issues,
    each containing code fixes with markdown fences.
    """
    content = """```json
{
  "status": "success",
  "message": "Found 2 issues",
  "issues_found": [
    {
      "severity": "high",
      "location": "file.py:10",
      "description": "Issue 1",
      "fix": "```python\\ndef foo():\\n    pass\\n```"
    },
    {
      "severity": "medium",
      "location": "file.py:20",
      "description": "Issue 2",
      "fix": "```python\\ndef bar():\\n    pass\\n```"
    }
  ]
}
```"""
    result = parse_llm_json(content)
    assert result is not None
    assert result["status"] == "success"
    assert result["message"] == "Found 2 issues"
    assert len(result["issues_found"]) == 2
    assert result["issues_found"][0]["severity"] == "high"
    assert result["issues_found"][1]["severity"] == "medium"
    assert "```python" in result["issues_found"][0]["fix"]
    assert "```python" in result["issues_found"][1]["fix"]
