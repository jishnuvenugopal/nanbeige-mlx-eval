import json

from nanbeige_mlx_eval.grading import grade


def test_tool_call_hermes_format():
    out = '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
    expect = {"type": "tool_call", "tool": "get_weather", "args": {"location": "Tokyo"}, "args_match": "subset"}
    assert grade(expect, out)["pass"]


def test_tool_call_after_reasoning():
    # The model reasons before emitting the call; the grader must still find it.
    out = (
        "The user wants the weather in Paris. I should call get_weather with "
        "location Paris.\n<tool_call>\n{\"name\": \"get_weather\", "
        "\"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"
    )
    expect = {"type": "tool_call", "tool": "get_weather", "args": {"location": "Paris"}, "args_match": "subset"}
    assert grade(expect, out)["pass"]


def test_tool_call_wrong_tool_fails():
    out = '<tool_call>\n{"name": "search_web", "arguments": {"query": "x"}}\n</tool_call>'
    expect = {"type": "tool_call", "tool": "get_weather", "args_match": "subset"}
    assert not grade(expect, out)["pass"]


def test_tool_call_native_tag_format():
    # The model's actual format: <function=..><parameter=..>.. tags, not JSON.
    out = (
        "The user wants the weather in Tokyo.\n</think>\n\n"
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nTokyo\n"
        "</parameter>\n</function>\n</tool_call>"
    )
    expect = {"type": "tool_call", "tool": "get_weather", "args": {"location": "Tokyo"}, "args_match": "subset"}
    assert grade(expect, out)["pass"]


def test_tool_call_no_call_found():
    assert not grade({"type": "tool_call", "tool": "x", "args_match": "subset"}, "just plain text")["pass"]


def test_tool_call_function_wrapper():
    out = '{"function": {"name": "send_email", "arguments": {"to": "a@b.com"}}}'
    expect = {"type": "tool_call", "tool": "send_email", "args": {"to": "a@b.com"}, "args_match": "subset"}
    assert grade(expect, out)["pass"]


def test_exact_match_normalized():
    expect = {"type": "exact_match", "value": "Hello World"}
    assert grade(expect, "  hello world\n")["pass"]


def test_contains():
    assert grade({"type": "contains", "value": "banana"}, "I love a BANANA split.")["pass"]


def test_choice():
    assert grade({"type": "choice", "options": ["Paris", "London"]}, "The answer is Paris.")["pass"]
    assert not grade({"type": "choice", "options": ["Paris"]}, "The answer is Berlin.")["pass"]


def test_json_schema_ok():
    out = '{"name": "Ada", "age": 36}'
    schema = {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    assert grade({"type": "json_schema", "schema": schema}, out)["pass"]


def test_json_schema_missing_field_fails():
    out = '{"name": "Ada"}'
    schema = {"type": "object", "required": ["name", "age"], "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    assert not grade({"type": "json_schema", "schema": schema}, out)["pass"]


# --- P1.1: grade only the final answer (reasoning isolation) ----------------

def test_grade_rejects_unclosed_think():
    # Valid JSON sitting inside an <think> block that never closed must FAIL —
    # the model was still reasoning, it never emitted an answer. This is the
    # exact 8-bit en-json-profile false-positive the review caught.
    out = (
        "<think>\nLet me build the JSON. A profile needs name and age.\n"
        'Here is a draft: {"name": "Alex", "age": 30}\n'
        "Let me write it"
    )
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    r = grade({"type": "json_schema", "schema": schema}, out)
    assert r["pass"] is False
    assert r["detail"] == "truncated_no_answer"
    assert r["reasoning_chars"] > 0


def test_grade_rejects_unclosed_think_tool_call():
    # A well-formed tool call inside an unclosed <think> must also fail.
    out = (
        "<think>\nThe user wants weather. I'll call get_weather.\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Tokyo"}}\n</tool_call>'
    )
    expect = {"type": "tool_call", "tool": "get_weather",
              "args": {"location": "Tokyo"}, "args_match": "subset"}
    r = grade(expect, out)
    assert r["pass"] is False
    assert r["detail"] == "truncated_no_answer"


def test_grade_ignores_reasoning_block_correct_after_close():
    # Wrong call in the reasoning, correct call after </think> -> PASS.
    out = (
        "<think>\nThe user might want search_web, no actually weather.\n"
        '<tool_call>\n{"name": "search_web", "arguments": {"query": "x"}}\n</tool_call>\n'
        "</think>\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>'
    )
    expect = {"type": "tool_call", "tool": "get_weather",
              "args": {"location": "Paris"}, "args_match": "subset"}
    assert grade(expect, out)["pass"]


def test_grade_ignores_reasoning_block_wrong_after_close():
    # Correct call in the reasoning, wrong call after </think> -> FAIL.
    out = (
        "<think>\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {"location": "Paris"}}\n</tool_call>\n'
        "</think>\n"
        '<tool_call>\n{"name": "search_web", "arguments": {"query": "x"}}\n</tool_call>'
    )
    expect = {"type": "tool_call", "tool": "get_weather",
              "args": {"location": "Paris"}, "args_match": "subset"}
    assert not grade(expect, out)["pass"]


def test_grade_require_answer_false_opt_out():
    # A non-thinking suite (require_answer=False) grades the whole stream.
    out = '<think>\n{"name": "Ada", "age": 36}\n'  # unclosed, but opted out
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    assert grade({"type": "json_schema", "schema": schema}, out, require_answer=False)["pass"]


def test_8bit_en_json_profile_regression():
    # The actual committed 8-bit EN output that was falsely graded schema_valid.
    # It is reasoning that cuts off mid-sentence having never closed <think> nor
    # emitted an answer; the draft JSON it wrote while reasoning must NOT pass.
    out = (
        "The user wants a profile JSON. A profile has name and age. "
        'Typically we output it cleanly.\n\n Let me write it'
    )
    schema = {"type": "object", "required": ["name", "age"],
              "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
    # No <think> tag at all here either, but also no answer — the grader treats
    # the whole stream as the answer and finds no JSON object, so it fails.
    r = grade({"type": "json_schema", "schema": schema}, out)
    assert r["pass"] is False


def test_json_schema_missing_jsonschema_raises():
    # B2: if jsonschema is absent, the grader must raise, not silently pass.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "jsonschema":
            raise ImportError("simulated absence")
        return real_import(name, *a, **k)

    builtins.__import__ = fake_import
    # Also drop any cached module so the import path re-runs.
    import sys
    cached = sys.modules.pop("jsonschema", None)
    try:
        import pytest
        out = '{"name": "Ada", "age": 36}'
        schema = {"type": "object", "required": ["name", "age"],
                  "properties": {"name": {"type": "string"}, "age": {"type": "number"}}}
        with pytest.raises(RuntimeError):
            grade({"type": "json_schema", "schema": schema}, out)
    finally:
        builtins.__import__ = real_import
        if cached is not None:
            sys.modules["jsonschema"] = cached
