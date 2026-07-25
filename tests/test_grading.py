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
