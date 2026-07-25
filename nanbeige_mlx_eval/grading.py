"""Grade generated model outputs against suite expectations.

Grading is deliberately format-tolerant: agentic models emit tool calls in many
shapes (raw JSON, ``<tool_call>`` blocks, ``name``/``arguments`` or
``tool``/``parameters`` keys). The graders normalize all of these before
comparison so a suite is not brittle to surface formatting.
"""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_tool_call_tags(text: str) -> str:
    """Remove common tool-call wrapper tags, leaving the inner content."""
    text = re.sub(r"</?tool_call>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?function_call>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?tool>", "", text, flags=re.IGNORECASE)
    return text


def _extract_first_json_object(text: str) -> Any | None:
    """Return the first balanced ``{...}`` object parsed as JSON, or ``None``."""
    text = _strip_tool_call_tags(text)
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    try:
                        return json.loads(blob)
                    except json.JSONDecodeError:
                        break  # try the next "{"
        start = text.find("{", start + 1)
    return None


def _normalize_tool_call(obj: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Pull a (tool_name, args) pair out of a parsed object across key conventions."""
    name: str | None = None
    for key in ("name", "tool", "tool_name"):
        if obj.get(key):
            name = obj[key]
            break
    if name is None and isinstance(obj.get("function"), dict):
        name = obj["function"].get("name")

    args: dict[str, Any] = {}
    for key in ("arguments", "args", "parameters"):
        cand = obj.get(key)
        if cand is None and isinstance(obj.get("function"), dict):
            cand = obj["function"].get(key)
        if isinstance(cand, dict):
            args = cand
            break
        if isinstance(cand, str):
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    args = parsed
                    break
            except json.JSONDecodeError:
                continue
    return name, args


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def grade(expect: dict[str, Any], output: str) -> dict[str, Any]:
    """Grade ``output`` against an ``expect`` spec.

    Returns ``{"pass": bool, "detail": str}``.
    """
    kind = expect.get("type")
    if kind == "tool_call":
        return _grade_tool_call(expect, output)
    if kind == "exact_match":
        ok = _norm_text(output) == _norm_text(str(expect.get("value", "")))
        return {"pass": ok, "detail": "exact_match" if ok else "mismatch"}
    if kind == "contains":
        ok = _norm_text(str(expect.get("value", ""))) in _norm_text(output)
        return {"pass": ok, "detail": "found" if ok else "missing"}
    if kind == "json_schema":
        return _grade_json_schema(expect, output)
    if kind == "choice":
        opts = [str(o) for o in expect.get("options", [])]
        norm_out = _norm_text(output)
        # accept the option whose normalized form appears in the output
        hit = next((o for o in opts if _norm_text(o) and _norm_text(o) in norm_out), None)
        return {"pass": hit is not None, "detail": hit or "no_option_matched"}
    return {"pass": False, "detail": f"unknown expect type: {kind}"}


_FN_TAG_RE = re.compile(r"<function=([^>\s/]+)>", re.IGNORECASE)
_PARAM_TAG_RE = re.compile(
    r"<parameter=([^>\s/]+)>(.*?)</parameter>", re.IGNORECASE | re.DOTALL
)


def _extract_tag_tool_call(text: str) -> tuple[str | None, dict[str, Any]]:
    """Parse the Nanbeige native tool-call format.

    The model emits calls as XML-ish tags rather than JSON::

        <tool_call>
        <function=get_weather>
        <parameter=location>Tokyo</parameter>
        </function>
        </tool_call>
    """
    m = _FN_TAG_RE.search(text)
    if not m:
        return None, None
    name = m.group(1).strip()
    args: dict[str, Any] = {}
    for pm in _PARAM_TAG_RE.finditer(text):
        args[pm.group(1).strip()] = pm.group(2).strip()
    return name, args


def _grade_tool_call(expect: dict[str, Any], output: str) -> dict[str, Any]:
    obj = _extract_first_json_object(output)
    if isinstance(obj, dict):
        name, args = _normalize_tool_call(obj)
    else:
        # Fall back to the model's native <function=..><parameter=..> format.
        name, args = _extract_tag_tool_call(output)
    if name is None:
        return {"pass": False, "detail": "no_tool_call_found"}
    want_tool = expect.get("tool")
    if name != want_tool:
        return {"pass": False, "detail": f"tool:{name}!={want_tool}"}
    mode = expect.get("args_match", "subset")
    want_args = expect.get("args", {}) or {}
    if mode == "exact":
        ok = _args_equal(args, want_args)
        return {"pass": ok, "detail": "args_exact" if ok else f"args:{args}"}
    # subset: every expected key/value must be present
    missing = [k for k, v in want_args.items() if not _value_loose_eq(args.get(k), v)]
    return {"pass": not missing, "detail": "args_subset" if not missing else f"missing:{missing}"}


def _args_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return _norm_val(a) == _norm_val(b)


def _norm_val(v: Any) -> Any:
    if isinstance(v, str):
        return _norm_text(v)
    if isinstance(v, dict):
        return {k: _norm_val(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    return v


def _value_loose_eq(actual: Any, expected: Any) -> bool:
    return _norm_val(actual) == _norm_val(expected)


def _grade_json_schema(expect: dict[str, Any], output: str) -> dict[str, Any]:
    obj = _extract_first_json_object(output)
    if not isinstance(obj, dict):
        return {"pass": False, "detail": "no_json_object_found"}
    try:
        import jsonschema  # type: ignore

        jsonschema.validate(instance=obj, schema=expect["schema"])
        return {"pass": True, "detail": "schema_valid"}
    except ImportError:
        # without jsonschema, just confirm it parsed as an object
        return {"pass": True, "detail": "json_object (schema unchecked)"}
    except jsonschema.ValidationError as exc:  # type: ignore
        return {"pass": False, "detail": f"schema_fail:{exc.message}"}
