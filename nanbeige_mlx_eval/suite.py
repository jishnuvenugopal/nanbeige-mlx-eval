"""Eval-suite schema, loading and validation.

A suite is a JSON file describing a set of cases and how each is graded. The
schema is enforced with ``jsonschema`` so that ``validate-suite`` gives a hard
guarantee a suite is well-formed before any model runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover - jsonschema is a dev dependency
    jsonschema = None  # type: ignore


# The set of grading kinds the harness understands. Adding a kind requires a
# matching branch in ``grading.grade``.
EXPECT_TYPES = ("tool_call", "exact_match", "contains", "json_schema", "choice")

SUITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["suite", "cases"],
    "additionalProperties": True,
    "properties": {
        "suite": {"type": "string", "minLength": 1},
        "description": {"type": "string"},
        "language": {"type": "string"},
        "category": {"type": "string"},
        "system": {"type": "string"},
        "cases": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["id", "prompt", "expect"],
                "additionalProperties": True,
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "prompt": {"type": "string", "minLength": 1},
                    "tools": {"type": "array"},
                    "context": {"type": "string"},
                    "max_tokens": {"type": "integer", "minimum": 1},
                    "note": {"type": "string"},
                    "expect": {
                        "type": "object",
                        "required": ["type"],
                        "additionalProperties": True,
                        "properties": {
                            "type": {"enum": list(EXPECT_TYPES)},
                            # tool_call
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                            "args_match": {"enum": ["exact", "subset"]},
                            # exact_match / contains / choice
                            "value": {},
                            "options": {"type": "array"},
                            # json_schema
                            "schema": {"type": "object"},
                        },
                    },
                },
            },
        },
    },
}


class SuiteError(ValueError):
    """Raised when a suite file is missing, unreadable or fails validation."""


def load_suite(path: str | Path) -> dict[str, Any]:
    """Load and return a suite dict from ``path``."""
    p = Path(path)
    if not p.exists():
        raise SuiteError(f"suite not found: {p}")
    try:
        suite = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SuiteError(f"invalid JSON in {p}: {exc}") from exc
    validate_suite(suite)
    return suite


def validate_suite(suite: dict[str, Any]) -> None:
    """Validate a suite dict against :data:`SUITE_SCHEMA` and extra invariants."""
    if not isinstance(suite, dict):
        raise SuiteError("suite must be a JSON object")
    if jsonschema is not None:
        try:
            jsonschema.validate(instance=suite, schema=SUITE_SCHEMA)
        except jsonschema.ValidationError as exc:  # type: ignore
            raise SuiteError(f"schema validation failed: {exc.message}") from exc
    else:  # minimal fallback if jsonschema is unavailable
        for key in ("suite", "cases"):
            if key not in suite:
                raise SuiteError(f"missing required key: {key}")
        for case in suite["cases"]:
            for key in ("id", "prompt", "expect"):
                if key not in case:
                    raise SuiteError(f"case missing required key: {key}")
            if case["expect"].get("type") not in EXPECT_TYPES:
                raise SuiteError(
                    f"case {case.get('id')} has invalid expect.type"
                )

    # extra invariants jsonschema cannot express cheaply
    ids = [c.get("id") for c in suite["cases"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SuiteError(f"duplicate case ids: {sorted(dupes)}")
    for case in suite["cases"]:
        exp = case["expect"]
        if exp["type"] == "tool_call" and "tool" not in exp:
            raise SuiteError(f"case {case['id']}: tool_call requires 'tool'")
        if exp["type"] == "json_schema" and "schema" not in exp:
            raise SuiteError(f"case {case['id']}: json_schema requires 'schema'")
        if exp["type"] == "choice" and not exp.get("options"):
            raise SuiteError(f"case {case['id']}: choice requires 'options'")


def builtin_suites_dir() -> Path:
    """Return the path to the packaged ``suites/`` directory."""
    return Path(__file__).resolve().parent.parent / "suites"


def list_builtin_suites() -> list[Path]:
    """Return sorted paths to all packaged suite files."""
    d = builtin_suites_dir()
    return sorted(d.glob("*.json")) if d.exists() else []
