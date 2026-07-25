from pathlib import Path

import pytest

from nanbeige_mlx_eval.suite import SuiteError, list_builtin_suites, load_suite

SUITES_DIR = Path(__file__).resolve().parent.parent / "suites"


def test_all_builtin_suites_validate():
    suites = list_builtin_suites()
    assert suites, "no packaged suites found"
    for p in suites:
        s = load_suite(p)
        assert s["suite"]
        assert len(s["cases"]) >= 1
        for c in s["cases"]:
            assert "expect" in c and "type" in c["expect"]


def test_duplicate_ids_rejected(tmp_path):
    bad = {
        "suite": "bad",
        "cases": [
            {"id": "x", "prompt": "p", "expect": {"type": "exact_match", "value": "a"}},
            {"id": "x", "prompt": "p", "expect": {"type": "exact_match", "value": "a"}},
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(__import__("json").dumps(bad))
    with pytest.raises(SuiteError):
        load_suite(p)


def test_bad_expect_type_rejected(tmp_path):
    import json

    bad = {"suite": "bad", "cases": [{"id": "x", "prompt": "p", "expect": {"type": "telepathy"}}]}
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SuiteError):
        load_suite(p)
