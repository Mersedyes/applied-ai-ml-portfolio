"""Tests for 05-medical-rag-llm/rerun_2026/rescore_judge.py.  Run: python -m pytest -q"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "rescore_judge", Path(__file__).resolve().parents[1] / "05-medical-rag-llm" / "rerun_2026" / "rescore_judge.py")
rj = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rj)

ONE = '{"score": 3, "reason": "fine"}'
MANY_TRUNCATED = '{"score": 3, "reason": "a"}\n\n{"score": 0, "reason": "b"}\n\n{"score": 2, "reas'


def test_parse_single_object():
    p = rj.parse_raw(ONE)
    assert p["n_json_objects"] == 1 and p["single_object"] and not p["truncated"]
    assert (p["score_first"], p["score_mean"], p["score_min"]) == (3, 3.0, 3)


def test_parse_multiple_objects_and_truncation():
    p = rj.parse_raw(MANY_TRUNCATED)
    assert p["n_json_objects"] == 3 and not p["single_object"] and p["truncated"]
    assert p["scores_all"] == "3;0;2"
    assert (p["score_first"], p["score_min"]) == (3, 0)
    assert p["score_mean"] == pytest.approx(5 / 3)


def test_parse_unparseable():
    p = rj.parse_raw("no json here")
    assert p["n_json_objects"] == 0 and pd.isna(p["score_first"])


def _raw_table():
    rows = []
    for q in ["Q1", "Q2"]:
        for system, raw in [("base", MANY_TRUNCATED), ("prompted", ONE), ("rag", ONE)]:
            for metric in ["groundedness", "relevance"]:
                rows.append({"id": q, "system": system, "metric": metric,
                             "score": rj.parse_raw(raw)["score_first"], "judge_raw": raw})
    return pd.DataFrame(rows)


def test_rescore_aggregations_and_drops_raw_text():
    parsed, summary = rj.rescore(_raw_table())
    assert "judge_raw" not in parsed.columns and "reason" not in "".join(parsed.columns)
    assert summary["format_compliance"]["multiple_json_objects"] == 4
    by = {(r["metric"], r["aggregation"]): r for r in summary["paired_rag_minus_base"]}
    assert by[("groundedness", "first")]["mean_diff"] == 0.0   # first score: 3 vs 3
    assert by[("groundedness", "min")]["mean_diff"] == 3.0     # min score: 3 vs 0
    assert by[("groundedness", "min")]["rag_better"] == 2


def test_rescore_refuses_when_reported_score_cannot_be_reproduced():
    raw = _raw_table()
    raw.loc[0, "score"] = 1  # not the first score in its raw text
    with pytest.raises(ValueError, match="differs from the reported score"):
        rj.rescore(raw)


def test_rescore_requires_raw_column():
    with pytest.raises(ValueError, match="missing columns"):
        rj.rescore(_raw_table().drop(columns="judge_raw"))
