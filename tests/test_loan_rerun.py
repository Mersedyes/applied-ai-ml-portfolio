"""Tests for 02-personal-loan-classification/rerun_2026/model_comparison.py (no dataset needed)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "02-personal-loan-classification" / "rerun_2026"))
import model_comparison as mc  # noqa: E402


def frame(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ID": range(n), "Age": rng.integers(25, 65, n), "Experience": rng.integers(-3, 40, n),
        "Income": rng.integers(10, 200, n), "ZIPCode": rng.choice([91107, 90089, 94720, 95616], n),
        "Family": rng.integers(1, 5, n), "CCAvg": rng.random(n) * 5, "Education": rng.integers(1, 4, n),
        "Mortgage": rng.integers(0, 300, n), "Personal_Loan": [0, 1] * (n // 2),
        "Securities_Account": rng.integers(0, 2, n), "CD_Account": rng.integers(0, 2, n),
        "Online": rng.integers(0, 2, n), "CreditCard": rng.integers(0, 2, n),
    })


def test_prepare_mirrors_notebook_choices():
    X, y = mc.prepare(frame())
    assert list(X.columns) == mc.NUMERIC + mc.CATEGORICAL + mc.BINARY
    assert "Experience" not in X and "ID" not in X
    assert set(X["ZIP2"]) <= {"91", "90", "94", "95"}
    assert set(y) == {0, 1}


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["Income"]), "missing columns"),
    (lambda d: d.assign(Income=np.nan), "missing values"),
    (lambda d: d.assign(Personal_Loan=2), "binary"),
])
def test_prepare_rejects_bad_input(mutate, msg):
    with pytest.raises(mc.DataError, match=msg):
        mc.prepare(mutate(frame()))


def test_load_reports_missing_file(tmp_path):
    with pytest.raises(mc.DataError, match="not found"):
        mc.load_and_prepare(tmp_path / "nope.csv")


def test_threshold_for_recall_hits_target_on_given_data():
    y = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
    s = np.array([.1, .2, .3, .9, .4, .8, .7, .6, .35, .95])
    thr = mc.threshold_for_recall(y, s, 0.9)
    assert ((s >= thr) & (y == 1)).sum() / y.sum() >= 0.9
    # and it is the highest such threshold: the next score up would miss the target
    higher = s[s > thr]
    if higher.size:
        assert ((s >= higher.min()) & (y == 1)).sum() / y.sum() < 0.9


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    pred = np.where(rng.random(300) < 0.85, y, 1 - y)
    lo, hi = mc.bootstrap_ci(y, pred, None, n=300)["accuracy"]
    point = (pred == y).mean()
    assert lo <= point <= hi
