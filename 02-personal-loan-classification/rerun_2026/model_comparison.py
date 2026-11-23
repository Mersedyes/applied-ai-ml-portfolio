#!/usr/bin/env python3
"""2026 re-run of the personal-loan model comparison, done the way it should have been.

What changed vs. the 2025 notebook
-----------------------------------
* Stratified 70/30 split (the original split was not stratified).
* All tuning and model selection use 5-fold stratified cross-validation on the
  TRAINING set only. The test set is used once, for the selected model and the baseline.
* A majority-class baseline is reported next to every model.
* Selection metric fixed in advance: average precision (PR-AUC). It summarizes the
  precision/recall trade-off across thresholds and is not inflated by the 90% majority class.
* An operating threshold for a recall target is chosen on out-of-fold training
  predictions, never on the test set.
* Test metrics carry bootstrap 95% confidence intervals.
* Every reported number is written by this script (RESULTS.md / results.json), with the
  data hash and library versions, so the write-up cannot drift from the run.

Usage
-----
    python model_comparison.py --data ../../data/Loan_Modelling.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, make_scorer, precision_recall_curve, precision_score,
                             recall_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

log = logging.getLogger("loan_rerun")

TARGET = "Personal_Loan"
REQUIRED = ["ID", "Age", "Experience", "Income", "ZIPCode", "Family", "CCAvg", "Education",
            "Mortgage", TARGET, "Securities_Account", "CD_Account", "Online", "CreditCard"]
NUMERIC = ["Age", "Income", "Family", "CCAvg", "Mortgage"]
CATEGORICAL = ["ZIP2", "Education"]
BINARY = ["Securities_Account", "CD_Account", "Online", "CreditCard"]
TARGET_RECALL = 0.90
SEED = 1


class DataError(ValueError):
    """Input data failed validation."""


# --------------------------------------------------------------------------- data
def load_and_prepare(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the CSV and apply the same preprocessing as the 2025 notebook."""
    if not path.is_file():
        raise DataError(f"data file not found: {path}")
    df = pd.read_csv(path)
    return prepare(df)


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    if df[REQUIRED].isna().any().any():
        raise DataError("unexpected missing values")
    if not set(df[TARGET].unique()) <= {0, 1}:
        raise DataError(f"{TARGET} must be binary 0/1")
    df = df.copy()
    # Same choices as the notebook: negative experience treated as a sign error,
    # Experience then dropped (r ~ 0.99 with Age), ZIP reduced to its first two digits.
    df["Experience"] = df["Experience"].abs()
    df["ZIP2"] = df["ZIPCode"].astype(str).str[:2]
    df["Education"] = df["Education"].astype(str)
    X = df[NUMERIC + CATEGORICAL + BINARY]
    y = df[TARGET].astype(int)
    return X, y


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
        ("bin", "passthrough", BINARY),
    ])


# --------------------------------------------------------------------------- models
def candidates() -> dict[str, tuple[object, dict]]:
    return {
        "Baseline (always 'no')": (DummyClassifier(strategy="most_frequent"), {}),
        "Logistic regression": (LogisticRegression(max_iter=5000),
                                {"clf__C": [0.01, 0.1, 1, 10, 100, 1000], "clf__class_weight": [None, "balanced"]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 7, 11, 15, 21, 31, 45], "clf__weights": ["uniform", "distance"]}),
        "Decision tree": (DecisionTreeClassifier(random_state=SEED),
                          {"clf__max_depth": [3, 4, 5, 6, 8, None], "clf__min_samples_leaf": [1, 5, 10, 25],
                           "clf__class_weight": [None, "balanced"]}),
        "Random forest": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
                          {"clf__max_depth": [None, 8], "clf__min_samples_leaf": [1, 3],
                           "clf__max_features": ["sqrt", 0.5], "clf__class_weight": [None, "balanced_subsample"]}),
    }


# --------------------------------------------------------------------------- metrics
def threshold_for_recall(y_true: np.ndarray, scores: np.ndarray, target: float) -> float:
    """Highest score threshold whose recall is >= target (computed on the data given)."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = recall[:-1] >= target  # recall[i] corresponds to thresholds[i]
    if not ok.any():
        return float(np.min(scores))
    return float(thresholds[ok].max())


def point_metrics(y_true: np.ndarray, pred: np.ndarray, scores: np.ndarray | None) -> dict[str, float]:
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "recall": recall_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if scores is not None:
        out["pr_auc"] = average_precision_score(y_true, scores)
    return {k: float(v) for k, v in out.items()}


def bootstrap_ci(y_true, pred, scores, n: int = 2000, seed: int = SEED) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    scores = None if scores is None else np.asarray(scores)
    draws: dict[str, list[float]] = {}
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:  # resample with no positives: metrics undefined
            continue
        m = point_metrics(y_true[idx], pred[idx], None if scores is None else scores[idx])
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}


@dataclass
class CVResult:
    name: str
    best_params: dict
    cv: dict[str, tuple[float, float]]  # metric -> (mean, sd)
    estimator: Pipeline


# --------------------------------------------------------------------------- run
def run(X: pd.DataFrame, y: pd.Series) -> dict:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"pr_auc": "average_precision",
               "f1": make_scorer(f1_score, zero_division=0),
               "recall": make_scorer(recall_score, zero_division=0),
               "precision": make_scorer(precision_score, zero_division=0),  # baseline predicts no positives
               "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}

    results: list[CVResult] = []
    for name, (clf, grid) in candidates().items():
        pipe = Pipeline([("prep", preprocessor()), ("clf", clf)])
        if grid:
            search = GridSearchCV(pipe, grid, scoring="average_precision", cv=cv, n_jobs=-1)
            search.fit(X_tr, y_tr)
            best, params = search.best_estimator_, search.best_params_
        else:
            best, params = pipe.fit(X_tr, y_tr), {}
        scores = cross_validate(best, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=-1)
        cvm = {m: (float(np.mean(scores[f"test_{m}"])), float(np.std(scores[f"test_{m}"]))) for m in scoring}
        results.append(CVResult(name, {k.replace("clf__", ""): v for k, v in params.items()}, cvm, best))
        log.info("%-24s CV PR-AUC %.3f ± %.3f  %s", name, *cvm["pr_auc"], params)

    models = [r for r in results if not r.name.startswith("Baseline")]
    final = max(models, key=lambda r: r.cv["pr_auc"][0])
    baseline = next(r for r in results if r.name.startswith("Baseline"))

    # Operating threshold for the recall target: chosen on out-of-fold TRAINING predictions.
    oof = cross_val_predict(final.estimator, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    thr = threshold_for_recall(y_tr.to_numpy(), oof, TARGET_RECALL)

    # --- the single use of the test set ---
    final.estimator.fit(X_tr, y_tr)
    te_scores = final.estimator.predict_proba(X_te)[:, 1]
    y_te_np = y_te.to_numpy()
    pred_default = (te_scores >= 0.5).astype(int)
    pred_target = (te_scores >= thr).astype(int)
    base_pred = baseline.estimator.predict(X_te)

    return {
        "n_rows": int(len(X)), "positive_rate": float(y.mean()),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "test_positives": int(y_te.sum()),
        "selection_metric": "average precision (PR-AUC), 5-fold stratified CV on training set",
        "cv": [{"model": r.name, "best_params": r.best_params, **{m: list(v) for m, v in r.cv.items()}} for r in results],
        "final_model": final.name, "final_params": final.best_params,
        "recall_target": TARGET_RECALL, "threshold_for_target": thr,
        "test": {
            "baseline": point_metrics(y_te_np, base_pred, None),
            "final_threshold_0.5": point_metrics(y_te_np, pred_default, te_scores),
            "final_threshold_0.5_ci95": bootstrap_ci(y_te_np, pred_default, te_scores),
            "final_recall_target": point_metrics(y_te_np, pred_target, te_scores),
            "final_recall_target_ci95": bootstrap_ci(y_te_np, pred_target, te_scores),
        },
    }


def render_markdown(r: dict, provenance: dict) -> str:
    f = lambda m, k: f"{m[k][0]:.3f} ± {m[k][1]:.3f}"  # noqa: E731
    lines = [
        "# Personal-loan model comparison: 2026 re-run",
        "",
        "_Generated by `model_comparison.py`; do not edit by hand._",
        "",
        f"Data: {r['n_rows']:,} customers, {r['positive_rate']:.1%} accepted. Stratified split: "
        f"{r['n_train']:,} train / {r['n_test']:,} test ({r['test_positives']} positives in test).",
        f"Selection metric (fixed before running): {r['selection_metric']}.",
        "",
        "## Cross-validation on the training set (mean ± SD over 5 folds, threshold 0.5)",
        "",
        "| Model | PR-AUC | F1 | Recall | Precision | Balanced acc. | Accuracy | Tuned parameters |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in r["cv"]:
        params = ", ".join(f"{k}={v}" for k, v in c["best_params"].items()) or "—"
        lines.append(f"| {c['model']} | {f(c, 'pr_auc')} | {f(c, 'f1')} | {f(c, 'recall')} | {f(c, 'precision')} | "
                     f"{f(c, 'balanced_accuracy')} | {f(c, 'accuracy')} | {params} |")
    t = r["test"]
    ci = lambda key, k: "[{:.3f}, {:.3f}]".format(*t[key][k])  # noqa: E731
    lines += [
        "",
        f"## Held-out test set (used once): **{r['final_model']}**",
        "",
        "| Operating point | Accuracy | Balanced acc. | Recall | Precision | F1 |",
        "|---|---|---|---|---|---|",
        "| Baseline (always 'no') | {accuracy:.3f} | {balanced_accuracy:.3f} | {recall:.3f} | {precision:.3f} | {f1:.3f} |".format(**t["baseline"]),
        "| Final model, threshold 0.5 | {accuracy:.3f} | {balanced_accuracy:.3f} | {recall:.3f} | {precision:.3f} | {f1:.3f} |".format(**t["final_threshold_0.5"]),
        f"| ↳ 95% bootstrap CI | {ci('final_threshold_0.5_ci95','accuracy')} | {ci('final_threshold_0.5_ci95','balanced_accuracy')} | "
        f"{ci('final_threshold_0.5_ci95','recall')} | {ci('final_threshold_0.5_ci95','precision')} | {ci('final_threshold_0.5_ci95','f1')} |",
        f"| Final model, threshold {r['threshold_for_target']:.3f} (recall target {r['recall_target']:.0%}, set on training folds) | "
        "{accuracy:.3f} | {balanced_accuracy:.3f} | {recall:.3f} | {precision:.3f} | {f1:.3f} |".format(**t["final_recall_target"]),
        f"| ↳ 95% bootstrap CI | {ci('final_recall_target_ci95','accuracy')} | {ci('final_recall_target_ci95','balanced_accuracy')} | "
        f"{ci('final_recall_target_ci95','recall')} | {ci('final_recall_target_ci95','precision')} | {ci('final_recall_target_ci95','f1')} |",
        "",
        f"Test PR-AUC (final model): {t['final_threshold_0.5']['pr_auc']:.3f} "
        f"(95% CI [{t['final_threshold_0.5_ci95']['pr_auc'][0]:.3f}, {t['final_threshold_0.5_ci95']['pr_auc'][1]:.3f}]).",
        "",
        "## Provenance",
        "",
        f"- Data SHA-256: `{provenance['data_sha256']}`",
        f"- Python {provenance['python']}, scikit-learn {provenance['sklearn']}, pandas {provenance['pandas']}, numpy {provenance['numpy']}",
        f"- Random seed: {SEED}",
        "",
        "Limitation: the CV figures for each model come from the same folds used to tune it, so they are "
        "slightly optimistic. Nested CV would remove that bias; the single test-set evaluation is unaffected.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=here.parents[1] / "data" / "Loan_Modelling.csv")
    ap.add_argument("--out", type=Path, default=here)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        X, y = load_and_prepare(args.data)
    except DataError as exc:
        log.error("%s", exc)
        return 2
    results = run(X, y)
    provenance = {
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps({**results, "provenance": provenance}, indent=2) + "\n")
    (args.out / "RESULTS.md").write_text(render_markdown(results, provenance))
    log.info("final model: %s -> %s", results["final_model"], args.out / "RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
