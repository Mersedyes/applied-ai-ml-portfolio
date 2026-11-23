#!/usr/bin/env python3
"""Re-score the LLM judge's raw outputs from the 2026 Colab run.

Why this exists
---------------
The judge was asked for ONE JSON object ({"score": 0-3, "reason": "..."}) per
answer. In the 23 Nov 2026 run it usually returned several objects, one per
claim it checked, and was often cut off by the 200-token limit. The notebook's
parser kept only the FIRST "score" it found, so the headline judge means depend
on whichever claim the judge happened to check first.

This script parses every score in each raw judge output and reports how the
headline result changes under three aggregations:

* first  -- what the notebook reported (first score in the output)
* mean   -- mean of all per-claim scores in the output
* min    -- the lowest per-claim score ("every claim supported" reading of the rubric)

It writes a table WITHOUT the judge's free text, because that text quotes the
copyrighted manual verbatim.

Usage:
    python rescore_judge.py RAW_judge_scores.csv OUT_DIR
Outputs:
    OUT_DIR/judge_scores_parsed.csv
    OUT_DIR/judge_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCORE = re.compile(r'"score"\s*:\s*([0-3])')
OBJECT_START = re.compile(r'\{\s*"score"')
SYSTEMS = ["base", "prompted", "rag"]
REQUIRED = {"id", "system", "metric", "score", "judge_raw"}
SEED, N_BOOT = 1, 5000  # same as the notebook's paired() function


def parse_raw(raw: str) -> dict:
    """Return every per-claim score in one raw judge output, plus format flags."""
    raw = raw if isinstance(raw, str) else ""
    scores = [int(s) for s in SCORE.findall(raw)]
    return {
        "n_json_objects": len(OBJECT_START.findall(raw)),
        "scores_all": ";".join(map(str, scores)),
        "score_first": scores[0] if scores else np.nan,
        "score_mean": float(np.mean(scores)) if scores else np.nan,
        "score_min": min(scores) if scores else np.nan,
        "single_object": len(OBJECT_START.findall(raw)) == 1,
        "truncated": not raw.rstrip().endswith("}"),
    }


def paired(df: pd.DataFrame, metric: str, col: str, a: str = "rag", b: str = "base") -> dict:
    """Paired per-question difference a - b with a bootstrap 95% interval (same method as the notebook)."""
    w = df[df.metric == metric].pivot(index="id", columns="system", values=col).dropna()
    d = (w[a] - w[b]).to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    boots = [rng.choice(d, len(d)).mean() for _ in range(N_BOOT)]
    return {"metric": metric, "aggregation": col.removeprefix("score_"), "n_questions": int(len(d)),
            "mean_diff": round(float(d.mean()), 3),
            "ci95": np.percentile(boots, [2.5, 97.5]).round(2).tolist(),
            "rag_better": int((d > 0).sum()), "tied": int((d == 0).sum()), "rag_worse": int((d < 0).sum())}


def rescore(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing = REQUIRED - set(raw.columns)
    if missing:
        raise ValueError(f"input is missing columns: {sorted(missing)}")
    parsed = pd.concat([raw[["id", "system", "metric", "score"]].rename(columns={"score": "score_reported"}),
                        pd.DataFrame([parse_raw(r) for r in raw.judge_raw])], axis=1)
    # The notebook's number must be reproducible from the raw text, or the parse is wrong.
    mismatch = parsed[parsed.score_reported.fillna(-1) != parsed.score_first.fillna(-1)]
    if len(mismatch):
        raise ValueError(f"first parsed score differs from the reported score in {len(mismatch)} rows")

    aggs = ["score_first", "score_mean", "score_min"]
    means = {col.removeprefix("score_"): parsed.pivot_table(index="system", columns="metric", values=col, aggfunc="mean")
             .reindex(SYSTEMS).round(2).to_dict() for col in aggs}
    summary = {
        "n_judgements": int(len(parsed)),
        "format_compliance": {
            "single_json_object": int(parsed.single_object.sum()),
            "multiple_json_objects": int((parsed.n_json_objects > 1).sum()),
            "truncated_at_token_limit": int(parsed.truncated.sum()),
            "unparseable": int(parsed.score_first.isna().sum()),
        },
        "judge_means_by_aggregation": means,
        "paired_rag_minus_base": [paired(parsed, m, col) for col in aggs for m in sorted(parsed.metric.unique())],
    }
    return parsed, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw_csv", type=Path, help="judge_scores.csv written by the notebook (with judge_raw)")
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args(argv)
    if not args.raw_csv.is_file():
        print(f"error: {args.raw_csv} not found", file=sys.stderr)
        return 2
    parsed, summary = rescore(pd.read_csv(args.raw_csv))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parsed.to_csv(args.out_dir / "judge_scores_parsed.csv", index=False)
    (args.out_dir / "judge_sensitivity.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["format_compliance"]))
    for row in summary["paired_rag_minus_base"]:
        print(f"{row['metric']:<13} {row['aggregation']:<6} RAG-base {row['mean_diff']:+.2f} "
              f"95% CI {row['ci95']}  better/tied/worse {row['rag_better']}/{row['tied']}/{row['rag_worse']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
