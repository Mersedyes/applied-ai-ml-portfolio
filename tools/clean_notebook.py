#!/usr/bin/env python3
"""Strip course scaffolding from a completed Jupyter notebook.

The notebooks in this repository were completed on course-provided "low-code"
templates. This tool removes the course-authored material so that what remains
is the analysis itself:

* every markdown cell before the first code cell (problem statement, context,
  instructions), replaced by one short header cell;
* course instruction notes ("Please read the instructions", "restart the
  kernel", "free-tier GPU" notes);
* markdown cells that are only blank-answer placeholders ("___");
* grading annotations such as "[0.5 mark]" or "[2 marks]";
* "Complete the code ..." comments inside code cells;
* Colab/editor metadata (widgets, accelerator, cell ids), which also keeps
  GitHub's notebook renderer from rejecting the file.

Code, outputs, and learner-written observations are left untouched. The tool
refuses to write a result whose code-cell count or output count changed.

Usage:
    python tools/clean_notebook.py SRC.ipynb DST.ipynb --title "Project title"
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("clean_notebook")

INSTRUCTION_PATTERNS = [
    re.compile(r"please read the instructions carefully", re.I),
    re.compile(r"this is a commented (jupyter|python)", re.I),
    re.compile(r"restart the notebook kernel", re.I),
    re.compile(r"run all cells sequentially", re.I),
    re.compile(r"free-tier GPU", re.I),
]
PLACEHOLDER_ONLY = re.compile(r"^[\s_\-*]*$")
MARKS = re.compile(r"\s*\[\s*\d+(\.\d+)?\s*marks?\s*\]", re.I)
COMPLETE_THE_CODE = re.compile(r"\s*#+\s*complete the code[^\n]*", re.I)
KEEP_NB_METADATA = {"kernelspec", "language_info"}


class CleaningError(RuntimeError):
    """Raised when cleaning would change code or outputs."""


def _text(cell: dict[str, Any]) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def _set_text(cell: dict[str, Any], text: str) -> None:
    cell["source"] = text.splitlines(keepends=True)


def _is_instruction(text: str) -> bool:
    return any(p.search(text) for p in INSTRUCTION_PATTERNS)


def _strip_code_comments(text: str) -> str:
    out = []
    for line in text.splitlines(keepends=True):
        cleaned = COMPLETE_THE_CODE.sub("", line.rstrip("\n"))
        if cleaned.strip() == "" and line.strip() != "":
            continue  # the whole line was a "Complete the code" comment
        out.append(cleaned + ("\n" if line.endswith("\n") else ""))
    return "".join(out)


def _count_outputs(cells: list[dict[str, Any]]) -> int:
    return sum(len(c.get("outputs", [])) for c in cells if c.get("cell_type") == "code")


def header_cell(title: str, readme: str = "README.md") -> dict[str, Any]:
    text = (
        f"# {title}\n\n"
        "Author: Mersedyes Livingston\n\n"
        f"Problem summary, data description, results and re-analysis: see [{readme}]({readme}).\n\n"
        "> Built on a course-provided template. Course instructions and problem text were "
        "removed with `tools/clean_notebook.py`; code, outputs and observations are unchanged. "
        "Data is not redistributed, so cells that load it will not run as-is.\n"
    )
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def clean(nb: dict[str, Any], title: str) -> tuple[dict[str, Any], dict[str, int]]:
    if nb.get("nbformat") != 4:
        raise CleaningError(f"expected nbformat 4, got {nb.get('nbformat')!r}")
    cells_in = nb.get("cells", [])
    first_code = next((i for i, c in enumerate(cells_in) if c.get("cell_type") == "code"), None)
    if first_code is None:
        raise CleaningError("notebook has no code cells")

    stats = {"dropped_intro": 0, "dropped_instruction": 0, "dropped_placeholder": 0,
             "marks_removed": 0, "code_comments_removed": 0}
    cells_out: list[dict[str, Any]] = [header_cell(title)]

    for i, cell in enumerate(cells_in):
        cell = copy.deepcopy(cell)
        ctype = cell.get("cell_type")
        cell["metadata"] = {k: v for k, v in cell.get("metadata", {}).items() if k == "tags"}
        text = _text(cell)

        if ctype == "markdown":
            if i < first_code:
                stats["dropped_intro"] += 1
                continue
            if _is_instruction(text):
                stats["dropped_instruction"] += 1
                continue
            if PLACEHOLDER_ONLY.match(text):
                stats["dropped_placeholder"] += 1
                continue
            new_text, n = MARKS.subn("", text)
            stats["marks_removed"] += n
            if n:
                _set_text(cell, new_text)
        elif ctype == "code":
            new_text = _strip_code_comments(text)
            if new_text != text:
                stats["code_comments_removed"] += len(COMPLETE_THE_CODE.findall(text))
                _set_text(cell, new_text)
        cells_out.append(cell)

    out = {
        "cells": cells_out,
        "metadata": {k: v for k, v in nb.get("metadata", {}).items() if k in KEEP_NB_METADATA},
        "nbformat": 4,
        "nbformat_minor": int(nb.get("nbformat_minor", 0)),
    }
    if out["nbformat_minor"] >= 5:  # nbformat >= 4.5 requires an id on every cell
        for n, c in enumerate(cells_out):
            c.setdefault("id", f"cell-{n}")

    # Invariants: code cells and outputs must be preserved exactly in number.
    code_in = sum(1 for c in cells_in if c.get("cell_type") == "code")
    code_out = sum(1 for c in cells_out if c.get("cell_type") == "code")
    if code_in != code_out:
        raise CleaningError(f"code cell count changed: {code_in} -> {code_out}")
    if _count_outputs(cells_in) != _count_outputs(cells_out):
        raise CleaningError("output count changed")
    return out, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.src.is_file():
        log.error("source not found: %s", args.src)
        return 2
    if args.dst.exists() and not args.force:
        log.error("destination exists (use --force): %s", args.dst)
        return 2
    try:
        nb = json.loads(args.src.read_text(encoding="utf-8"))
        out, stats = clean(nb, args.title)
    except (json.JSONDecodeError, CleaningError) as exc:
        log.error("%s: %s", args.src, exc)
        return 1

    try:  # optional schema validation
        import nbformat
        nbformat.validate(nbformat.from_dict(out))
    except ImportError:
        log.warning("nbformat not installed; skipped schema validation")
    except Exception as exc:  # nbformat.ValidationError and friends
        log.error("cleaned notebook failed validation: %s", exc)
        return 1

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    args.dst.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    log.info("%s -> %s %s", args.src.name, args.dst, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
