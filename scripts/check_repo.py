#!/usr/bin/env python3
"""Pre-publish checks for this repository.

Fails (exit 1) if the working tree contains:
  * files larger than MAX_MB;
  * data, model, or slide files that must not be redistributed;
  * course licensing notices copied in by mistake;
  * strings that look like credentials;
  * RAG result tables with columns that carry the manual's text (raw judge output, passages);
  * notebooks that do not validate against the nbformat schema.

Usage:  python scripts/check_repo.py [ROOT]
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

MAX_MB = 25
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", "node_modules"}
BLOCKED_EXT = {".csv", ".tsv", ".xlsx", ".npy", ".npz", ".zip", ".keras", ".h5", ".joblib",
               ".pkl", ".pickle", ".pt", ".pth", ".pptx", ".pdf", ".parquet",
               ".sqlite3", ".sqlite", ".db", ".gguf", ".bin", ".safetensors"}
TEXT_EXT = {".py", ".md", ".ipynb", ".txt", ".yml", ".yaml", ".toml", ".cfg", ".cff", ".json", ".sh"}
# The only binary files allowed: the original presentation PDFs, one folder per project.
# They are published deliberately and as submitted (see README "Provenance and scope").
ALLOWED_BINARY = re.compile(r"^\d{2}-[^/]+/presentation/[^/]+\.pdf$")
MAX_PRESENTATION_MB = 10
# Small generated result tables are allowed in exactly one place: the RAG run's outputs
# (model answers and judge scores, no source text). Everything else that is .csv stays blocked.
ALLOWED_RESULTS = re.compile(r"^05-medical-rag-llm/rerun_2026/rag_outputs/[^/]+\.csv$")
MAX_RESULTS_MB = 2
# Columns that carry source text and so must never be published: the raw judge output
# quotes the copyrighted manual verbatim, and retrieved passages are the manual itself.
BLOCKED_RESULT_COLUMNS = {"judge_raw", "context", "contexts", "passage", "passages", "chunk", "chunk_text"}
NOTICE_PATTERNS = [
    re.compile(r"proprietary content", re.I),
    re.compile(r"meant for personal use", re.I),
    re.compile(r"unauthorized use or distribution", re.I),
]
SECRET_PATTERNS = [
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    # OpenAI-style keys; the lookbehind and alnum-only body skip sklearn's HTML CSS classes (#sk-container-id-1)
    re.compile(r"(?<![\w.#-])sk-(?:proj-)?[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{36}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]
SELF_EXEMPT = {"check_repo.py", "test_check_repo.py"}  # these files contain the patterns on purpose


def _git_publishable(root: Path) -> list[Path] | None:
    """Files git would publish (tracked + untracked-not-ignored), or None outside a git repo."""
    if not (root / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
                              "--exclude-standard"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return [root / f for f in out.decode("utf-8").split("\0") if f and (root / f).is_file()]


def iter_files(root: Path):
    """In a git repo, check exactly what git would publish (so gitignored local data is fine);
    otherwise walk the tree."""
    files = _git_publishable(root)
    if files is not None:
        yield from files
        return
    for p in root.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIRS for part in p.relative_to(root).parts):
            yield p


def check(root: Path) -> list[str]:
    problems: list[str] = []
    for p in iter_files(root):
        rel = p.relative_to(root).as_posix()
        size_mb = p.stat().st_size / 1_048_576
        if size_mb > MAX_MB:
            problems.append(f"too large ({size_mb:.1f} MB > {MAX_MB} MB): {rel}")
        if ALLOWED_BINARY.match(rel):
            if size_mb > MAX_PRESENTATION_MB:
                problems.append(f"presentation too large ({size_mb:.1f} MB > {MAX_PRESENTATION_MB} MB): {rel}")
            continue
        if ALLOWED_RESULTS.match(rel):
            if size_mb > MAX_RESULTS_MB:
                problems.append(f"result table too large ({size_mb:.1f} MB > {MAX_RESULTS_MB} MB): {rel}")
            problems.extend(_check_result_table(p, rel))
            continue
        if p.suffix.lower() in BLOCKED_EXT:
            problems.append(f"blocked file type {p.suffix}: {rel}")
            continue
        if p.suffix.lower() not in TEXT_EXT or p.name in SELF_EXEMPT:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"not UTF-8 text: {rel}")
            continue
        for pat in NOTICE_PATTERNS:
            if pat.search(text):
                problems.append(f"course licensing notice ('{pat.pattern}'): {rel}")
        for pat in SECRET_PATTERNS:
            if pat.search(text):
                problems.append(f"possible secret ({pat.pattern[:12]}…): {rel}")
        if p.suffix == ".ipynb":
            problems.extend(_check_notebook(text, rel))
    return problems


def _check_result_table(path: Path, rel: str) -> list[str]:
    try:
        with path.open(encoding="utf-8", newline="") as fh:
            header = next(csv.reader(fh), [])
    except UnicodeDecodeError:
        return [f"result table is not UTF-8: {rel}"]
    blocked = sorted({c.strip().lower() for c in header} & BLOCKED_RESULT_COLUMNS)
    return [f"result table has source-text column(s) {blocked}: {rel}"] if blocked else []


def _check_notebook(text: str, rel: str) -> list[str]:
    try:
        nb = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f"invalid JSON notebook ({exc.msg}): {rel}"]
    if "widgets" in nb.get("metadata", {}):
        return [f"notebook has widget metadata (GitHub may not render it): {rel}"]
    try:
        import nbformat
    except ImportError:
        return []  # schema check optional when nbformat is absent
    try:
        nbformat.validate(nbformat.from_dict(nb))
    except Exception as exc:  # nbformat.ValidationError
        return [f"notebook fails nbformat validation ({str(exc).splitlines()[0]}): {rel}"]
    return []


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    problems = check(root)
    for msg in problems:
        print(f"FAIL  {msg}")
    print(f"{'FAILED' if problems else 'OK'}: {len(problems)} problem(s) in {root}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
