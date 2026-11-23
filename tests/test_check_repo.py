"""Tests for scripts/check_repo.py.  Run: python -m pytest -q"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import check_repo as cr  # noqa: E402

GOOD_NB = {"cells": [], "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
           "nbformat": 4, "nbformat_minor": 5}


def test_clean_tree_passes(tmp_path):
    (tmp_path / "README.md").write_text("# hello\n")
    (tmp_path / "a.ipynb").write_text(json.dumps(GOOD_NB))
    assert cr.check(tmp_path) == []


def test_blocks_data_files_and_notices(tmp_path):
    (tmp_path / "data.csv").write_text("a,b\n")
    (tmp_path / "notes.md").write_text("Proprietary content. All rights reserved.")
    problems = cr.check(tmp_path)
    assert any("blocked file type .csv" in p for p in problems)
    assert any("licensing notice" in p for p in problems)


def test_detects_secret_like_strings(tmp_path):
    (tmp_path / "x.py").write_text('TOKEN = "hf_' + "a" * 34 + '"\n')
    (tmp_path / "y.py").write_text('KEY = "sk-' + "B" * 40 + '"\n')
    found = [p for p in cr.check(tmp_path) if "possible secret" in p]
    assert len(found) == 2


def test_ignores_sklearn_html_css(tmp_path):
    css = "#sk-container-id-6 label.sk-toggleable__label-arrow:before {} div.sk-estimator:hover {}"
    (tmp_path / "nb.md").write_text(css)
    assert cr.check(tmp_path) == []


def test_flags_widget_metadata_and_bad_json(tmp_path):
    bad = dict(GOOD_NB, metadata={"widgets": {}})
    (tmp_path / "w.ipynb").write_text(json.dumps(bad))
    (tmp_path / "broken.ipynb").write_text("{not json")
    problems = cr.check(tmp_path)
    assert any("widget metadata" in p for p in problems)
    assert any("invalid JSON" in p for p in problems)


def test_skips_git_dir(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "big.pdf").write_text("x")
    assert cr.check(tmp_path) == []
