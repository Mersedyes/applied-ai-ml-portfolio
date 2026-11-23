"""Tests for tools/clean_notebook.py.  Run: python -m pytest -q"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import clean_notebook as cn  # noqa: E402


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text, outputs=None):
    return {"cell_type": "code", "metadata": {"colab": {"x": 1}, "id": "abc"},
            "execution_count": 1, "outputs": outputs or [], "source": text}


def nb(cells, minor=0):
    return {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "colab": {}, "widgets": {}},
            "nbformat": 4, "nbformat_minor": minor}


STREAM = {"output_type": "stream", "name": "stdout", "text": "ok\n"}


def test_drops_intro_and_adds_header():
    out, stats = cn.clean(nb([md("# Problem Statement"), md("Context..."), code("import x")]), "T")
    assert out["cells"][0]["source"][0].startswith("# T")
    assert stats["dropped_intro"] == 2
    assert [c["cell_type"] for c in out["cells"]] == ["markdown", "code"]


def test_drops_instruction_and_placeholder_cells_after_first_code():
    cells = [code("a=1"), md("**Note**: restart the notebook kernel"), md("___"), md("## Observations\nreal text")]
    out, stats = cn.clean(nb(cells), "T")
    texts = ["".join(c["source"]) for c in out["cells"]]
    assert stats["dropped_instruction"] == 1 and stats["dropped_placeholder"] == 1
    assert any("real text" in t for t in texts)


def test_strips_marks_and_complete_the_code_comments():
    cells = [code("x = f()  ## Complete the code to call f\n# Complete the code below\ny = 2\n", [STREAM]),
             md("### Question 1: rows? [0.5 mark]"), md("### Question 2 [2 marks]")]
    out, stats = cn.clean(nb(cells), "T")
    src = "".join(out["cells"][1]["source"])
    assert src == "x = f()\ny = 2\n"
    assert "mark" not in "".join(out["cells"][2]["source"])
    assert stats["marks_removed"] == 2 and stats["code_comments_removed"] == 2


def test_preserves_outputs_and_strips_metadata():
    out, _ = cn.clean(nb([code("a", [STREAM, STREAM])]), "T")
    assert out["cells"][1]["outputs"] == [STREAM, STREAM]
    assert out["cells"][1]["metadata"] == {}
    assert set(out["metadata"]) == {"kernelspec"}


def test_adds_ids_for_nbformat_4_5():
    out, _ = cn.clean(nb([code("a")], minor=5), "T")
    assert all("id" in c for c in out["cells"])


def test_rejects_notebook_without_code():
    with pytest.raises(cn.CleaningError):
        cn.clean(nb([md("only text")]), "T")


def test_cli_refuses_overwrite_and_validates(tmp_path):
    src, dst = tmp_path / "in.ipynb", tmp_path / "out.ipynb"
    src.write_text(json.dumps(nb([md("intro"), code("a", [STREAM])], minor=5)))
    assert cn.main([str(src), str(dst), "--title", "T"]) == 0
    assert cn.main([str(src), str(dst), "--title", "T"]) == 2  # exists, no --force
    assert cn.main([str(src), str(dst), "--title", "T", "--force"]) == 0
    assert json.loads(dst.read_text())["cells"][0]["cell_type"] == "markdown"
