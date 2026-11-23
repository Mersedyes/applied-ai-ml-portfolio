"""Smoke test for medical_rag_2026.ipynb: runs it end to end with stub models and a synthetic PDF (no GPU, no downloads).

Verifies chunking, ChromaDB indexing, retrieval, judge parsing, statistics, and the human-calibration path.
It cannot verify the CUDA build, model downloads, or real model outputs.

Requires: nbformat nbclient ipykernel chromadb pypdf langchain-text-splitters reportlab pandas scikit-learn
Run:      python smoke_test.py
"""
import copy
import sys
import tempfile
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("medical_rag_2026.ipynb")
STUBS = r'''
import sys, types, hashlib
import numpy as np
st = types.ModuleType("sentence_transformers"); st.__version__ = "stub"
class SentenceTransformer:
    def __init__(self, name): self.name = name
    def encode(self, texts, normalize_embeddings=True):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return np.array(out)
st.SentenceTransformer = SentenceTransformer
hh = types.ModuleType("huggingface_hub"); hh.hf_hub_download = lambda repo_id, filename: f"/tmp/{filename}"
lc = types.ModuleType("llama_cpp"); lc.__version__ = "stub"
class Llama:
    def __init__(self, **kw): self.kw = kw
    def create_chat_completion(self, messages, **kw):
        roles = [m["role"] for m in messages]
        assert roles[-1] == "user"
        txt = messages[-1]["content"]
        c = '{"score": %d, "reason": "stub"}' % (len(txt) % 4) if "Respond only with JSON" in txt else "stub answer [p.1]"
        return {"choices": [{"message": {"content": c}}]}
lc.Llama = Llama
sys.modules.update({"sentence_transformers": st, "huggingface_hub": hh, "llama_cpp": lc})
'''

with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    c = canvas.Canvas(str(d / "medical_diagnosis_manual.pdf"), pagesize=letter)
    topics = ["sepsis antibiotics fluids vasopressors lactate", "appendicitis pain fever appendectomy surgery",
              "alopecia areata patchy hair loss corticosteroids", "traumatic brain injury rehabilitation monitoring",
              "leg fracture splint immobilization recovery"]
    for p in range(12):
        y = 720
        for line in range(30):
            c.drawString(40, y, f"Page {p+1} line {line}: " + topics[p % 5]); y -= 22
        c.showPage()
    c.save()

    nb = nbf.read(SRC, as_version=4)
    test = copy.deepcopy(nb)
    install_idx = next(i for i, cell in enumerate(test.cells) if cell.cell_type == "code" and "pip install" in cell.source)
    test.cells[install_idx] = nbf.v4.new_code_cell(STUBS)
    for i, cell in enumerate(test.cells):  # the CUDA 12 preload cell needs a GPU runtime: skip it
        if cell.cell_type == "code" and "CUDA12_LIBS" in cell.source:
            test.cells[i] = nbf.v4.new_code_cell("pass  # CUDA preload skipped in smoke test")
    NotebookClient(test, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(d)}}).execute()

    out = d / "rag_outputs"
    produced = sorted(p.name for p in out.iterdir())
    print("outputs:", produced)
    for need in ["answers.csv", "judge_scores.csv", "human_rating_blind.csv", "human_rating_key.csv", "results.json"]:
        assert need in produced, f"missing {need}"
    import pandas as pd
    s = pd.read_csv(out / "judge_scores.csv")
    assert len(s) == 5 * 3 * 2 and s.score.notna().all(), "expected 30 parsed judge scores"
    print("judge rows:", len(s), "| systems:", sorted(s.system.unique()))
    # second run: human ratings present -> kappa path
    blind = pd.read_csv(out / "human_rating_blind.csv")
    blind["groundedness_human"] = [i % 4 for i in range(len(blind))]
    blind["relevance_human"] = [(i + 1) % 4 for i in range(len(blind))]
    blind.to_csv(out / "human_ratings.csv", index=False)
    test2 = copy.deepcopy(test)
    NotebookClient(test2, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(d)}}).execute()
    kappa_out = [o for cell in test2.cells if cell.cell_type == "code" and "cohen_kappa_score" in cell.source
                 for o in cell.outputs if o.get("output_type") == "stream"]
    print("kappa cell output:", "".join(o["text"] for o in kappa_out).strip())
    assert "weighted kappa" in "".join(o["text"] for o in kappa_out)
print("SMOKE TEST PASSED")
