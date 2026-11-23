# 05 · Medical Question Answering with Retrieval-Augmented Generation

**Question.** Does grounding an open LLM in a trusted medical reference make its answers to clinical questions more grounded and relevant than the model alone or with prompt engineering?
**Source.** A widely used medical reference manual (PDF). It is copyrighted, so neither the manual nor the vector store built from it is redistributed.

> Research prototype for a course project. Not medical advice and not validated for clinical use.

## System

| Component | Choice |
|---|---|
| Generator | Mistral-7B-Instruct |
| Chunking | Recursive character splitter, 500 characters, 100 overlap |
| Embeddings | all-MiniLM-L6-v2 |
| Vector store | ChromaDB (persisted) |
| Retrieval | Top-k = 5 chunks, concatenated with the question into one prompt |
| Prompting experiments | System-prompt variants; temperature 0.0 / 0.3 / 0.7; top-p 0.9 |

## Evaluation
Five clinical questions (sepsis management, appendicitis, patchy hair loss, traumatic brain injury, leg fracture) were answered by the base LLM and by the RAG system. Each answer was scored 0–3 by an LLM judge for:
- **Groundedness:** how well the answer is supported by the retrieved context.
- **Relevance:** how directly and completely it answers the question.

Results from my submitted presentation:

| Question | Groundedness: base → RAG | Relevance: base → RAG |
|---|---|---|
| Sepsis | 2 → 3 | 2 → 3 |
| Appendicitis | 2 → 3 | 3 → 3 |
| Hair loss | 1 → 2 | 2 → 2 |
| Brain injury | 1 → 2 | 2 → 3 |
| Leg fracture | 2 → 3 | 3 → 3 |
| **Mean** | **1.6 → 2.6** | **2.4 → 2.8** |

## 2026 re-audit
- **The groundedness rubric doesn't apply cleanly to the baseline.** It scores support "by the retrieved context," but the base model received no context. Its scores therefore measure something different: agreement with whatever context the judge was shown. A fair comparison scores both systems against the same reference passages.
- **The judge is unspecified and uncalibrated.** The judge model isn't recorded. If the same model judged its own outputs, self-preference bias is possible. There is no check of agreement with a human (ideally clinician) rater.
- **The sample is too small to support "significant."** Five questions on a 4-point integer scale show a consistent direction (RAG ≥ base on every item) but not statistical significance. The original slides' claims about reduced hallucination and clinical trust weren't measured.
- **Retrieval itself was never evaluated.** Answer quality depends on whether the right chunks were retrieved. Retrieval hit rate or recall@k on labeled questions would separate retrieval failures from generation failures.
- **A better design:**
  - 50+ questions stratified by type, with reference answers.
  - Blind human ratings on a subset to calibrate the LLM judge (report agreement, e.g., Cohen's κ).
  - Separate retrieval metrics.
  - Paired comparison across base, prompt-engineered, and RAG systems.

## 2026 rebuild (to run on a GPU)
The submitted notebook was not recovered. [`rerun_2026/medical_rag_2026.ipynb`](rerun_2026/medical_rag_2026.ipynb) rebuilds the same system for Colab and implements the better design above:
- **Same passages for every system.** Base, prompt-engineered, and RAG answers are all judged against the same retrieved passages for each question, so the comparison is fair.
- **A blind judge from another model family.** Llama 3.1 8B (GGUF) judges Mistral's answers. It is named in the config and never sees which system wrote an answer.
- **Retrieval scored on its own:** hit@k against reference pages you label.
- **Paired statistics:** per-question RAG − base differences with bootstrap intervals, instead of "significant."
- **Human calibration:** a blind rating CSV, plus weighted Cohen's κ between judge and human once ratings exist.
- **Everything recorded:** models, chunking, temperature, seed, and library versions are saved to `results.json`.

**Status: not yet executed.** It needs a Colab GPU runtime and your copy of the manual. [`rerun_2026/smoke_test.py`](rerun_2026/smoke_test.py) runs it end to end with stub models and a synthetic PDF. That verifies chunking, indexing, retrieval, judge parsing, statistics, and the calibration step. It cannot verify the CUDA build, model downloads, or real outputs; those are checked on the first Colab run.

## Presentation
[`presentation/Medical_RAG_Presentation.pdf`](presentation/Medical_RAG_Presentation.pdf) is my submitted deck (17 slides), unedited.

## Files
- [`rerun_2026/medical_rag_2026.ipynb`](rerun_2026/medical_rag_2026.ipynb): Colab notebook (outputs added after it is run).
- [`rerun_2026/smoke_test.py`](rerun_2026/smoke_test.py): stub end-to-end test of the notebook's logic.
- [`presentation/Medical_RAG_Presentation.pdf`](presentation/Medical_RAG_Presentation.pdf): original 2025 presentation.
