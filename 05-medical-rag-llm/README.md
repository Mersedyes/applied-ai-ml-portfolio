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

## Files
Notebook to be added. The submitted RAG notebook was not recovered; the configuration above is enough to rebuild it, and the evaluation will be re-run under the improved design.
