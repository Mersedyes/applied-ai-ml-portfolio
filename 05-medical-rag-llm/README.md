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

## 2026 rebuild
The submitted notebook was not recovered. [`rerun_2026/medical_rag_2026.ipynb`](rerun_2026/medical_rag_2026.ipynb) rebuilds the same system for Colab and implements the better design above:
- **Same passages for every system.** Base, prompt-engineered, and RAG answers are all judged against the same retrieved passages for each question, so the comparison is fair.
- **A blind judge from another model family.** Llama 3.1 8B (GGUF) judges Mistral's answers. It is named in the config and never sees which system wrote an answer.
- **Retrieval scored on its own:** hit@k against reference pages you label.
- **Paired statistics:** per-question RAG − base differences with bootstrap intervals, instead of "significant."
- **Human calibration:** a blind rating CSV, plus weighted Cohen's κ between judge and human once ratings exist.
- **Everything recorded:** models, chunking, temperature, seed, and library versions are saved to `results.json`.

## 2026 run (Colab T4, 23 Nov 2026)
**Setup.** Mistral-7B-Instruct v0.2 (Q4_K_M) generates; Llama 3.1 8B Instruct (Q4_K_M) judges. The manual's 4,114 pages became 35,067 chunks. Top-5 retrieval, temperature 0, seed 1. Versions are in [`results.json`](rerun_2026/rag_outputs/results.json).

**Judge means (0–3), 5 questions × 3 systems.** For groundedness the judge returned one score *per claim* instead of one overall score (see finding 3), so the table shows three ways to combine them. "First" is what the notebook reported.

| System | Groundedness: first | Groundedness: mean of claims | Groundedness: lowest claim | Relevance |
|---|---|---|---|---|
| Base model | 2.4 | 2.57 | 1.8 | 3.0 |
| Prompt-engineered | 2.4 | 2.28 | 1.2 | 3.0 |
| RAG | 3.0 | 2.93 | 2.8 | 3.0 |

**Paired RAG − base groundedness** (per question, bootstrap 95% interval over the 5 questions):

| Aggregation | Mean difference | 95% interval | RAG better / tied / worse |
|---|---|---|---|
| First claim (as reported) | +0.60 | [0.0, 1.8] | 1 / 4 / 0 |
| Mean of claims | +0.36 | [0.0, 0.79] | 2 / 3 / 0 |
| Lowest claim | +1.00 | [0.0, 2.2] | 2 / 3 / 0 |

Relevance: RAG − base = 0 on every question.

**What the run shows**
1. **Same direction as 2025, but no evidence of an effect yet.** RAG never scored below the base model on groundedness under any aggregation. The size of the gain depends on how the judge's output is parsed (+0.36 to +1.00), and every interval includes zero. Five questions can show a direction, not an improvement.
2. **The relevance score is at ceiling, so it measures nothing yet.** All 15 answers scored 3/3, including:
   - three base answers that hit the 512-token limit and stop mid-sentence (Q1, Q3, Q4);
   - the RAG answer to the critical-care sepsis question. It opens with general first-aid steps for shock (keep the patient warm, turn the head to one side) before covering antibiotics and source control. Those steps are faithful to the retrieved passages but don't match the question's ICU setting. This is my reading; a clinician should confirm it.
3. **The judge broke its output format.** It was asked for one JSON object. For 14 of 15 groundedness ratings it returned one object per claim (up to 6), and 13 of 30 outputs hit the 200-token limit. The notebook kept the first score, so the reported groundedness is the score of whichever claim the judge checked first. [`rescore_judge.py`](rerun_2026/rescore_judge.py) re-parses every claim and produced the sensitivity tables above. The stub smoke test passed because its fake judge always returns a single object: it verifies the plumbing, not the measurement.
4. **Grounded is not the same as appropriate.** The sepsis answer is well supported by its passages, yet part of it addresses a different setting. That is a retrieval problem, and hit@5 could not be computed: no reference pages were labeled before the run.
5. **The 2025 system prompt did not help groundedness.** The prompt-engineered system's mean was equal to or below the base model's under every aggregation (n = 5).

**Next run**
- Constrain the judge to a JSON schema (llama.cpp `response_format`), reject multi-object outputs, and log truncation.
- Label reference pages for each question, then grow the set to 50+ questions stratified by type.
- Collect blind human ratings with [`human_rating_blind.csv`](rerun_2026/rag_outputs/human_rating_blind.csv) (ideally from a clinician) and report weighted κ.
- Give the relevance rubric anchors that penalize setting mismatches and incomplete answers; raise the answer token limit or ask for concise answers.

[`rerun_2026/smoke_test.py`](rerun_2026/smoke_test.py) still runs the notebook end to end with stub models and a synthetic PDF, for checking code changes without a GPU.

## Presentation
[`presentation/Medical_RAG_Presentation.pdf`](presentation/Medical_RAG_Presentation.pdf) is my submitted deck (17 slides), unedited.

## Files
- [`rerun_2026/medical_rag_2026.ipynb`](rerun_2026/medical_rag_2026.ipynb): Colab notebook, executed 23 Nov 2026, with run notes.
- [`rerun_2026/rag_outputs/`](rerun_2026/rag_outputs/): generated results.
  - `answers.csv`: all 15 answers (5 questions × base, prompt-engineered, RAG).
  - `judge_scores_parsed.csv`: every judge rating, with each per-claim score and format flags.
  - `judge_sensitivity.json`: judge means and paired differences under each aggregation.
  - `results.json`: configuration, headline judge means, paired differences, retrieval table, versions.
  - `human_rating_blind.csv`, `human_rating_key.csv`: blind rating sheet and its answer key.
- [`rerun_2026/rescore_judge.py`](rerun_2026/rescore_judge.py): re-parses the raw judge outputs.
- [`rerun_2026/smoke_test.py`](rerun_2026/smoke_test.py): stub end-to-end test of the notebook's logic.
- [`presentation/Medical_RAG_Presentation.pdf`](presentation/Medical_RAG_Presentation.pdf): original 2025 presentation.

**Not published:** the manual, the ChromaDB index, and the notebook's raw `judge_scores.csv`. The index and the judge's free-text reasons both contain the manual's text verbatim. The answers include short paraphrases of the manual with page citations.
