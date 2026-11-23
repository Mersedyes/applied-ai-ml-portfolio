# Re-auditing Seven Applied ML Projects: What the Evaluation Missed

*Mersedyes Livingston · November 2026*

## Purpose
In 2025 I completed seven applied ML projects spanning EDA, classical ML, ensembles, neural networks, computer vision, retrieval-augmented generation (RAG), and model deployment. Each was graded and presented as successful. In 2026 I re-audited them with the question I ask of production systems in my DevSecOps and compliance work: **does the evidence support the claim?**

## Method
1. **Audit from saved evidence.** I re-read every executed notebook's saved outputs and recomputed descriptive statistics and trivial baselines from the raw data.
2. **Integrity checks.**
   - Exact-duplicate hashing of all images in the vision dataset.
   - The deployed model artifact's detected classes compared with the documented model.
   - Presentation claims compared with notebook outputs.
3. **Re-runs under a corrected protocol** for the three projects where the audit found the most:
   - Personal loan: [`rerun_2026/RESULTS.md`](../02-personal-loan-classification/rerun_2026/RESULTS.md).
   - Visa certification: [`rerun_2026/easyvisa_rerun_2026.ipynb`](../03-easyvisa-ensemble-classification/rerun_2026/easyvisa_rerun_2026.ipynb).
   - SuperKart: [`rerun_2026/superkart_rerun_2026.ipynb`](../07-superkart-model-deployment/rerun_2026/superkart_rerun_2026.ipynb).

   The protocol is the same in all three:
   - A stratified split.
   - Selection by cross-validation on the training data only, with the metric fixed in advance.
   - A trivial baseline in every table.
   - The test set used once, with bootstrap confidence intervals.
   - Data hash and library versions recorded.

   The medical RAG system was rebuilt with a corrected evaluation design and run on a Colab T4 GPU: [`rerun_2026/medical_rag_2026.ipynb`](../05-medical-rag-llm/rerun_2026/medical_rag_2026.ipynb).

## Findings

| # | Problem | Where | Evidence |
|---|---|---|---|
| F1 | **No trivial baseline** | Visa certification (66.8% positive) | "Approve all" scores F1 0.801; the 2025 model's test F1 was 0.792. A tuned XGBoost was a constant predictor (precision = base rate) with the *highest* validation F1. **Re-run:** the best model beats the trivial rule by only 0.02 on positive-class F1, but by 0.20 on balanced accuracy (0.700 vs. 0.500). |
| | | Personal loan (9.6% positive) | Majority-class accuracy 0.904; 2025 pruned trees 0.779–0.823. **Re-run:** random forest 0.983. |
| | | Bank churn (20.4% positive) | Majority-class accuracy 0.796; final model 0.82 |
| F2 | **Test-set reuse** | Personal loan | Pre-pruning hyperparameters selected by *test* recall; pruning curve read on test recall. **Re-run** with CV-only selection: random forest, test recall 0.861 (95% CI 0.802–0.916) at precision 0.961. |
| F3 | **Selection inconsistent with the reported metric** | Bank churn | Selected model's validation recall 0.622 vs. 0.705 for an alternative with a smaller train–validation gap; only recall was tabulated |
| F4 | **Ceiling effect on a small test set** | Helmet detection | 95/95 correct; 95% upper bound on error ≈ 3.2% (rule of three); no exact duplicates across 631 images; shortcut learning untested |
| F5 | **Measurement validity of an LLM judge** | Medical RAG | n = 5 questions. The groundedness rubric references retrieved context the baseline never received. Judge model unrecorded; no human-agreement check; retrieval never measured. **Rebuild and run** (same passages for all systems, blind different-family judge): RAG ≥ base on groundedness for all 5 questions, but the paired gain ranges from +0.36 to +1.00 depending on how the judge's output is parsed, and every 95% interval includes 0. The judge returned one score per claim instead of one overall score in 14 of 15 groundedness ratings, and gave all 15 answers 3/3 for relevance, including three that stop mid-sentence. |
| F6 | **Deployment ≠ documentation** | SuperKart API | Docs report a random forest; the deployed artifact contains XGBoost classes; nothing linked the evaluated and the deployed artifact. **Re-run:** the two models are within one SD on CV error, so the choice was defensible but unrecorded. A model card now ties the artifact's SHA-256 to its data hash and metrics. |
| F7 | **Untested fairness claim, amplified disparity** | Visa certification | Historical certification rate by continent: 57.9%–79.2%. **Re-run:** with continent as an input, the model's predicted rates differ by 43 points between Europe and South America (historical gap: 21). False-denial rates run from 6.4% to 43.5% by continent. Without continent: 22%–28%, at a cost of 0.015 balanced accuracy. |
| F8 | **Hand-transcribed results drift** | FoodHub, personal loan, SuperKart | FoodHub slides said weekend deliveries were slower (data: weekday 28.3 min vs. weekend 22.5). The personal-loan deck's four-model comparison isn't reproducible from the saved notebook. SuperKart's reported R² ≈ 0.96 vs. 0.925 in the re-run. |
| F9 | **The importance method changes the story** | Visa certification | The 2025 deck ranked prevailing wage as the top predictor (impurity-based importance). **Re-run** permutation importance: education 0.147, job experience 0.064, annualized wage 0.004. |
| F10 | **Generalization beyond the sampled units was never tested** | SuperKart | 4 stores, each with fixed size, tier, and type. Random-split R² 0.925, but **leave-one-store-out R² is negative for every store** (−0.45 to −30.7). The deck's store-expansion recommendations aren't supported. |

## What I do differently now

| Problem | Practice |
|---|---|
| F1 | Every results table includes a trivial baseline. Under imbalance, report balanced accuracy, macro-F1, or PR-AUC alongside the business metric. |
| F2, F3 | Fix the selection criterion before training. Tune with cross-validation on training data only; touch the test set once. Report all candidate metrics, not just the winning one. |
| F4 | Report confidence intervals; when results are perfect, look for shortcuts (saliency maps, external test sets) before believing them. |
| F5 | Treat an LLM judge as a measurement instrument: specify it, keep it blind, constrain and validate its output format, check for ceiling effects, calibrate it against human raters, and evaluate retrieval separately from generation. Report how sensitive the result is to the scoring rule. |
| F6, F8 | Generate reported numbers from the same run that produced the artifact. Record artifact hash, data version, and metrics in a model card. Have CI block deployment when they don't match. |
| F7 | Disaggregate error rates by sensitive or proxy attributes before making any fairness claim; test the model with those features removed; compare predicted with historical disparities. |
| F9 | Prefer permutation importance on held-out data; state which method was used. |
| F10 | Match the validation split to the deployment question (grouped or temporal splits when predictions are for new units or future periods). |

## Open questions I want to research
1. **Evaluation literacy in organizational ML.** What reporting formats help non-specialist decision-makers tell a useful model from a trivial one, and do they change decisions?
2. **LLM-as-judge validity.** Under what conditions is an LLM judge a valid instrument for domain question answering, and how much human calibration is enough?
3. **Provenance and accountability for deployed models.** What minimum provenance (artifact identity, data lineage, evaluation evidence) makes a deployed model auditable, and can compliance-as-code pipelines enforce it without slowing delivery?
4. **Sensitive attributes in decision support.** How do protected or proxy attributes enter models used in public-facing adjudication, when do models amplify historical disparities, and what disclosures and tests should be required?

## Limits of this audit
- This is a self-audit of course projects.
- Three were re-run. Their 2025 originals (visa certification and SuperKart modeling notebooks) were not recovered, so the 2025 side of those comparisons relies on my submitted presentations.
- The medical RAG run is a pilot: 5 questions, no labeled reference pages, and no human ratings yet. Its judge scores show a direction, not an improvement.
- The continent analysis is a disparity check on one dataset, not a fairness audit. Removing a feature does not remove its proxies.
- Findings F1–F10 are not unusual in applied ML. That is what makes them worth studying.
