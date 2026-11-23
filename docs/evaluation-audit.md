# Re-auditing Seven Applied ML Projects: What the Evaluation Missed

*Mersedyes Livingston · November 2026*

## Purpose
In 2025 I completed seven applied ML projects spanning EDA, classical ML, ensembles, neural networks, computer vision, retrieval-augmented generation (RAG), and model deployment. Each was graded and presented as successful. In 2026 I re-audited them with the question I ask of production systems in my DevSecOps and compliance work: **does the evidence support the claim?**

## Method
- **No retraining.** I re-read every executed notebook's saved outputs and recomputed descriptive statistics and trivial baselines from the raw data.
- **Integrity checks on inputs:** exact-duplicate hashing of all images in the vision dataset.
- **Deployment check:** compared the deployed model artifact's detected classes with the model the documentation reports.
- **Documentation check:** compared presentation claims with notebook outputs.

## Findings

| # | Problem | Where | Evidence |
|---|---|---|---|
| F1 | **No trivial baseline** | Visa certification (66.8% positive) | "Approve all" scores F1 0.801; final model test F1 0.792. A tuned XGBoost was a constant predictor (precision = base rate) and had the *highest* validation F1. |
| | | Personal loan (9.6% positive) | Majority-class accuracy 0.904; pruned trees 0.779–0.823 |
| | | Bank churn (20.4% positive) | Majority-class accuracy 0.796; final model 0.82 |
| F2 | **Test-set reuse** | Personal loan | Pre-pruning hyperparameters selected by *test* recall; pruning curve read on test recall |
| F3 | **Selection inconsistent with the reported metric** | Bank churn | Selected model's validation recall 0.622 vs. 0.705 for an alternative with a smaller train–validation gap; only recall was tabulated |
| F4 | **Ceiling effect on a small test set** | Helmet detection | 95/95 correct; 95% upper bound on error ≈ 3.2% (rule of three); no exact duplicates across 631 images; shortcut learning untested |
| F5 | **Measurement validity of an LLM judge** | Medical RAG | n = 5 questions. The groundedness rubric references retrieved context the baseline never received. Judge model unrecorded; no human-agreement check; retrieval quality never measured. |
| F6 | **Deployment ≠ documentation** | SuperKart API | Docs report a random forest; the deployed artifact contains XGBoost classes; no hash or model card linked the evaluated and the deployed artifact |
| F7 | **Untested fairness claim** | Visa certification | Certification rate by applicant continent: 57.9%–79.2%. Continent was a model input; "improves fairness" was asserted without measurement. |
| F8 | **Hand-transcribed results drift** | FoodHub | Slides said weekend deliveries were slower; data: weekday 28.3 min vs. weekend 22.5 min |

## What I do differently now

| Problem | Practice |
|---|---|
| F1 | Every results table includes a trivial baseline. Under imbalance, report balanced accuracy, macro-F1, or PR-AUC alongside the business metric. |
| F2, F3 | Fix the selection criterion before training. Tune with cross-validation on training data only; touch the test set once. Report all candidate metrics, not just the winning one. |
| F4 | Report confidence intervals; when results are perfect, look for shortcuts (saliency maps, external test sets) before believing them. |
| F5 | Treat an LLM judge as a measurement instrument: specify it, calibrate it against human raters (agreement statistics), and evaluate retrieval separately from generation. |
| F6, F8 | Generate reported numbers from the same run that produced the artifact. Record artifact hash, data version, and metrics in a model card. Have CI block deployment when they don't match. |
| F7 | Disaggregate error rates by sensitive or proxy attributes before making any fairness claim; test the model with those features removed. |

## Open questions I want to research
1. **Evaluation literacy in organizational ML.** What reporting formats help non-specialist decision-makers tell a useful model from a trivial one, and do they change decisions?
2. **LLM-as-judge validity.** Under what conditions is an LLM judge a valid instrument for domain question answering, and how much human calibration is enough?
3. **Provenance and accountability for deployed models.** What minimum provenance (artifact identity, data lineage, evaluation evidence) makes a deployed model auditable, and can compliance-as-code pipelines enforce it without slowing delivery?
4. **Sensitive attributes in decision support.** How do protected or proxy attributes enter models used in public-facing adjudication, and what disclosures and tests should be required?

## Limits of this audit
- This is a self-audit of course projects, using saved outputs; no model was retrained.
- Three notebooks (visa certification, medical RAG, SuperKart modeling) were not recovered, so those findings rely on my submitted presentations and, for SuperKart, the deployed service.
- Findings F1–F8 are not unusual in applied ML. That is what makes them worth studying.
