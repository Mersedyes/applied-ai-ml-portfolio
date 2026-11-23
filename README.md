# Applied AI/ML Portfolio

**Mersedyes Livingston** · DevSecOps & compliance engineering · [LinkedIn](https://linkedin.com/in/mersedyes)

Seven applied machine-learning projects, from exploratory analysis to a deployed prediction API and an LLM retrieval system, completed between January and August 2025. In November 2026 I re-audited each one for **evaluation validity**: does the reported result actually support the claim being made?

## Why the re-audit

My professional work is making software systems *prove* what they claim: pipelines that generate evidence, controls that can be traced, and deployments that match their documentation. Applying the same lens to my own ML work surfaced the same kinds of gaps:

- **Metrics without baselines.** Two classifiers scored well on headline metrics yet barely beat (or fell below) a model that predicts the majority class every time.
- **Test-set reuse.** Hyperparameters chosen by test-set performance, which turns a held-out estimate into an optimistic one.
- **Perfect scores on small test sets.** 100% test accuracy on 95 images says less than it appears to.
- **Measurement validity in LLM evaluation.** An LLM-as-judge rubric that does not cleanly apply to the baseline it is compared against.
- **Deployment traceability.** A deployed model artifact that differs from the model the documentation reports.

These gaps are the questions I want to study: how organizations can **verify, document, and govern** the ML systems they deploy. The cross-project write-up is in [`docs/evaluation-audit.md`](docs/evaluation-audit.md).

## Projects

| # | Project | Methods | Result | Re-audit finding |
|---|---|---|---|---|
| 1 | [FoodHub order analysis](01-foodhub-eda/) | pandas, seaborn, EDA | Weekday deliveries 28.3 min vs 22.5 min on weekends | Corrected two findings in the original slides |
| 2 | [Personal-loan propensity](02-personal-loan-classification/) | Decision trees, pre/post-pruning, class weighting | Unpruned tree: test recall 0.93, precision 0.93 | Pruned trees were tuned on the test set and fall below the 0.904 majority-class accuracy |
| 3 | [Visa certification](03-easyvisa-ensemble-classification/) | Bagging, boosting, resampling, tuning, stacking | AdaBoost: test F1 0.792 | "Approve everyone" scores F1 0.801; fairness claim untested |
| 4 | [Bank churn](04-bank-churn-neural-network/) | Keras MLPs, SGD/Adam, dropout, SMOTE | Test churn recall 0.65, precision 0.55 | Model choice not consistent with the stated metric |
| 5 | [Medical RAG assistant](05-medical-rag-llm/) | Mistral-7B, MiniLM embeddings, ChromaDB, LLM-as-judge | Groundedness 1.6 → 2.6 (0–3 scale) | n = 5; judge unnamed; rubric validity |
| 6 | [Helmet detection](06-helmet-detection-cv/) | CNN, VGG-16 transfer learning, augmentation | Test accuracy 1.00 (n = 95) | Ceiling effect; 95% bound ≈ 96.8% |
| 7 | [SuperKart sales API](07-superkart-model-deployment/) | sklearn pipelines, RF/XGBoost, Flask, Docker, Streamlit | Test R² ≈ 0.96 (reported) | Deployed artifact ≠ documented model |

Results come from executed notebook outputs or from recomputation on the raw data. Where a notebook is not yet included (projects 3, 5, 7), figures come from my submitted presentation and are labeled that way.

## Provenance and scope

These projects were completed in **The University of Texas at Austin Post Graduate Program in Artificial Intelligence and Machine Learning: Business Applications** (delivered with Great Learning; January 2025 cohort).

- Projects used course-provided "low-code" notebook templates. I completed the implementation, ran the experiments, and wrote the analysis.
- Course instructions and problem text were removed with [`tools/clean_notebook.py`](tools/clean_notebook.py). Code, outputs, and my observations are unchanged; the tool refuses to write a notebook whose code-cell or output count changes.
- **Not redistributed:** datasets, course slides, lecture material, and my original presentation decks (they use the course template). Notebook cells that load data will not run as-is.
- The 2026 re-audit sections are new analysis written for this repository.

## Repository layout

```
├── 01-…07-*/            one folder per project: README.md (+ cleaned notebook where available)
├── docs/evaluation-audit.md   cross-project findings and open research questions
├── tools/clean_notebook.py    removes course scaffolding from notebooks
├── scripts/check_repo.py      pre-publish checks (sizes, data files, licensing notices, secrets)
├── tests/                     pytest suite for both tools
└── .github/workflows/checks.yml   runs the tests and checks on every push
```

## Quality checks

```bash
python -m pip install nbformat pytest
python -m pytest -q          # tool tests
python scripts/check_repo.py # fails on data files, >25 MB files, course notices, or secret-like strings
```

## Citation

See [`CITATION.cff`](CITATION.cff), or:

> Livingston, M. (2026). *Applied AI/ML Portfolio: Seven Projects with an Evaluation Re-audit* [Computer software]. GitHub. https://github.com/Mersedyes/applied-ai-ml-portfolio
