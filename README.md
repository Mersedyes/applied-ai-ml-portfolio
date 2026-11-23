# Applied AI/ML Portfolio

**Mersedyes Livingston** · DevSecOps & compliance engineering · [LinkedIn](https://linkedin.com/in/mersedyes)

Seven applied machine-learning projects, from exploratory analysis to a deployed prediction API and an LLM retrieval system, completed between January and August 2025. Each includes the original analysis and the presentation I gave.

In November 2026 I re-audited each one for **evaluation validity**: does the reported result actually support the claim being made? I then re-ran three of them (personal loan, visa certification, SuperKart) under a corrected protocol.

## Why the re-audit

My professional work is making software systems *prove* what they claim: pipelines that generate evidence, controls that can be traced, and deployments that match their documentation. Applying the same lens to my own ML work surfaced the same kinds of gaps:

- **Metrics without baselines.** Two classifiers scored well on headline metrics yet barely beat (or fell below) a model that predicts the majority class every time.
- **Test-set reuse.** Hyperparameters chosen by test-set performance, which turns a held-out estimate into an optimistic one.
- **Perfect scores on small test sets.** 100% test accuracy on 95 images says less than it appears to.
- **Measurement validity in LLM evaluation.** An LLM-as-judge rubric that does not cleanly apply to the baseline it is compared against.
- **Deployment traceability.** A deployed model artifact that differs from the model the documentation reports.
- **Unstated generalization limits and amplified disparities.**
  - A sales model with R² 0.93 on a random split cannot predict a store it hasn't seen.
  - A visa model doubles the historical approval gap between applicant continents.

These gaps are the questions I want to study: how organizations can **verify, document, and govern** the ML systems they deploy. The cross-project write-up is in [`docs/evaluation-audit.md`](docs/evaluation-audit.md).

## Projects

| # | Project | Methods | Result | Re-audit finding |
|---|---|---|---|---|
| 1 | [FoodHub order analysis](01-foodhub-eda/) | pandas, seaborn, EDA | Weekday deliveries 28.3 min vs 22.5 min on weekends | Corrected two findings in the original slides |
| 2 | [Personal-loan propensity](02-personal-loan-classification/) | Decision trees, pruning; **2026 re-run** with LR, KNN, random forest, CV-only selection, bootstrap CIs | Re-run (random forest): test recall 0.861, precision 0.961 | 2025 models were tuned on the test set; the re-run fixes the protocol |
| 3 | [Visa certification](03-easyvisa-ensemble-classification/) | Bagging, boosting, stacking; **2026 re-run** with baselines, balanced metrics, per-group error rates, ablation | Re-run (random forest): balanced accuracy 0.700, ROC-AUC 0.777 | With continent as an input, false-denial rates range 6%–44% by continent; without it, 22%–28% at −1.5 pts balanced accuracy |
| 4 | [Bank churn](04-bank-churn-neural-network/) | Keras MLPs, SGD/Adam, dropout, SMOTE | Test churn recall 0.65, precision 0.55 | Model choice not consistent with the stated metric |
| 5 | [Medical RAG assistant](05-medical-rag-llm/) | Mistral-7B, MiniLM embeddings, ChromaDB, LLM-as-judge; **2026 rebuild** for Colab | Groundedness 1.6 → 2.6 (0–3 scale, as presented) | n = 5; judge unnamed; rubric validity. The rebuild fixes the design (not yet executed) |
| 6 | [Helmet detection](06-helmet-detection-cv/) | CNN, VGG-16 transfer learning, augmentation | Test accuracy 1.00 (n = 95) | Ceiling effect; 95% bound ≈ 96.8% |
| 7 | [SuperKart sales API](07-superkart-model-deployment/) | sklearn pipelines, RF/XGBoost, Flask, Docker, Streamlit; **2026 re-run** with model card + artifact hash | Re-run: test R² 0.925, RMSE 292 | Deployed artifact ≠ documented model; leave-one-store-out R² < 0 for every store |

Results come from executed notebooks or scripts in this repository, or from recomputation on the raw data. 2025 figures that exist only in a presentation are labeled "as presented."

## Provenance and scope

These projects were completed in **The University of Texas at Austin Post Graduate Program in Artificial Intelligence and Machine Learning: Business Applications** (delivered with Great Learning; January 2025 cohort).

- Projects used course-provided "low-code" notebook templates. I completed the implementation, ran the experiments, and wrote the analysis.
- Course instructions and problem text were removed with [`tools/clean_notebook.py`](tools/clean_notebook.py). Code, outputs, and my observations are unchanged; the tool refuses to write a notebook whose code-cell or output count changes.
- **Presentations:** each project's `presentation/` folder holds the deck I submitted, unedited. The decks are my work on the course's slide template, so they carry its logo and footer. Where a deck's claims differ from the code, the project README says so and points to the verified result.
- **Not redistributed:** datasets, course slides and lecture material. To re-run, place the CSVs in `data/` (gitignored).
- The 2026 re-audits, re-runs (`rerun_2026/` folders), and tooling are new work written for this repository.

## Repository layout

```
├── 01-…07-*/                  one folder per project:
│   ├── README.md              question, approach, results, re-audit
│   ├── *.ipynb                cleaned 2025 notebook (projects 1, 2, 4, 6)
│   ├── rerun_2026/            2026 re-run notebook/script + generated results (projects 2, 3, 5, 7)
│   └── presentation/          original submitted deck (PDF)
├── docs/evaluation-audit.md   cross-project findings and open research questions
├── tools/clean_notebook.py    removes course scaffolding from notebooks
├── scripts/check_repo.py      pre-publish checks (sizes, data files, licensing notices, secrets)
├── tests/                     pytest suite (cleaner, checker, loan re-run)
└── .github/workflows/checks.yml   runs the tests and checks on every push
```

## Quality checks

```zsh
python3 -m pip install nbformat pytest numpy pandas scikit-learn
python3 -m pytest -q
python3 scripts/check_repo.py .
```

The tests cover the notebook cleaner, the pre-publish checker, and the Personal Loan re-run. The checker fails on:
- data files, files over 25 MB, PDFs outside a project's `presentation/` folder;
- course licensing notices in text files;
- secret-like strings.

In a git repo it checks exactly what git would publish, so gitignored local data is allowed. The RAG notebook has its own stub smoke test (`05-medical-rag-llm/rerun_2026/smoke_test.py`).

## Citation

See [`CITATION.cff`](CITATION.cff), or:

> Livingston, M. (2026). *Applied AI/ML Portfolio: Seven Projects with an Evaluation Re-audit* [Computer software]. GitHub. https://github.com/Mersedyes/applied-ai-ml-portfolio
