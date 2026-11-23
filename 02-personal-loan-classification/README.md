# 02 · Personal-Loan Propensity: Decision Trees and a 2026 Re-run

**Question.** Which existing bank customers are likely to accept a personal-loan offer, and what characterizes them?
**Data.** 5,000 customers with demographics, income, credit-card spending, education, and account holdings. **9.6%** accepted the previous campaign's offer. Course-provided; not redistributed.
**Tools.** pandas, seaborn, scikit-learn: decision trees and cost-complexity pruning (2025); pipelines, stratified CV, logistic regression, KNN, random forest, and bootstrap CIs (2026 re-run).

## Approach
1. EDA and data cleaning, income binning, and scaling.
2. 70/30 train/test split (`random_state=1`).
3. Three decision trees:
   - **Default:** unconstrained sklearn tree.
   - **Pre-pruned:** a manual search over depth (2–6), maximum leaf nodes, and minimum split size with balanced class weights. It selected depth 2.
   - **Post-pruned:** cost-complexity pruning with `ccp_alpha = 0.047` and class weights {0: 0.15, 1: 0.85}. The result is a single split on income.

## Results (test set, from notebook outputs)

| Model | Structure | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Default tree | Fully grown | **0.986** | 0.933 | **0.927** | **0.930** |
| Pre-pruned | Depth 2 (income, CCAvg, family) | 0.779 | **1.000** | 0.310 | 0.474 |
| Post-pruned | One split on income | 0.823 | 0.906 | 0.350 | 0.505 |
| *Baseline: predict "no" for everyone* | — | *0.904* | *0.000* | — | — |

Feature importance (default tree): income 0.31, family size 0.26, education (graduate 0.17, advanced/professional 0.15), credit-card spending 0.05.

## 2026 re-audit
- **The test set was used for model selection.** The pre-pruning search chose its hyperparameters by *test* recall ("best test recall: 1.0"), and the pruning curve was inspected on test recall. So the pruned models' test scores are optimistically biased: they are no longer held-out estimates. Selection should use cross-validation on the training set, with the test set touched once at the end.
- **Both pruned trees fall below the majority-class baseline on accuracy.** Class weighting shifts the decision threshold toward the minority class. That buys recall but costs precision: about 65% of the post-pruned model's offers would go to non-acceptors. That model is effectively the rule *"make an offer if income exceeds a threshold"*. That rule is a useful, transparent baseline, but it shouldn't be presented as a tuned model.
- **"Overfitting" was judged on the wrong evidence.** The default tree scores 1.0 on training data, which motivated pruning. But its test performance (0.986 accuracy, 0.93 recall and precision) is the best of the three. A perfect training score alone is not evidence of harmful overfitting; the train–test gap is.
- **The right model depends on a cost assumption that was never stated.** If a missed acceptor costs much more than a wasted offer, the post-pruned tree is defensible. Otherwise the default tree is better on every metric except recall. A cost-weighted metric or a tuned decision threshold would make that trade-off explicit.
- **Methodological gaps:**
  - The split was not stratified despite a 9.6% positive rate.
  - There was no cross-validation estimate of variance.
  - Conclusions are summarized in this README rather than in the notebook.
  - 52 rows had negative years of experience. They were sign-flipped on the assumption of a data-entry error, an assumption that should be stated rather than implicit.

## 2026 re-run: a sound model comparison
The 2025 notebook compared decision trees only. [`rerun_2026/model_comparison.py`](rerun_2026/model_comparison.py) adds logistic regression, KNN, and a random forest, under a protocol that fixes the problems above:
- Same preprocessing as the notebook.
- Stratified 70/30 split.
- Hyperparameters and model choice set by **5-fold cross-validation on the training set only**, using a selection metric fixed in advance (PR-AUC).
- A majority-class baseline in every table.
- The test set used **once**, with bootstrap 95% confidence intervals.

Every number comes from the script's generated report, [`rerun_2026/RESULTS.md`](rerun_2026/RESULTS.md), which also records the data hash and library versions. Headline results:

| | CV PR-AUC (train) | Test recall [95% CI] | Test precision [95% CI] | Test accuracy |
|---|---|---|---|---|
| Baseline (always "no") | 0.096 | 0.000 | — | 0.904 |
| Logistic regression | 0.863 ± 0.026 | — | — | — |
| KNN | 0.880 ± 0.039 | — | — | — |
| Decision tree (depth 6, min leaf 10) | 0.960 ± 0.013 | — | — | — |
| **Random forest (selected)** | **0.982 ± 0.008** | **0.861** [0.802, 0.916] | **0.961** [0.925, 0.992] | 0.983 |

Only the selected model and the baseline were scored on the test set; that is the point of a held-out set.

What the re-run shows:
- **The 2025 trade-off was unnecessary.** Pruning with class weights bought recall by cutting precision to 0.31–0.35. A properly selected model reaches about 0.86–0.88 recall while keeping precision around 0.96. (The 2025 split was not stratified, so its numbers are not directly comparable.)
- **A recall target set on training data can miss on new data.** The threshold chosen on out-of-fold training predictions to hit 90% recall (0.443) gave 0.882 test recall (95% CI 0.828–0.931). An operating target needs a margin and monitoring after deployment.
- **The threshold matters as much as the model.** KNN ranks customers well (cross-validated PR-AUC 0.88) but at the default 0.5 threshold catches only 30% of acceptors in cross-validation.
- **Interpretability vs. accuracy is a real but small trade-off here.** The depth-6 tree is within about 0.02 PR-AUC of the forest and can be explained to a marketing team rule by rule.

To reproduce (data not included):

```zsh
mkdir -p data
cp "<course folder>/Machine Learning/Project/Loan_Modelling.csv" data/
python3 -m pip install numpy pandas scikit-learn
python3 02-personal-loan-classification/rerun_2026/model_comparison.py
```

`data/` is gitignored. Small differences can appear across library versions, so compare the provenance block in `RESULTS.md`.

## Presentation
[`presentation/Personal_Loan_Presentation.pdf`](presentation/Personal_Loan_Presentation.pdf) is my submitted deck (64 slides), unedited. Its four-model comparison chart, cross-validation statement, and final-tree parameters can't be reproduced from the saved 2025 notebook, which covers decision trees only. The 2026 re-run above is the verified comparison.

## Files
- [`personal_loan_decision_tree.ipynb`](personal_loan_decision_tree.ipynb): cleaned 2025 notebook (course instructions removed; code and outputs unchanged).
- [`rerun_2026/model_comparison.py`](rerun_2026/model_comparison.py): 2026 re-run script (tested in `tests/test_loan_rerun.py`).
- [`rerun_2026/RESULTS.md`](rerun_2026/RESULTS.md) and [`results.json`](rerun_2026/results.json): generated outputs of the re-run.
- [`presentation/Personal_Loan_Presentation.pdf`](presentation/Personal_Loan_Presentation.pdf): original 2025 presentation.
