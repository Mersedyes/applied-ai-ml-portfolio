# 02 · Personal-Loan Propensity: Decision-Tree Classification

**Question.** Which existing bank customers are likely to accept a personal-loan offer, and what characterizes them?
**Data.** 5,000 customers with demographics, income, credit-card spending, education, and account holdings. **9.6%** accepted the previous campaign's offer. Course-provided; not redistributed.
**Tools.** pandas, scikit-learn (DecisionTreeClassifier, GridSearchCV, cost-complexity pruning), seaborn.

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

## Files
- [`personal_loan_decision_tree.ipynb`](personal_loan_decision_tree.ipynb): cleaned notebook (course instructions removed; code and outputs unchanged).
