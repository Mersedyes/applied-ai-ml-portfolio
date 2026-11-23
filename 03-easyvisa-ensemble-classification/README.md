# 03 · Visa Certification: Ensemble Classification and a 2026 Re-run

**Question.** Can applicant and employer attributes predict whether a U.S. employment-based labor-certification case is certified, and which factors matter most?
**Data.** 25,480 cases with applicant education, experience, continent, employer size and age, region, wage, and wage unit. **66.8%** were certified. Course-provided; not redistributed.
**Tools.** pandas, scikit-learn (bagging, random forest, AdaBoost, gradient boosting, stacking), XGBoost, imbalanced-learn.

## 2025 project
1. **EDA and preprocessing:** corrected negative employee counts, checked outliers, annualized wages across units, and encoded categoricals.
2. **Model building:** six model families on original, oversampled, and undersampled data, shortlisted by cross-validated recall.
3. **Tuning:** randomized search on AdaBoost (oversampled), random forest (undersampled), gradient boosting, and XGBoost (oversampled), then a stacking comparison.

Results as reported in my submitted presentation (the executed 2025 notebook was not recovered):

| Model (post-tuning) | Val accuracy | Val recall | Val precision | Val F1 |
|---|---|---|---|---|
| XGBoost (oversampled) | 0.667 | 1.000 | 0.666 | 0.800 |
| **AdaBoost (oversampled), final** | **0.716** | 0.826 | 0.766 | **0.795** |
| Random forest (undersampled) | 0.711 | 0.735 | **0.813** | 0.772 |

**Final model, test set:** accuracy 0.713, recall 0.818, precision 0.766, F1 0.792.

## 2026 re-audit
- **The headline metric could not tell a useful model from a trivial one.**
  - Approving every case scores F1 0.801 on the certified class, above the final model's 0.792.
  - The tuned XGBoost *was* that rule: its validation precision (0.666) equals the base rate. Most likely tuning chose `scale_pos_weight=5` on data already oversampled to balance, correcting for imbalance twice.
- **The fairness claim was untested.** The presentation says the model "can enhance fairness" and lists applicant origin as a key driver, but no fairness analysis was done.

## 2026 re-run
[`rerun_2026/easyvisa_rerun_2026.ipynb`](rerun_2026/easyvisa_rerun_2026.ipynb) is an executed notebook that keeps the 2025 preprocessing and fixes the protocol:
- Original class balance, with the decision threshold set explicitly on out-of-fold training predictions.
- Selection by 5-fold CV ROC-AUC on the training set only.
- An "approve everyone" baseline in every comparison.
- The test set used once, with bootstrap 95% confidence intervals.

**Selection (5-fold CV on training data):** random forest, ROC-AUC 0.787 ± 0.004. Gradient boosting (0.783), XGBoost (0.782), logistic regression (0.772), and AdaBoost (0.771) follow.

**Test set (7,644 cases):**

| | Accuracy | Balanced acc. | Macro-F1 | F1 (certified) | Recall: denied | ROC-AUC |
|---|---|---|---|---|---|---|
| Approve everyone | 0.668 | 0.500 | 0.400 | **0.801** | 0.000 | — |
| Random forest, threshold 0.5 | 0.745 | 0.678 | 0.689 | 0.821 | 0.479 | 0.777 |
| Random forest, threshold 0.646 (set on training folds) | 0.724 | **0.700** [0.689, 0.711] | **0.696** | 0.789 | **0.629** | 0.777 [0.767, 0.788] |

- On positive-class F1 the model beats the trivial rule by only 0.02; on balanced accuracy it is 0.18–0.20 ahead. The metric choice decides whether the model looks useless or useful.
- The threshold matters: moving from 0.5 to 0.646 raises recall on denied cases from 48% to 63%, at a cost in certified-case recall.

**Error rates by applicant continent (test set, threshold set on training folds):**

| Continent | n | Actual certified | Predicted certified, **with** continent | False-denial rate, with | Predicted certified, **without** continent | False-denial rate, without |
|---|---|---|---|---|---|---|
| Asia | 5,061 | 0.651 | 0.631 | 0.234 | 0.644 | 0.225 |
| Europe | 1,104 | 0.792 | 0.841 | **0.064** | 0.668 | 0.247 |
| North America | 1,021 | 0.634 | 0.513 | 0.374 | 0.631 | 0.261 |
| South America | 237 | 0.582 | 0.414 | **0.435** | 0.553 | 0.283 |
| Africa | 154 | 0.708 | 0.630 | 0.257 | 0.649 | 0.220 |
| Oceania | 67 | 0.627 | 0.582 | 0.262 | 0.582 | 0.262 |

- **With continent, the model amplifies the historical gap.**
  - Europe vs. South America: actual certification rates differ by 21 points; predicted rates differ by 43.
  - Certified applicants from South America are wrongly predicted "denied" 43.5% of the time; from Europe, 6.4%.
- **Without continent, the spread shrinks.** False-denial rates range from 22% to 28%, and false-approval rates from 32% to 44% (vs. 20% to 48% with continent).
- **The cost is small:** test ROC-AUC 0.777 → 0.760 and balanced accuracy 0.700 → 0.685.
- **Limits:** removing a feature does not remove its proxies. Equal error rates are one fairness criterion among several, and they trade off against each other. Small groups (Africa, Oceania) have wide uncertainty. This is a disparity check, not a full fairness audit.

**What drives the model (permutation importance, ROC-AUC drop on test):**
- Education 0.147, job experience 0.064, continent 0.031, region 0.027, wage unit 0.022; annualized wage only 0.004.
- The 2025 presentation ranked prevailing wage as the top predictor, using impurity-based importance, which is known to favor high-cardinality continuous features. The importance method changes the story.

## Presentation
[`presentation/EasyVisa_Presentation.pdf`](presentation/EasyVisa_Presentation.pdf) is my submitted deck (72 slides), unedited. The re-audit and re-run above supersede its fairness statements and feature-importance ranking.

## Files
- [`rerun_2026/easyvisa_rerun_2026.ipynb`](rerun_2026/easyvisa_rerun_2026.ipynb): executed 2026 re-run (needs `data/EasyVisa.csv` to re-execute).
- [`rerun_2026/results.json`](rerun_2026/results.json): all re-run numbers, with the data hash and library versions.
- [`presentation/EasyVisa_Presentation.pdf`](presentation/EasyVisa_Presentation.pdf): original 2025 presentation.
