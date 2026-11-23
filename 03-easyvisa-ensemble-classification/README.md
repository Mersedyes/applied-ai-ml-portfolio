# 03 · Visa Certification: Ensemble Classification

**Question.** Can applicant and employer attributes predict whether a U.S. employment-based labor-certification case is certified, and which factors matter most?
**Data.** 25,480 cases with applicant education, experience, continent, employer size and age, region, wage, and wage unit. **66.8%** were certified. Course-provided; not redistributed.
**Tools.** pandas, scikit-learn (bagging, random forest, AdaBoost, gradient boosting, stacking), XGBoost, imbalanced-learn.

## Approach
1. **EDA and preprocessing:**
   - Corrected negative employee counts.
   - Checked outliers.
   - Annualized wages across hourly, weekly, and monthly units.
   - Encoded categorical variables.
2. **Model building:** trained six model families on original, oversampled, and undersampled training data. Cross-validated recall was used to shortlist models.
3. **Tuning:** randomized search on AdaBoost (oversampled), random forest (undersampled), gradient boosting, and XGBoost (oversampled), followed by a stacking comparison.

## Results (from my submitted presentation; notebook export incomplete)

| Model (post-tuning) | Val accuracy | Val recall | Val precision | Val F1 |
|---|---|---|---|---|
| XGBoost (oversampled) | 0.667 | 1.000 | 0.666 | 0.800 |
| **AdaBoost (oversampled)** | **0.716** | 0.826 | 0.766 | **0.795** |
| Random forest (undersampled) | 0.711 | 0.735 | **0.813** | 0.772 |

**Final model (AdaBoost), test set:** accuracy 0.713, recall 0.818, precision 0.766, F1 0.792.
Top predictors: prevailing wage, annualized wage, employer size, region of employment, and education.

## 2026 re-audit
- **The headline metric cannot tell a useful model from a trivial one.**
  - A rule that approves every case scores accuracy 0.668, recall 1.000, precision 0.668, and **F1 0.801**, above the final model's test F1 of 0.792.
  - The tuned XGBoost is effectively that rule: its validation precision (0.666) equals the base rate. It most likely got there because tuning selected `scale_pos_weight=5` on data that had already been oversampled to balance, correcting for imbalance twice.
  - AdaBoost was the right pick, but positive-class F1 is the wrong yardstick here. Balanced accuracy, macro-F1, recall on *denied* cases, or ROC-AUC, all reported against a trivial baseline, would show the model's real lift (accuracy +4.5 points over the baseline).
- **The fairness claim was untested.**
  - Certification rates by applicant continent range from 57.9% (South America) to 79.2% (Europe), and continent was a model input.
  - The original presentation said the model could improve fairness, but no fairness analysis was done.
  - A proper test would compare error rates by continent and retrain without continent (and its proxies) to measure the trade-off.
  - For a decision-support tool in a public adjudication process, that analysis is a prerequisite, not an extra.
- **Notebook status:** only a partial Colab export (33 pages, ending mid-tuning) was recovered. The executed notebook will be added if recovered; otherwise it will be re-run and the results above replaced.
