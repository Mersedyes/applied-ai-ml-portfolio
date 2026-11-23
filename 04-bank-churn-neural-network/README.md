# 04 · Bank Customer Churn: Neural-Network Classification

**Question.** Which bank customers are likely to leave, so retention efforts can be targeted?
**Data.** 10,000 customers with demographics, tenure, balance, products, activity, and salary. **20.4%** churned. Course-provided; not redistributed.
**Tools.** TensorFlow/Keras, scikit-learn, imbalanced-learn (SMOTE).

## Approach
1. EDA, encoding, scaling, and train/validation/test split.
2. Six multilayer-perceptron variants: SGD vs. Adam optimizer, with and without dropout, and with and without SMOTE oversampling of the training set.
3. The stated metrics were recall on churners (missing a customer who leaves is the costly error) and F1 (because of class imbalance).

## Results (from notebook outputs)

| Model | Train recall | Validation recall | Train − val gap |
|---|---|---|---|
| NN, SGD | 0.049 | 0.044 | 0.005 |
| NN, Adam | 0.513 | 0.410 | 0.103 |
| NN, Adam + dropout | 0.536 | 0.462 | 0.074 |
| NN, SMOTE + SGD | 0.745 | **0.705** | **0.040** |
| NN, SMOTE + Adam | 0.930 | 0.641 | 0.289 |
| **NN, SMOTE + Adam + dropout (selected)** | 0.884 | 0.622 | 0.262 |

**Test set, selected model:**

| | Precision | Recall | F1 |
|---|---|---|---|
| Churned | 0.55 | 0.65 | 0.59 |
| Stayed | 0.91 | 0.86 | 0.88 |

Accuracy 0.82 (majority-class baseline: 0.796); macro-F1 0.74.

## 2026 re-audit
- **The selection isn't supported by the comparison that was reported.** The model comparison table shows recall only. On it, SMOTE + SGD beats the selected model on validation (0.705 vs. 0.622) and generalizes better (train–validation gap 0.040 vs. 0.262). The choice may be defensible on F1 or precision, but those weren't tabulated. Selection criteria should be fixed before results are seen and reported in full: precision, recall, PR-AUC, and train–validation gaps together.
- **Small lift over baseline on accuracy** (+2.4 points). The model's value lies in the churner trade-off (65% caught, 55% of flags correct), which should be framed as a retention-cost decision.
- **Next steps:**
  - Tune the decision threshold on validation.
  - Calibrate predicted probabilities.
  - Report confidence intervals (only 407 churners in the test set).

## Files
- [`bank_churn_neural_network.ipynb`](bank_churn_neural_network.ipynb): cleaned notebook (course instructions removed; code and outputs unchanged).
