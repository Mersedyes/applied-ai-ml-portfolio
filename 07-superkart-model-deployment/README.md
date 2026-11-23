# 07 · SuperKart: Sales Forecasting Model, API, Front End, and a 2026 Re-run

**Question.** Can product and store attributes predict a product's total sales at a given store, and can that model be served reliably to business users?
**Data.** 8,763 product–store records:
- **Product attributes:** weight, sugar content, allocated shelf area, MRP (maximum retail price), and type.
- **Store attributes:** size, city tier, type, and establishment year.

Course-provided; not redistributed.
**Tools.** scikit-learn pipelines, XGBoost, Flask, gunicorn, Docker, Streamlit, Hugging Face Spaces.

## Modeling
- **Feature engineering:**
  - Store age (from establishment year).
  - Product-family code (from product ID).
  - Grouped product categories.
  - Capped extreme MRP values.
- **A single sklearn `Pipeline`:** `ColumnTransformer` (median and most-frequent imputation, one-hot encoding) plus the regressor. It is serialized with `joblib`, so the API applies exactly the same preprocessing as training.
- **Models:** linear-regression baseline, decision tree, random forest, and XGBoost, tuned with grid and randomized search.

Results from my submitted presentation (values were rounded in the source):

| Model | Test R² | Test RMSE |
|---|---|---|
| Random forest (tuned) | ≈ 0.96 | ≈ 210 |
| XGBoost (tuned) | ≈ 0.95 | ≈ 215 |

## Deployment

```
Streamlit front end (HF Space) ──HTTP──▶ Flask API (HF Space, Docker)
                                         POST /v1/predict  → {"Sales": <float>}
                                         gunicorn, 4 workers, port 7860
                                         python:3.9-slim image
                                         loads backend_files/final_model.joblib
```

- **API:** [huggingface.co/spaces/mersedyes/SuperKart-Sales-Prediction](https://huggingface.co/spaces/mersedyes/SuperKart-Sales-Prediction)
- **Front end:** [huggingface.co/spaces/mersedyes/SuperKart-Frontend](https://huggingface.co/spaces/mersedyes/SuperKart-Frontend)
- Free-tier Spaces sleep when idle; the first request takes a minute to wake them.

The request body carries 10 fields: `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, `Product_MRP`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`, `Product_Id_char`, `Store_Age_Years`, `Product_Type_Category`.

## 2026 re-audit: traceability and hardening
- **The deployed model doesn't match the documented one.** The presentation reports a tuned random forest as the final model. Hugging Face's pickle scan of the deployed `final_model.joblib` detects XGBoost classes. So the reported metrics describe a model that isn't the one serving predictions. In a regulated setting this is an audit finding. The fix is provenance by construction:
  - Write a model card with the artifact's SHA-256, training-data version, and evaluation metrics at training time.
  - Have the API expose that metadata (e.g., `GET /v1/model`).
  - Have CI fail if the deployed hash doesn't match the evaluated one.
- **Hardening backlog:**

| Issue | Fix |
|---|---|
| No input validation (a missing field raises an unhandled error → HTTP 500) | Request schema (e.g., pydantic) returning 422 with field errors |
| No automated tests | Unit tests for the payload → DataFrame mapping; contract test against a fixture model |
| Model loaded from a pickle | Load only artifacts whose hash matches the model card; never accept uploaded models |
| `python:3.9-slim` (Python 3.9 reached end of life in October 2025) | Upgrade to a supported Python; pin dependency versions; image scanning in CI |
| Container runs as root by default | Non-root `USER`; read-only filesystem where possible |
| No health or readiness endpoint | `GET /healthz` checking that the model loaded |
| Leftover comments from another project's template | Clean up during the refactor |

## 2026 re-run: traceable model selection
The 2025 modeling notebook was not recovered. [`rerun_2026/superkart_rerun_2026.ipynb`](rerun_2026/superkart_rerun_2026.ipynb) is an executed notebook that rebuilds the model behind the API's exact 10-field contract and adds the missing provenance:
- Selection by 5-fold CV on the training set (metric fixed in advance: RMSE), with the test set used once.
- A leave-one-store-out generalization check.
- A model card with the artifact's SHA-256.
- The payload validation the API lacks.

Data cleaning found in the process: `reg` and `Regular` were the same sugar-content category, and each of the 4 stores has a fixed size, tier, and type.

**Cross-validation (training set):**

| Model | CV RMSE | CV R² |
|---|---|---|
| Predict the mean (baseline) | 1,063.7 | 0.00 |
| Linear regression | 447.9 | 0.82 |
| **Random forest (selected)** | **280.8 ± 23.9** | **0.930** |
| XGBoost | 285.4 ± 22.0 | 0.928 |

**Test set (selected model, used once):** RMSE 292.3 [263.4, 321.7], MAE 110.2, R² 0.925 [0.910, 0.939]. These are lower than the presentation's rounded R² ≈ 0.96, which could not be reproduced.

What the re-run shows:
- **The model choice was never the real problem.** Random forest and XGBoost are within one standard deviation on CV error, so deploying XGBoost was defensible. XGBoost's artifact is also far smaller: 186 kB deployed vs. 27 MB for this random forest. The problem was that the choice wasn't recorded. `rerun_2026/model_card.json` now ties the evaluated artifact's SHA-256, data hash, metrics, input schema, and library versions together. A deploy step can refuse any artifact whose hash differs.
- **The model can't predict sales for a store it hasn't seen.** Holding out each store in turn, R² falls to between −0.45 and −30.7: worse than predicting the training mean. A random split hides this because every test row comes from a known store. The model is fit for the 4 existing stores only. The presentation's store-expansion recommendations ("Supermarket Type 1 expansion", "Tier 1 penetration") would need data from more stores to support them.
- **Payload validation:** a `validate_payload` function returns field-level errors (HTTP 422) instead of an unhandled exception. It is tested in the notebook against the trained pipeline.

## Presentation
[`presentation/SuperKart_Presentation.pdf`](presentation/SuperKart_Presentation.pdf) is my submitted deck (29 slides), unedited. Its final-model slides (random forest, R² ≈ 0.96) don't match the deployed artifact; see the re-audit and re-run above.

## Code
The API and front-end source live in the two Hugging Face Space repositories above. To mirror them here (run on your own machine):

```zsh
git clone https://huggingface.co/spaces/mersedyes/SuperKart-Sales-Prediction api
git clone https://huggingface.co/spaces/mersedyes/SuperKart-Frontend frontend
rm -rf api/.git frontend/.git
```

`backend_files/final_model.joblib` stays out of git; `*.joblib` is gitignored.

## Files
- [`rerun_2026/superkart_rerun_2026.ipynb`](rerun_2026/superkart_rerun_2026.ipynb): executed 2026 re-run (needs `data/SuperKart.csv` to re-execute).
- [`rerun_2026/model_card.json`](rerun_2026/model_card.json): model card with artifact hash, metrics, schema, and versions.
- [`presentation/SuperKart_Presentation.pdf`](presentation/SuperKart_Presentation.pdf): original 2025 presentation.
