# 07 · SuperKart: Sales Forecasting Model, API, and Front End

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

## Code
The API and front-end source live in the two Hugging Face Space repositories above. To mirror them here:

```bash
git clone https://huggingface.co/spaces/mersedyes/SuperKart-Sales-Prediction api
git clone https://huggingface.co/spaces/mersedyes/SuperKart-Frontend frontend
rm -rf api/.git frontend/.git
# keep backend_files/final_model.joblib out of git; it is ignored by .gitignore
```
