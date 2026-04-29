# Credit Detection API (Custom SVM)

This backend implements a linear soft-margin SVM from scratch (no prebuilt SVM library) and exposes JSON endpoints for:

- User prediction with explanation
- Credit officer uncertain-case review
- Data scientist/admin stats, retrain, and uncertainty-margin tuning

## 1) Setup

```bash
cd backend
pip install -r requirements.txt
```

## 2) Run

```bash
set API_KEY=change-this-key
set APP_ENV=development
set DISABLE_ADMIN_AUTH=true
python app.py
```

When `DISABLE_ADMIN_AUTH=true` (or `1`, `yes`, `on`), admin endpoints do not require `X-API-Key`.
If `DISABLE_ADMIN_AUTH` is not set, then `APP_ENV=development` (or `dev`) disables admin auth.
In other environments, admin endpoints require `X-API-Key: <API_KEY>`.

Server starts at `http://localhost:8000`.

Interactive API docs are available at `http://localhost:8000/docs`.

## 2.1) Run Tests

```bash
cd backend
python -m unittest discover -s tests -p "test_*.py" -v
```

The test suite uses temporary model/log artifacts so it does not overwrite your normal runtime files.

## 3) Endpoints

### Health
- `GET /api/v1/health`

### User endpoint
- `POST /api/v1/predict`
- Content-Type: `application/json`
- Request JSON body must include all 20 German Credit features:

```json
{
  "status_checking_account": "A11",
  "duration_months": 6,
  "credit_history": "A34",
  "purpose": "A43",
  "credit_amount": 1169,
  "savings_account": "A65",
  "employment_since": "A75",
  "installment_rate_pct": 4,
  "personal_status_sex": "A93",
  "other_debtors": "A101",
  "residence_since": 4,
  "property": "A121",
  "age": 67,
  "other_installment_plans": "A143",
  "housing": "A152",
  "existing_credits": 2,
  "job": "A173",
  "liable_dependents": 1,
  "telephone": "A192",
  "foreign_worker": "A201"
}
```

Response:

```json
{
  "ok": true,
  "prediction": 0,
  "decision_score": -0.274,
  "uncertain": true,
  "uncertainty_margin": 0.35,
  "explanation": [
    {
      "feature": "credit_amount",
      "contribution": 0.41,
      "impact": "increase_risk"
    }
  ]
}
```

### Credit officer endpoint
- `GET /api/v1/officer/uncertain?page=1&limit=20`
- Returns prediction cases where `abs(decision_score) <= uncertainty_margin`

### Data scientist/admin endpoints
- `GET /api/v1/admin/stats`
- `POST /api/v1/admin/retrain` with optional JSON body: `{ "dataset_zip_path": "data/new-dataset.zip" }`
- `GET /api/v1/admin/uncertainty-margin`
- `POST /api/v1/admin/uncertainty-margin` with JSON body: `{ "uncertainty_margin": 0.4 }`

Auth behavior:

- `DISABLE_ADMIN_AUTH=true` (or `1`, `yes`, `on`): no API key required for admin routes.
- If `DISABLE_ADMIN_AUTH` is unset, `APP_ENV=development` or `APP_ENV=dev`: no API key required.
- Otherwise: requires header `X-API-Key: <API_KEY>`.

## 4) PHP Integration Notes

- Send JSON only with `Content-Type: application/json`.
- PHP should not do scaling or SVM math; API handles preprocessing and inference.
- Use `decision_score` and `explanation` for PDF reporting.

## 5) Postman

Files:

- `postman/credit-svm-api.collection.json`
- `postman/credit-svm-api.environment.json`

Import both files into Postman, select the environment, update `baseUrl` or `apiKey` if needed, then run the requests directly.
