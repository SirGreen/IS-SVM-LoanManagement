import os
import tempfile
import unittest
from unittest import mock

from app import FEATURE_NAMES, ModelService, Preprocessor, create_app, load_german_credit_rows


SAMPLE_PAYLOAD = {
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
    "foreign_worker": "A201",
}


class CreditApiTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dataset_zip = os.path.join(root_dir, "statlog+german+credit+data.zip")
        state_file = os.path.join(self.temp_dir.name, "model_state.json")
        log_file = os.path.join(self.temp_dir.name, "predictions.jsonl")

        env = {
            "DATASET_ZIP_PATH": dataset_zip,
            "MODEL_STATE_FILE": state_file,
            "UNCERTAIN_LOG_FILE": log_file,
            "API_KEY": "test-key",
            "APP_ENV": "production",
            "DISABLE_ADMIN_AUTH": "false",
            "UNCERTAINTY_MARGIN": "0.35",
            "MODEL_SEED": "42",
            "TEST_RATIO": "0.2",
        }

        self.env_patcher = mock.patch.dict(os.environ, env, clear=False)
        self.env_patcher.start()
        self.addCleanup(self.env_patcher.stop)

        self.app = create_app()
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["service"], "credit-svm-api")

    def test_predict_endpoint_returns_result_and_explanation(self):
        response = self.client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn(body["prediction"], [0, 1])
        self.assertIn("decision_score", body)
        self.assertIsInstance(body["explanation"], list)
        self.assertGreater(len(body["explanation"]), 0)

    def test_predict_endpoint_validates_missing_fields(self):
        payload = dict(SAMPLE_PAYLOAD)
        payload.pop("age")
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertFalse(body["ok"])
        self.assertTrue(any("age" in item for item in body["errors"]))

    def test_predict_accepts_salary_alias(self):
        payload = dict(SAMPLE_PAYLOAD)
        payload.pop("credit_amount")
        payload["salary"] = 1169
        response = self.client.post("/api/v1/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])

    def test_admin_stats_requires_api_key(self):
        response = self.client.get("/api/v1/admin/stats")
        self.assertEqual(response.status_code, 401)

    def test_admin_stats_returns_metrics(self):
        response = self.client.get("/api/v1/admin/stats", headers={"X-API-Key": "test-key"})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("accuracy", body["stats"])
        self.assertIn("f1_score", body["stats"])

    def test_uncertain_cases_endpoint_lists_logged_cases(self):
        margin_response = self.client.post(
            "/api/v1/admin/uncertainty-margin",
            json={"uncertainty_margin": 100.0},
            headers={"X-API-Key": "test-key"},
        )
        self.assertEqual(margin_response.status_code, 200)

        predict_response = self.client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
        self.assertEqual(predict_response.status_code, 200)

        response = self.client.get("/api/v1/officer/uncertain?page=1&limit=20")
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertGreaterEqual(body["total"], 1)
        self.assertEqual(body["page"], 1)
        self.assertIsInstance(body["items"], list)

    def test_retrain_endpoint_rebuilds_model(self):
        response = self.client.post(
            "/api/v1/admin/retrain",
            json={},
            headers={"X-API-Key": "test-key"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertTrue(body["ok"])
        self.assertIn("accuracy", body["stats"])


class CoreModelTestCase(unittest.TestCase):
    def test_preprocessor_transform_is_consistent_shape(self):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        dataset_zip = os.path.join(root_dir, "statlog+german+credit+data.zip")
        rows, _ = load_german_credit_rows(dataset_zip)
        preprocessor = Preprocessor()
        preprocessor.fit(rows[:20])

        x1 = preprocessor.transform(rows[:3])
        x2 = preprocessor.transform(rows[3:6])

        self.assertEqual(x1.shape[1], x2.shape[1])
        self.assertEqual(x1.shape[1], len(preprocessor.vector_to_raw_feature))

    def test_model_service_retrain_returns_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            dataset_zip = os.path.join(root_dir, "statlog+german+credit+data.zip")
            service = ModelService(
                dataset_zip_path=dataset_zip,
                state_file=os.path.join(temp_dir, "state.json"),
                uncertain_log_file=os.path.join(temp_dir, "predictions.jsonl"),
                project_root=root_dir,
                test_ratio=0.2,
                seed=42,
                uncertainty_margin=0.35,
            )
            stats = service.retrain()
            self.assertIn("accuracy", stats)
            self.assertIn("f1_score", stats)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "state.json")))


if __name__ == "__main__":
    unittest.main()
