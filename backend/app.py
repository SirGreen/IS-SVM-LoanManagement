import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple
from zipfile import ZipFile

import numpy as np
from flask import Flask, jsonify, request


SWAGGER_UI_HTML = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
    <title>Credit SVM API Docs</title>
    <link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui.css\" />
    <style>
        html, body { margin: 0; padding: 0; }
        body { background: #fafafa; }
    </style>
</head>
<body>
    <div id=\"swagger-ui\"></div>
    <script src=\"https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js\"></script>
    <script>
        window.ui = SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [SwaggerUIBundle.presets.apis],
        });
    </script>
</body>
</html>
"""


FEATURE_NAMES = [
    "status_checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate_pct",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "liable_dependents",
    "telephone",
    "foreign_worker",
]

NUMERIC_FEATURES = {
    "duration_months",
    "credit_amount",
    "installment_rate_pct",
    "residence_since",
    "age",
    "existing_credits",
    "liable_dependents",
}

ORDINAL_FEATURE_MAPS = {
    "status_checking_account": {"A14": 0, "A11": 1, "A12": 2, "A13": 3},
    "credit_history": {"A30": 0, "A31": 1, "A32": 2, "A33": 3, "A34": 4},
    "savings_account": {"A65": 0, "A61": 1, "A62": 2, "A63": 3, "A64": 4},
    "employment_since": {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4},
    "job": {"A171": 0, "A172": 1, "A173": 2, "A174": 3},
}

NOMINAL_FEATURES = [
    name for name in FEATURE_NAMES if name not in NUMERIC_FEATURES and name not in ORDINAL_FEATURE_MAPS
]

FIELD_ALIASES = {
    "salary": "credit_amount",
}


@dataclass
class ModelStats:
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    train_size: int
    test_size: int


class Preprocessor:
    def __init__(self) -> None:
        self.numeric_means: Dict[str, float] = {}
        self.numeric_stds: Dict[str, float] = {}
        self.nominal_categories: Dict[str, List[str]] = {}
        self.vector_to_raw_feature: List[str] = []

    def fit(self, rows: List[Dict[str, str]]) -> None:
        for feature in NUMERIC_FEATURES:
            vals = np.array([float(r[feature]) for r in rows], dtype=np.float64)
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            self.numeric_means[feature] = mean
            self.numeric_stds[feature] = std if std > 1e-12 else 1.0

        for feature in NOMINAL_FEATURES:
            self.nominal_categories[feature] = sorted({str(r[feature]) for r in rows})

        self.vector_to_raw_feature = []
        for feature in FEATURE_NAMES:
            if feature in NUMERIC_FEATURES or feature in ORDINAL_FEATURE_MAPS:
                self.vector_to_raw_feature.append(feature)
            else:
                for _ in self.nominal_categories[feature]:
                    self.vector_to_raw_feature.append(feature)

    def _row_to_vector(self, row: Dict[str, str]) -> List[float]:
        vec: List[float] = []
        for feature in FEATURE_NAMES:
            val = row[feature]
            if feature in NUMERIC_FEATURES:
                x = float(val)
                z = (x - self.numeric_means[feature]) / self.numeric_stds[feature]
                vec.append(float(z))
            elif feature in ORDINAL_FEATURE_MAPS:
                vec.append(float(ORDINAL_FEATURE_MAPS[feature].get(str(val), 0.0)))
            else:
                cats = self.nominal_categories[feature]
                for cat in cats:
                    vec.append(1.0 if str(val) == cat else 0.0)
        return vec

    def transform(self, rows: List[Dict[str, str]]) -> np.ndarray:
        return np.array([self._row_to_vector(r) for r in rows], dtype=np.float64)

    def to_dict(self) -> Dict[str, object]:
        return {
            "numeric_means": self.numeric_means,
            "numeric_stds": self.numeric_stds,
            "nominal_categories": self.nominal_categories,
            "vector_to_raw_feature": self.vector_to_raw_feature,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, object]) -> "Preprocessor":
        obj = cls()
        obj.numeric_means = {k: float(v) for k, v in state["numeric_means"].items()}
        obj.numeric_stds = {k: float(v) for k, v in state["numeric_stds"].items()}
        obj.nominal_categories = {
            k: [str(x) for x in values] for k, values in state["nominal_categories"].items()
        }
        obj.vector_to_raw_feature = [str(x) for x in state["vector_to_raw_feature"]]
        return obj


class LinearSVM:
    def __init__(
        self,
        c: float = 1.2,
        learning_rate: float = 0.01,
        epochs: int = 120,
        lr_decay: float = 0.01,
        class_weight_positive: float = 1.0,
        class_weight_negative: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.c = c
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.lr_decay = lr_decay
        self.class_weight_positive = class_weight_positive
        self.class_weight_negative = class_weight_negative
        self.seed = seed
        self.w: np.ndarray = np.array([], dtype=np.float64)
        self.b: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for epoch in range(self.epochs):
            lr = self.learning_rate / (1.0 + self.lr_decay * epoch)
            for idx in rng.permutation(n_samples):
                xi = x[idx]
                yi = y[idx]
                margin = yi * (float(np.dot(xi, self.w)) + self.b)
                sample_weight = self.class_weight_positive if yi == 1.0 else self.class_weight_negative
                if margin >= 1.0:
                    grad_w = self.w
                    grad_b = 0.0
                else:
                    grad_w = self.w - (self.c * sample_weight * yi * xi)
                    grad_b = -self.c * sample_weight * yi

                self.w -= lr * grad_w
                self.b -= lr * grad_b

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.dot(x, self.w) + self.b

    def predict_pm1(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        return np.where(scores >= 0.0, 1.0, -1.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "c": self.c,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "lr_decay": self.lr_decay,
            "class_weight_positive": self.class_weight_positive,
            "class_weight_negative": self.class_weight_negative,
            "seed": self.seed,
            "w": self.w.tolist(),
            "b": self.b,
        }

    @classmethod
    def from_dict(cls, state: Dict[str, object]) -> "LinearSVM":
        obj = cls(
            c=float(state["c"]),
            learning_rate=float(state["learning_rate"]),
            epochs=int(state["epochs"]),
            lr_decay=float(state["lr_decay"]),
            class_weight_positive=float(state.get("class_weight_positive", 1.0)),
            class_weight_negative=float(state.get("class_weight_negative", 1.0)),
            seed=int(state["seed"]),
        )
        obj.w = np.array(state["w"], dtype=np.float64)
        obj.b = float(state["b"])
        return obj


def train_test_split(
    rows: List[Dict[str, str]], labels: np.ndarray, test_ratio: float, seed: int
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], np.ndarray, np.ndarray]:
    n = len(rows)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    test_size = int(n * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    train_rows = [rows[int(i)] for i in train_idx]
    test_rows = [rows[int(i)] for i in test_idx]
    y_train = labels[train_idx]
    y_test = labels[test_idx]
    return train_rows, test_rows, y_train, y_test


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ModelStats:
    y_true01 = (y_true == 1.0).astype(np.int64)
    y_pred01 = (y_pred == 1.0).astype(np.int64)
    tp = int(np.sum((y_true01 == 1) & (y_pred01 == 1)))
    tn = int(np.sum((y_true01 == 0) & (y_pred01 == 0)))
    fp = int(np.sum((y_true01 == 0) & (y_pred01 == 1)))
    fn = int(np.sum((y_true01 == 1) & (y_pred01 == 0)))

    accuracy = (tp + tn) / max(1, len(y_true01))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1_score = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))

    return ModelStats(
        accuracy=float(accuracy),
        f1_score=float(f1_score),
        precision=float(precision),
        recall=float(recall),
        train_size=0,
        test_size=len(y_true01),
    )


def load_german_credit_rows(zip_path: str) -> Tuple[List[Dict[str, str]], np.ndarray]:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Dataset zip not found: {zip_path}")

    target_name = None
    with ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith("german.data"):
                target_name = name
                break
        if not target_name:
            raise FileNotFoundError("german.data not found in dataset zip")

        text = zf.read(target_name).decode("utf-8", errors="ignore")

    rows: List[Dict[str, str]] = []
    labels: List[float] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 21:
            continue
        row = {FEATURE_NAMES[i]: parts[i] for i in range(20)}
        label = 1.0 if parts[20] == "1" else -1.0
        rows.append(row)
        labels.append(label)

    if not rows:
        raise ValueError("No valid rows parsed from dataset")

    return rows, np.array(labels, dtype=np.float64)


def validate_payload(payload: Dict[str, object]) -> Tuple[Dict[str, str], List[str]]:
    cleaned: Dict[str, str] = {}
    errors: List[str] = []

    mutable = dict(payload)
    for src_key, target_key in FIELD_ALIASES.items():
        if src_key in mutable and target_key not in mutable:
            mutable[target_key] = mutable[src_key]

    for feature in FEATURE_NAMES:
        if feature not in mutable:
            errors.append(f"Missing field: {feature}")
            continue
        value = mutable[feature]
        if feature in NUMERIC_FEATURES:
            try:
                float(value)
                cleaned[feature] = str(value)
            except (ValueError, TypeError):
                errors.append(f"Field must be numeric: {feature}")
        else:
            cleaned[feature] = str(value)

    return cleaned, errors


class ModelService:
    def __init__(
        self,
        dataset_zip_path: str,
        state_file: str,
        uncertain_log_file: str,
        project_root: str,
        test_ratio: float,
        seed: int,
        uncertainty_margin: float,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self.dataset_zip_path = self._resolve_project_path(dataset_zip_path)
        self.state_file = state_file
        self.uncertain_log_file = uncertain_log_file
        self.test_ratio = test_ratio
        self.seed = seed
        self.uncertainty_margin = uncertainty_margin

        self.preprocessor = Preprocessor()
        self.svm = LinearSVM(seed=seed)
        self.stats = ModelStats(0.0, 0.0, 0.0, 0.0, 0, 0)
        self.decision_threshold = 0.0

    def load_or_train(self) -> None:
        if os.path.exists(self.state_file):
            try:
                self._load_state()
                return
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                # Saved artifacts may be from an older schema; retrain and overwrite.
                pass
        self.retrain(self.dataset_zip_path)

    def retrain(self, dataset_zip_path: str = "") -> Dict[str, object]:
        if dataset_zip_path:
            self.dataset_zip_path = self._resolve_project_path(dataset_zip_path)

        rows, labels = load_german_credit_rows(self.dataset_zip_path)
        train_rows, test_rows, y_train, y_test = train_test_split(rows, labels, self.test_ratio, self.seed)

        self.preprocessor = Preprocessor()
        self.preprocessor.fit(train_rows)
        x_train = self.preprocessor.transform(train_rows)
        x_test = self.preprocessor.transform(test_rows)

        # Reserve part of training data for model/threshold selection.
        n_train = len(y_train)
        rng = np.random.default_rng(self.seed + 7)
        idx = rng.permutation(n_train)
        val_size = max(1, int(n_train * 0.2))
        val_idx = idx[:val_size]
        fit_idx = idx[val_size:]
        if len(fit_idx) == 0:
            fit_idx = val_idx

        x_fit = x_train[fit_idx]
        y_fit = y_train[fit_idx]
        x_val = x_train[val_idx]
        y_val = y_train[val_idx]

        candidate_params = [
            {"c": 0.8, "learning_rate": 0.02, "epochs": 120, "lr_decay": 0.01, "class_weight_positive": 1.0, "class_weight_negative": 1.0},
            {"c": 1.0, "learning_rate": 0.01, "epochs": 150, "lr_decay": 0.01, "class_weight_positive": 1.0, "class_weight_negative": 1.4},
            {"c": 1.2, "learning_rate": 0.01, "epochs": 180, "lr_decay": 0.008, "class_weight_positive": 1.0, "class_weight_negative": 1.8},
            {"c": 1.6, "learning_rate": 0.008, "epochs": 220, "lr_decay": 0.006, "class_weight_positive": 1.0, "class_weight_negative": 2.2},
        ]

        best_cfg = None
        best_threshold = 0.0
        best_val_acc = -1.0
        for params in candidate_params:
            candidate = LinearSVM(seed=self.seed, **params)
            candidate.fit(x_fit, y_fit)
            val_scores = candidate.decision_function(x_val)
            tuned_threshold = self._tune_threshold(val_scores, y_val)
            val_pred = np.where(val_scores >= tuned_threshold, 1.0, -1.0)
            val_metrics = compute_binary_metrics(y_val, val_pred)
            if val_metrics.accuracy > best_val_acc:
                best_val_acc = val_metrics.accuracy
                best_cfg = params
                best_threshold = tuned_threshold

        # Refit selected config on full training set, then tune decision threshold on validation subset.
        self.svm = LinearSVM(seed=self.seed, **best_cfg)
        self.svm.fit(x_train, y_train)
        val_scores = self.svm.decision_function(x_val)
        self.decision_threshold = self._tune_threshold(val_scores, y_val)
        if abs(self.decision_threshold) < 1e-12:
            self.decision_threshold = best_threshold

        test_scores = self.svm.decision_function(x_test)
        y_test_pred = np.where(test_scores >= self.decision_threshold, 1.0, -1.0)
        metrics = compute_binary_metrics(y_test, y_test_pred)
        self.stats = ModelStats(
            accuracy=metrics.accuracy,
            f1_score=metrics.f1_score,
            precision=metrics.precision,
            recall=metrics.recall,
            train_size=int(len(y_train)),
            test_size=int(len(y_test)),
        )

        self._save_state()
        return self.get_stats()

    def predict(self, payload: Dict[str, object]) -> Dict[str, object]:
        cleaned, errors = validate_payload(payload)
        if errors:
            return {"ok": False, "errors": errors}

        x = self.preprocessor.transform([cleaned])
        score = float(self.svm.decision_function(x)[0])
        prediction = 1 if score >= self.decision_threshold else 0
        uncertain = abs(score - self.decision_threshold) <= self.uncertainty_margin

        feature_contrib: Dict[str, float] = {}
        for i, weight in enumerate(self.svm.w):
            raw_feature = self.preprocessor.vector_to_raw_feature[i]
            contrib = float(x[0, i] * weight)
            feature_contrib[raw_feature] = feature_contrib.get(raw_feature, 0.0) + contrib

        ranked = sorted(feature_contrib.items(), key=lambda kv: abs(kv[1]), reverse=True)[:6]
        explanation = [
            {
                "feature": name,
                "contribution": float(value),
                "impact": "increase_risk" if value >= 0 else "decrease_risk",
            }
            for name, value in ranked
        ]

        output = {
            "ok": True,
            "prediction": prediction,
            "decision_score": score,
            "decision_threshold": self.decision_threshold,
            "uncertain": uncertain,
            "uncertainty_margin": self.uncertainty_margin,
            "explanation": explanation,
        }
        self._append_prediction_log(cleaned, output)
        return output

    def list_uncertain(self, page: int, limit: int) -> Dict[str, object]:
        if not os.path.exists(self.uncertain_log_file):
            return {"items": [], "page": page, "limit": limit, "total": 0}

        items = []
        with open(self.uncertain_log_file, "r", encoding="utf-8") as file_obj:
            for line in file_obj:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if (
                    abs(float(record.get("decision_score", 0.0)) - self.decision_threshold)
                    <= self.uncertainty_margin
                ):
                    items.append(record)

        items.sort(key=lambda row: row.get("timestamp", 0), reverse=True)
        total = len(items)
        start = max(0, (page - 1) * limit)
        end = start + limit
        return {
            "items": items[start:end],
            "page": page,
            "limit": limit,
            "total": total,
            "uncertainty_margin": self.uncertainty_margin,
        }

    def set_uncertainty_margin(self, margin: float) -> Dict[str, object]:
        self.uncertainty_margin = float(margin)
        self._save_state()
        return {"ok": True, "uncertainty_margin": self.uncertainty_margin}

    def get_stats(self) -> Dict[str, object]:
        return {
            "accuracy": self.stats.accuracy,
            "f1_score": self.stats.f1_score,
            "precision": self.stats.precision,
            "recall": self.stats.recall,
            "train_size": self.stats.train_size,
            "test_size": self.stats.test_size,
            "uncertainty_margin": self.uncertainty_margin,
            "decision_threshold": self.decision_threshold,
            "dataset_zip_path": self._to_project_relative(self.dataset_zip_path),
        }

    def _append_prediction_log(self, payload: Dict[str, str], result: Dict[str, object]) -> None:
        os.makedirs(os.path.dirname(self.uncertain_log_file), exist_ok=True)
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": int(time.time()),
            "input": payload,
            "prediction": result["prediction"],
            "decision_score": result["decision_score"],
            "uncertain": result["uncertain"],
        }
        with open(self.uncertain_log_file, "a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(record) + "\n")

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        payload = {
            "dataset_zip_path": self._to_project_relative(self.dataset_zip_path),
            "test_ratio": self.test_ratio,
            "seed": self.seed,
            "uncertainty_margin": self.uncertainty_margin,
            "decision_threshold": self.decision_threshold,
            "preprocessor": self.preprocessor.to_dict(),
            "svm": self.svm.to_dict(),
            "stats": {
                "accuracy": self.stats.accuracy,
                "f1_score": self.stats.f1_score,
                "precision": self.stats.precision,
                "recall": self.stats.recall,
                "train_size": self.stats.train_size,
                "test_size": self.stats.test_size,
            },
            "saved_at": int(time.time()),
        }
        with open(self.state_file, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2)

    def _load_state(self) -> None:
        with open(self.state_file, "r", encoding="utf-8") as file_obj:
            payload = json.load(file_obj)

        self.dataset_zip_path = self._resolve_project_path(
            payload.get("dataset_zip_path", self.dataset_zip_path)
        )
        self.test_ratio = float(payload.get("test_ratio", self.test_ratio))
        self.seed = int(payload.get("seed", self.seed))
        self.uncertainty_margin = float(payload.get("uncertainty_margin", self.uncertainty_margin))
        self.decision_threshold = float(payload.get("decision_threshold", 0.0))
        self.preprocessor = Preprocessor.from_dict(payload["preprocessor"])
        self.svm = LinearSVM.from_dict(payload["svm"])

        stats = payload["stats"]
        self.stats = ModelStats(
            accuracy=float(stats["accuracy"]),
            f1_score=float(stats["f1_score"]),
            precision=float(stats["precision"]),
            recall=float(stats["recall"]),
            train_size=int(stats["train_size"]),
            test_size=int(stats["test_size"]),
        )

    def _resolve_project_path(self, path_value: str) -> str:
        candidate = str(path_value).strip()
        if not candidate:
            return os.path.join(self.project_root, "statlog+german+credit+data.zip")
        if os.path.isabs(candidate):
            return os.path.normpath(candidate)
        return os.path.normpath(os.path.join(self.project_root, candidate))

    def _to_project_relative(self, path_value: str) -> str:
        abs_path = os.path.abspath(path_value)
        rel_path = os.path.relpath(abs_path, self.project_root)
        # Keep API output consistent across OSes for clients.
        return rel_path.replace("\\", "/")

    def _tune_threshold(self, scores: np.ndarray, y_true: np.ndarray) -> float:
        if scores.size == 0:
            return 0.0

        # Use quantiles to keep threshold search fast and robust.
        quantiles = np.linspace(0.05, 0.95, 31)
        candidates = np.unique(np.quantile(scores, quantiles))
        candidates = np.concatenate(([float(np.min(scores)) - 1e-6], candidates, [float(np.max(scores)) + 1e-6]))

        best_threshold = 0.0
        best_accuracy = -1.0
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = np.where(scores >= threshold, 1.0, -1.0)
            metrics = compute_binary_metrics(y_true, y_pred)
            if metrics.accuracy > best_accuracy or (
                abs(metrics.accuracy - best_accuracy) < 1e-12 and metrics.f1_score > best_f1
            ):
                best_accuracy = metrics.accuracy
                best_f1 = metrics.f1_score
                best_threshold = float(threshold)

        return best_threshold


def create_app() -> Flask:
    app = Flask(__name__)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    def resolve_project_path(path_value: str) -> str:
        candidate = str(path_value).strip()
        if os.path.isabs(candidate):
            return os.path.normpath(candidate)
        return os.path.normpath(os.path.join(root_dir, candidate))

    dataset_zip_path = resolve_project_path(
        os.environ.get("DATASET_ZIP_PATH", "statlog+german+credit+data.zip")
    )
    state_file = resolve_project_path(
        os.environ.get("MODEL_STATE_FILE", "backend/artifacts/model_state.json")
    )
    uncertain_log_file = resolve_project_path(
        os.environ.get("UNCERTAIN_LOG_FILE", "backend/artifacts/predictions.jsonl")
    )
    api_key = os.environ.get("API_KEY", "change-this-key")
    app_env = os.environ.get("APP_ENV", "").strip().lower()
    disable_admin_auth_raw = os.environ.get("DISABLE_ADMIN_AUTH", "").strip().lower()
    if disable_admin_auth_raw:
        disable_admin_auth = disable_admin_auth_raw in {"1", "true", "yes", "y", "on"}
    else:
        disable_admin_auth = app_env in {"dev", "development"}

    service = ModelService(
        dataset_zip_path=dataset_zip_path,
        state_file=state_file,
        uncertain_log_file=uncertain_log_file,
        project_root=root_dir,
        test_ratio=float(os.environ.get("TEST_RATIO", "0.2")),
        seed=int(os.environ.get("MODEL_SEED", "42")),
        uncertainty_margin=float(os.environ.get("UNCERTAINTY_MARGIN", "0.35")),
    )
    service.load_or_train()

    def require_json() -> Tuple[bool, object]:
        if not request.is_json:
            return False, (jsonify({"ok": False, "error": "Content-Type must be application/json"}), 415)
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return False, (jsonify({"ok": False, "error": "Invalid JSON object"}), 400)
        return True, payload

    def check_admin_key() -> Tuple[bool, object]:
        if disable_admin_auth:
            return True, None
        if request.headers.get("X-API-Key", "") != api_key:
            return False, (jsonify({"ok": False, "error": "Unauthorized"}), 401)
        return True, None

    @app.get("/openapi.json")
    def openapi_spec() -> object:
        sample_predict = {
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
        return jsonify(
            {
                "openapi": "3.0.3",
                "info": {
                    "title": "Credit SVM API",
                    "version": "1.0.0",
                    "description": "Custom from-scratch SVM credit scoring API",
                },
                "servers": [{"url": request.host_url.rstrip("/")}],
                "components": {
                    "securitySchemes": {
                        "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
                    },
                    "schemas": {
                        "PredictRequest": {
                            "type": "object",
                            "required": FEATURE_NAMES,
                            "properties": {
                                name: {"type": "string" if name not in NUMERIC_FEATURES else "number"}
                                for name in FEATURE_NAMES
                            },
                            "example": sample_predict,
                        }
                    },
                },
                "paths": {
                    "/api/v1/health": {
                        "get": {
                            "summary": "Health check",
                            "responses": {"200": {"description": "OK"}},
                        }
                    },
                    "/api/v1/predict": {
                        "post": {
                            "summary": "Predict credit outcome",
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {"$ref": "#/components/schemas/PredictRequest"}
                                    }
                                },
                            },
                            "responses": {"200": {"description": "Prediction result"}, "400": {"description": "Validation error"}},
                        }
                    },
                    "/api/v1/officer/uncertain": {
                        "get": {
                            "summary": "List uncertain cases",
                            "parameters": [
                                {"name": "page", "in": "query", "schema": {"type": "integer", "default": 1}},
                                {"name": "limit", "in": "query", "schema": {"type": "integer", "default": 20}},
                            ],
                            "responses": {"200": {"description": "Uncertain cases"}},
                        }
                    },
                    "/api/v1/admin/stats": {
                        "get": {
                            "summary": "Get model metrics",
                            "security": [{"ApiKeyAuth": []}],
                            "responses": {"200": {"description": "Model stats"}, "401": {"description": "Unauthorized"}},
                        }
                    },
                    "/api/v1/admin/retrain": {
                        "post": {
                            "summary": "Retrain model",
                            "security": [{"ApiKeyAuth": []}],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {"dataset_zip_path": {"type": "string"}},
                                            "example": {},
                                        }
                                    }
                                },
                            },
                            "responses": {"200": {"description": "Retrain result"}, "401": {"description": "Unauthorized"}},
                        }
                    },
                    "/api/v1/admin/uncertainty-margin": {
                        "get": {
                            "summary": "Get uncertainty margin",
                            "security": [{"ApiKeyAuth": []}],
                            "responses": {"200": {"description": "Current margin"}, "401": {"description": "Unauthorized"}},
                        },
                        "post": {
                            "summary": "Set uncertainty margin",
                            "security": [{"ApiKeyAuth": []}],
                            "requestBody": {
                                "required": True,
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "required": ["uncertainty_margin"],
                                            "properties": {"uncertainty_margin": {"type": "number", "minimum": 0}},
                                            "example": {"uncertainty_margin": 0.5},
                                        }
                                    }
                                },
                            },
                            "responses": {"200": {"description": "Updated margin"}, "401": {"description": "Unauthorized"}},
                        },
                    },
                },
            }
        )

    @app.get("/docs")
    def docs_page() -> object:
        return SWAGGER_UI_HTML, 200, {"Content-Type": "text/html; charset=utf-8"}

    @app.get("/api/v1/health")
    def health() -> object:
        return jsonify({"ok": True, "service": "credit-svm-api"})

    @app.post("/api/v1/predict")
    def predict_route() -> object:
        ok, payload_or_error = require_json()
        if not ok:
            return payload_or_error

        result = service.predict(payload_or_error)
        if not result.get("ok"):
            return jsonify(result), 400
        return jsonify(result)

    @app.get("/api/v1/officer/uncertain")
    def uncertain_route() -> object:
        page = max(1, int(request.args.get("page", "1")))
        limit = min(200, max(1, int(request.args.get("limit", "20"))))
        return jsonify(service.list_uncertain(page=page, limit=limit))

    @app.get("/api/v1/admin/stats")
    def stats_route() -> object:
        ok, error = check_admin_key()
        if not ok:
            return error
        return jsonify({"ok": True, "stats": service.get_stats()})

    @app.post("/api/v1/admin/retrain")
    def retrain_route() -> object:
        ok_auth, auth_error = check_admin_key()
        if not ok_auth:
            return auth_error

        ok_json, payload_or_error = require_json()
        if not ok_json:
            return payload_or_error

        payload = payload_or_error
        dataset_path = str(payload.get("dataset_zip_path", "")).strip()
        if dataset_path and not os.path.exists(service._resolve_project_path(dataset_path)):
            return jsonify({"ok": False, "error": "dataset_zip_path does not exist"}), 400

        return jsonify({"ok": True, "stats": service.retrain(dataset_path)})

    @app.get("/api/v1/admin/uncertainty-margin")
    def get_margin_route() -> object:
        ok, error = check_admin_key()
        if not ok:
            return error
        return jsonify({"ok": True, "uncertainty_margin": service.uncertainty_margin})

    @app.post("/api/v1/admin/uncertainty-margin")
    def set_margin_route() -> object:
        ok_auth, auth_error = check_admin_key()
        if not ok_auth:
            return auth_error

        ok_json, payload_or_error = require_json()
        if not ok_json:
            return payload_or_error

        payload = payload_or_error
        if "uncertainty_margin" not in payload:
            return jsonify({"ok": False, "error": "Missing uncertainty_margin"}), 400

        try:
            margin = float(payload["uncertainty_margin"])
            if margin < 0:
                raise ValueError("negative")
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "uncertainty_margin must be non-negative numeric"}), 400

        return jsonify(service.set_uncertainty_margin(margin))

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
