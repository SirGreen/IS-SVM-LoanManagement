import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import numpy as np

from app import (
    FEATURE_NAMES,
    LinearSVM,
    NUMERIC_FEATURES,
    NOMINAL_FEATURES,
    Preprocessor,
    compute_binary_metrics,
    load_german_credit_rows,
    train_test_split,
)


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate evaluation graphs for the credit SVM model.")
    parser.add_argument(
        "--state-file",
        default="backend/artifacts/model_state.json",
        help="Path to model_state.json",
    )
    parser.add_argument(
        "--output-dir",
        default="backend/artifacts/plots",
        help="Directory where PNG charts will be saved",
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Optional override for dataset source (zip or german.data file)",
    )
    return parser.parse_args()


def load_plain_german_data(path: Path) -> Tuple[List[Dict[str, str]], np.ndarray]:
    rows: List[Dict[str, str]] = []
    labels: List[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as file_obj:
        for line in file_obj:
            parts = line.strip().split()
            if len(parts) != 21:
                continue
            row = {FEATURE_NAMES[i]: parts[i] for i in range(20)}
            label = 1.0 if parts[20] == "1" else -1.0
            rows.append(row)
            labels.append(label)

    if not rows:
        raise ValueError(f"No valid rows found in dataset file: {path}")

    return rows, np.array(labels, dtype=np.float64)


def resolve_dataset_path(project_root: Path, state_dataset_path: str, override_dataset_path: str) -> Path:
    if override_dataset_path:
        candidate = Path(override_dataset_path)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if candidate.exists():
            return candidate.resolve()
        raise FileNotFoundError(f"--dataset-path not found: {candidate}")

    candidates: List[Path] = []
    if state_dataset_path:
        raw = Path(state_dataset_path)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(project_root / raw)
            candidates.append(project_root / raw.name)

    candidates.extend(
        [
            project_root / "statlog+german+credit+data.zip",
            project_root / "data" / "german.data",
            project_root / "backend" / "artifacts" / "german.data",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    joined = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not locate dataset. Checked:\n{joined}")


def load_rows_and_labels(dataset_path: Path) -> Tuple[List[Dict[str, str]], np.ndarray]:
    if dataset_path.suffix.lower() == ".zip":
        return load_german_credit_rows(str(dataset_path))
    return load_plain_german_data(dataset_path)


def compute_roc_points(y_true01: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    thresholds = np.concatenate(([np.inf], np.sort(np.unique(scores))[::-1], [-np.inf]))
    tpr_values: List[float] = []
    fpr_values: List[float] = []

    positives = np.sum(y_true01 == 1)
    negatives = np.sum(y_true01 == 0)

    for threshold in thresholds:
        pred01 = (scores >= threshold).astype(np.int64)
        tp = np.sum((pred01 == 1) & (y_true01 == 1))
        fp = np.sum((pred01 == 1) & (y_true01 == 0))
        tpr = float(tp / positives) if positives > 0 else 0.0
        fpr = float(fp / negatives) if negatives > 0 else 0.0
        tpr_values.append(tpr)
        fpr_values.append(fpr)

    fpr_array = np.array(fpr_values, dtype=np.float64)
    tpr_array = np.array(tpr_values, dtype=np.float64)
    order = np.argsort(fpr_array)
    auc = float(np.trapz(tpr_array[order], fpr_array[order]))
    return fpr_array, tpr_array, auc


def save_class_distribution(output_dir: Path, labels: np.ndarray) -> Path:
    y01 = (labels == 1.0).astype(np.int64)
    values = [int(np.sum(y01 == 0)), int(np.sum(y01 == 1))]

    fig, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(["Bad Credit (0)", "Good Credit (1)"], values, color=["#ef6f6c", "#43aa8b"])
    axis.set_title("Dataset Class Distribution")
    axis.set_ylabel("Number of Samples")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 3, str(value), ha="center", va="bottom")
    fig.tight_layout()

    output_path = output_dir / "class_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_metrics_chart(output_dir: Path, metrics: Dict[str, float]) -> Path:
    metric_names = ["accuracy", "precision", "recall", "f1_score"]
    values = [float(metrics[name]) for name in metric_names]

    fig, axis = plt.subplots(figsize=(7, 4))
    bars = axis.bar(metric_names, values, color=["#577590", "#4d908e", "#90be6d", "#f9c74f"])
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Model Metrics on Test Split")
    axis.set_ylabel("Score")
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()

    output_path = output_dir / "metrics_bar_chart.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_confusion_matrix(output_dir: Path, tp: int, tn: int, fp: int, fn: int) -> Path:
    matrix = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    fig, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="YlGnBu")
    fig.colorbar(image, ax=axis)

    axis.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
    axis.set_yticks([0, 1], labels=["True 0", "True 1"])
    axis.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            axis.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_score_distribution(output_dir: Path, scores: np.ndarray, y_true: np.ndarray, threshold: float) -> Path:
    y01 = (y_true == 1.0).astype(np.int64)
    neg_scores = scores[y01 == 0]
    pos_scores = scores[y01 == 1]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(neg_scores, bins=30, alpha=0.65, label="True class 0", color="#f94144")
    axis.hist(pos_scores, bins=30, alpha=0.6, label="True class 1", color="#277da1")
    axis.axvline(threshold, color="#222222", linestyle="--", linewidth=1.6, label=f"Threshold {threshold:.3f}")
    axis.set_title("Decision Score Distribution")
    axis.set_xlabel("Decision score")
    axis.set_ylabel("Count")
    axis.legend()
    fig.tight_layout()

    output_path = output_dir / "decision_score_distribution.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_roc_curve(output_dir: Path, y_true: np.ndarray, scores: np.ndarray) -> Path:
    y01 = (y_true == 1.0).astype(np.int64)
    fpr, tpr, auc = compute_roc_points(y01, scores)

    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot(fpr, tpr, color="#1d3557", linewidth=2, label=f"ROC AUC = {auc:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6c757d", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")
    fig.tight_layout()

    output_path = output_dir / "roc_curve.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_missing_value_chart(output_dir: Path, rows: List[Dict[str, str]]) -> Path:
    counts: List[int] = []
    for feature in FEATURE_NAMES:
        missing = 0
        for row in rows:
            value = str(row.get(feature, "")).strip()
            if value == "" or value.lower() == "nan":
                missing += 1
        counts.append(missing)

    fig, axis = plt.subplots(figsize=(11, 4.5))
    axis.bar(np.arange(len(FEATURE_NAMES)), counts, color="#adb5bd")
    axis.set_title("Missing Value Count per Feature")
    axis.set_ylabel("Missing count")
    axis.set_xlabel("Feature index")
    axis.set_xticks(np.arange(len(FEATURE_NAMES)), labels=[str(i + 1) for i in range(len(FEATURE_NAMES))])
    axis.text(0.01, 0.95, f"Total missing values: {sum(counts)}", transform=axis.transAxes, va="top")
    fig.tight_layout()

    output_path = output_dir / "pre_missing_values.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def get_numeric_matrix(rows: List[Dict[str, str]]) -> np.ndarray:
    ordered_numeric = [name for name in FEATURE_NAMES if name in NUMERIC_FEATURES]
    matrix = np.array(
        [[float(row[name]) for name in ordered_numeric] for row in rows],
        dtype=np.float64,
    )
    return matrix


def save_numeric_distributions(output_dir: Path, rows: List[Dict[str, str]]) -> Path:
    ordered_numeric = [name for name in FEATURE_NAMES if name in NUMERIC_FEATURES]
    matrix = get_numeric_matrix(rows)

    cols = 3
    rows_plot = int(np.ceil(len(ordered_numeric) / cols))
    fig, axes = plt.subplots(rows_plot, cols, figsize=(14, 3.7 * rows_plot))
    axes_flat = np.array(axes).reshape(-1)

    for idx, feature in enumerate(ordered_numeric):
        axis = axes_flat[idx]
        axis.hist(matrix[:, idx], bins=25, color="#3a86ff", alpha=0.8)
        axis.set_title(feature)
        axis.set_xlabel("Raw value")
        axis.set_ylabel("Count")

    for idx in range(len(ordered_numeric), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Numeric Feature Distributions (Raw)", y=0.995)
    fig.tight_layout()

    output_path = output_dir / "pre_numeric_distributions.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_numeric_correlation_heatmap(output_dir: Path, rows: List[Dict[str, str]]) -> Path:
    ordered_numeric = [name for name in FEATURE_NAMES if name in NUMERIC_FEATURES]
    matrix = get_numeric_matrix(rows)
    corr = np.corrcoef(matrix, rowvar=False)

    fig, axis = plt.subplots(figsize=(8, 6.5))
    image = axis.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(np.arange(len(ordered_numeric)), labels=ordered_numeric, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(ordered_numeric)), labels=ordered_numeric)
    axis.set_title("Numeric Feature Correlation Heatmap")
    fig.tight_layout()

    output_path = output_dir / "pre_numeric_correlation_heatmap.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_standardization_effect(output_dir: Path, rows: List[Dict[str, str]], preprocessor: Preprocessor) -> Path:
    ordered_numeric = [name for name in FEATURE_NAMES if name in NUMERIC_FEATURES]
    raw_matrix = get_numeric_matrix(rows)

    std_means: List[float] = []
    std_stds: List[float] = []
    for feature in ordered_numeric:
        raw = np.array([float(row[feature]) for row in rows], dtype=np.float64)
        standardized = (raw - preprocessor.numeric_means[feature]) / preprocessor.numeric_stds[feature]
        std_means.append(float(np.mean(standardized)))
        std_stds.append(float(np.std(standardized)))

    raw_means = np.mean(raw_matrix, axis=0)
    raw_stds = np.std(raw_matrix, axis=0)
    x = np.arange(len(ordered_numeric))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].bar(x - width / 2, raw_means, width, label="Raw mean", color="#ffb703")
    axes[0].bar(x + width / 2, std_means, width, label="Standardized mean", color="#219ebc")
    axes[0].set_ylabel("Mean")
    axes[0].set_title("Preprocessing Effect: Mean shift after standardization")
    axes[0].legend()

    axes[1].bar(x - width / 2, raw_stds, width, label="Raw std", color="#fb8500")
    axes[1].bar(x + width / 2, std_stds, width, label="Standardized std", color="#023047")
    axes[1].set_ylabel("Std")
    axes[1].set_title("Preprocessing Effect: Std normalization")
    axes[1].set_xticks(x, labels=ordered_numeric, rotation=30, ha="right")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / "pre_standardization_effect.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_categorical_risk_rates(output_dir: Path, rows: List[Dict[str, str]], labels: np.ndarray) -> Path:
    y01 = (labels == 1.0).astype(np.int64)
    rows_count = len(rows)
    summaries: List[Tuple[str, float]] = []

    for feature in NOMINAL_FEATURES:
        categories = sorted({str(row[feature]) for row in rows})
        weighted_deviation = 0.0
        for category in categories:
            indices = [i for i, row in enumerate(rows) if str(row[feature]) == category]
            if not indices:
                continue
            rate = float(np.mean(y01[indices]))
            weighted_deviation += abs(rate - float(np.mean(y01))) * (len(indices) / rows_count)
        summaries.append((feature, weighted_deviation))

    summaries.sort(key=lambda item: item[1], reverse=True)
    top = summaries[:8]

    fig, axis = plt.subplots(figsize=(10, 5.5))
    names = [item[0] for item in top][::-1]
    values = [item[1] for item in top][::-1]
    axis.barh(names, values, color="#588157")
    axis.set_xlabel("Weighted class-rate deviation")
    axis.set_title("Categorical Features with Highest Class-Rate Separation")
    fig.tight_layout()

    output_path = output_dir / "pre_categorical_risk_separation.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_feature_impact(output_dir: Path, rows: List[Dict[str, str]], preprocessor: Preprocessor, svm: LinearSVM) -> Path:
    x_all = preprocessor.transform(rows)
    contrib_matrix = np.abs(x_all * svm.w.reshape(1, -1))

    grouped: Dict[str, float] = {}
    for idx, feature_name in enumerate(preprocessor.vector_to_raw_feature):
        grouped[feature_name] = grouped.get(feature_name, 0.0) + float(np.mean(contrib_matrix[:, idx]))

    ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:12]
    names = [item[0] for item in ranked][::-1]
    values = [item[1] for item in ranked][::-1]

    fig, axis = plt.subplots(figsize=(10, 6))
    axis.barh(names, values, color="#9b5de5")
    axis.set_xlabel("Mean absolute contribution |x*w|")
    axis.set_title("Top Attribute Impact after Preprocessing")
    fig.tight_layout()

    output_path = output_dir / "pre_attribute_impact.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent

    state_path = Path(args.state_file)
    if not state_path.is_absolute():
        state_path = project_root / state_path
    state_path = state_path.resolve()

    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    with state_path.open("r", encoding="utf-8") as file_obj:
        state = json.load(file_obj)

    preprocessor = Preprocessor.from_dict(state["preprocessor"])
    svm = LinearSVM.from_dict(state["svm"])
    seed = int(state.get("seed", 42))
    test_ratio = float(state.get("test_ratio", 0.2))
    threshold = float(state.get("decision_threshold", 0.0))

    dataset_path = resolve_dataset_path(
        project_root=project_root,
        state_dataset_path=str(state.get("dataset_zip_path", "")),
        override_dataset_path=args.dataset_path,
    )

    rows, labels = load_rows_and_labels(dataset_path)
    _, test_rows, _, y_test = train_test_split(rows, labels, test_ratio=test_ratio, seed=seed)

    x_test = preprocessor.transform(test_rows)
    scores = svm.decision_function(x_test)
    y_pred = np.where(scores >= threshold, 1.0, -1.0)

    metrics = compute_binary_metrics(y_test, y_pred)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [
        save_missing_value_chart(output_dir, rows),
        save_numeric_distributions(output_dir, rows),
        save_numeric_correlation_heatmap(output_dir, rows),
        save_standardization_effect(output_dir, rows, preprocessor),
        save_categorical_risk_rates(output_dir, rows, labels),
        save_feature_impact(output_dir, rows, preprocessor, svm),
        save_class_distribution(output_dir, labels),
        save_metrics_chart(
            output_dir,
            {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
            },
        ),
        save_confusion_matrix(output_dir, tp=metrics.tp, tn=metrics.tn, fp=metrics.fp, fn=metrics.fn),
        save_score_distribution(output_dir, scores, y_test, threshold),
        save_roc_curve(output_dir, y_test, scores),
    ]

    print("Generated evaluation charts:")
    for file_path in generated_files:
        print(f"- {file_path}")


if __name__ == "__main__":
    main()
