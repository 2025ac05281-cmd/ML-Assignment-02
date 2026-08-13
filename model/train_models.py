"""Train and evaluate all classifiers required for Machine Learning Assignment 2.

The project uses the UCI Breast Cancer Wisconsin (Diagnostic) dataset available
through scikit-learn. The supplied ``test_data.csv`` contains 114 labelled rows.
Those exact rows are held out from the 569-row source dataset, and all remaining
455 rows are used for model training. This keeps the supplied test data fully
unseen during fitting and makes the reported results reproducible.

Target convention used throughout the project:
    0 -> Benign
    1 -> Malignant (positive class)
"""

from __future__ import annotations

import argparse
import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COLUMN = "diagnosis"
CLASS_LABELS = {0: "Benign", 1: "Malignant"}
ROUND_DECIMALS_FOR_MATCHING = 8


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved project locations used by training and artifact generation."""

    root: Path
    test_csv: Path
    model_dir: Path
    results_dir: Path
    confusion_dir: Path
    predictions_dir: Path

    @classmethod
    def from_root(cls, root: Path) -> "ProjectPaths":
        root = root.resolve()
        return cls(
            root=root,
            test_csv=root / "test_data.csv",
            model_dir=root / "model",
            results_dir=root / "results",
            confusion_dir=root / "results" / "confusion_matrices",
            predictions_dir=root / "results" / "predictions",
        )

    def create_directories(self) -> None:
        for directory in (
            self.model_dir,
            self.results_dir,
            self.confusion_dir,
            self.predictions_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _slugify(model_name: str) -> str:
    """Convert a display name into a stable file name."""

    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.strip().lower())
    return cleaned.strip("_")


def load_source_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the public dataset and convert the target to 1=malignant."""

    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data.copy()
    feature_names = list(features.columns)

    # scikit-learn stores 0=malignant and 1=benign. The assignment project uses
    # malignant as the positive class so precision/recall have a clear meaning.
    diagnosis = (1 - dataset.target.astype(int)).rename(TARGET_COLUMN)
    return features, diagnosis, feature_names


def load_and_validate_test_data(
    test_csv: Path, feature_names: list[str]
) -> pd.DataFrame:
    """Read the supplied test file and enforce the expected schema."""

    if not test_csv.exists():
        raise FileNotFoundError(f"Test data not found: {test_csv}")

    test_frame = pd.read_csv(test_csv)
    required_columns = feature_names + [TARGET_COLUMN]
    missing = [column for column in required_columns if column not in test_frame.columns]
    if missing:
        raise ValueError(
            "The supplied test data is missing required columns: " + ", ".join(missing)
        )

    extra = [column for column in test_frame.columns if column not in required_columns]
    if extra:
        raise ValueError(
            "The supplied test data contains unexpected columns: " + ", ".join(extra)
        )

    test_frame = test_frame[required_columns].copy()
    for column in feature_names:
        test_frame[column] = pd.to_numeric(test_frame[column], errors="raise")

    test_frame[TARGET_COLUMN] = pd.to_numeric(
        test_frame[TARGET_COLUMN], errors="raise"
    ).astype(int)

    if test_frame.empty:
        raise ValueError("The supplied test data is empty.")
    if test_frame[feature_names].isna().any().any():
        raise ValueError("The supplied test data contains missing feature values.")
    if test_frame[TARGET_COLUMN].isna().any():
        raise ValueError("The supplied test data contains missing target values.")

    invalid_targets = sorted(set(test_frame[TARGET_COLUMN]) - set(CLASS_LABELS))
    if invalid_targets:
        raise ValueError(
            "The diagnosis column must contain only 0 and 1. Invalid values: "
            + str(invalid_targets)
        )

    if test_frame.duplicated(subset=feature_names).any():
        raise ValueError("The supplied test data contains duplicate feature rows.")

    return test_frame


def derive_training_partition(
    full_features: pd.DataFrame,
    full_target: pd.Series,
    test_frame: pd.DataFrame,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series, list[int]]:
    """Remove the supplied test rows from the source data and return the complement.

    Matching is performed on all 30 features rounded to eight decimal places. The
    rounding only protects against CSV text formatting differences; the model is
    trained and evaluated using the original unrounded numeric values.
    """

    source_match = full_features[feature_names].round(
        ROUND_DECIMALS_FOR_MATCHING
    ).reset_index(names="source_index")
    supplied_match = test_frame[feature_names].round(
        ROUND_DECIMALS_FOR_MATCHING
    ).reset_index(names="test_row")

    matched = supplied_match.merge(
        source_match,
        on=feature_names,
        how="left",
        validate="one_to_one",
    )

    if matched["source_index"].isna().any():
        missing_rows = matched.loc[matched["source_index"].isna(), "test_row"].tolist()
        raise ValueError(
            "Some supplied test rows could not be found in the source dataset. "
            f"Unmatched row numbers: {missing_rows[:10]}"
        )

    source_indices = matched["source_index"].astype(int).tolist()
    expected_test_target = full_target.loc[source_indices].reset_index(drop=True)
    provided_test_target = test_frame[TARGET_COLUMN].reset_index(drop=True)
    if not expected_test_target.equals(provided_test_target):
        mismatch_rows = np.flatnonzero(
            expected_test_target.to_numpy() != provided_test_target.to_numpy()
        ).tolist()
        raise ValueError(
            "The diagnosis labels in test_data.csv do not match the source dataset. "
            f"Mismatched rows: {mismatch_rows[:10]}"
        )

    training_mask = ~full_features.index.isin(source_indices)
    train_features = full_features.loc[training_mask, feature_names].copy()
    train_target = full_target.loc[training_mask].copy()

    if len(train_features) + len(test_frame) != len(full_features):
        raise RuntimeError("Training/test partition does not cover the source dataset.")

    return train_features, train_target, source_indices


def build_models() -> dict[str, Any]:
    """Create the five classifiers explicitly listed in the assignment."""

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
        ),
        "kNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    KNeighborsClassifier(
                        n_neighbors=7,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "Naive Bayes": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", GaussianNB()),
            ]
        ),
        "Naive Bayes (Multinomial)": Pipeline(
            steps=[
                ("scaler", MinMaxScaler()),
                ("classifier", MultinomialNB()),
            ]
        ),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_probability: np.ndarray
) -> dict[str, float]:
    """Calculate every metric required in Step 2 of the assignment."""

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_probability)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    output_path: Path,
) -> None:
    """Write one labelled confusion-matrix figure per model."""

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
    )
    figure, axis = plt.subplots(figsize=(6.2, 5.0))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"{model_name} - Confusion Matrix", pad=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_metric_comparison_chart(
    metrics_frame: pd.DataFrame, output_path: Path
) -> None:
    """Create a readable grouped comparison chart for the report and notebook."""

    metric_columns = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    plot_frame = metrics_frame.set_index("ML Model Name")[metric_columns]
    axis = plot_frame.plot(kind="bar", figsize=(12.5, 6.5), width=0.82)
    axis.set_title("Classification Model Performance on Supplied Test Data", pad=16)
    axis.set_xlabel("ML Model")
    axis.set_ylabel("Score")
    axis.set_ylim(0.70, 1.02)
    axis.tick_params(axis="x", rotation=20)
    axis.legend(loc="lower left", ncol=3, frameon=True)
    axis.grid(axis="y", alpha=0.25)
    figure = axis.get_figure()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve_chart(
    roc_payload: dict[str, tuple[np.ndarray, np.ndarray, float]], output_path: Path
) -> None:
    """Create a combined ROC curve chart for all models."""

    figure, axis = plt.subplots(figsize=(8.2, 6.4))
    for model_name, (false_positive_rate, true_positive_rate, auc_value) in roc_payload.items():
        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random baseline")
    axis.set_title("ROC Curves on Supplied Test Data", pad=14)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8.5)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def train_and_evaluate(paths: ProjectPaths) -> pd.DataFrame:
    """Run the complete assignment experiment and save all reproducible artifacts."""

    paths.create_directories()
    full_features, full_target, feature_names = load_source_dataset()
    test_frame = load_and_validate_test_data(paths.test_csv, feature_names)
    train_features, train_target, test_source_indices = derive_training_partition(
        full_features,
        full_target,
        test_frame,
        feature_names,
    )

    test_features = test_frame[feature_names]
    test_target = test_frame[TARGET_COLUMN]

    models = build_models()
    metrics_rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    confusion_payload: dict[str, list[list[int]]] = {}
    roc_payload: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    artifact_map: dict[str, str] = {}
    combined_predictions = test_frame.copy()

    for model_name, model in models.items():
        model.fit(train_features, train_target)
        predictions = model.predict(test_features).astype(int)
        probabilities = model.predict_proba(test_features)[:, 1]
        model_metrics = compute_metrics(test_target, predictions, probabilities)
        metrics_rows.append({"ML Model Name": model_name, **model_metrics})

        report_dict = classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            output_dict=True,
            zero_division=0,
        )
        report_text = classification_report(
            test_target,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            digits=4,
            zero_division=0,
        )
        reports[model_name] = {"structured": report_dict, "text": report_text}

        matrix = confusion_matrix(test_target, predictions, labels=[0, 1])
        confusion_payload[model_name] = matrix.astype(int).tolist()

        false_positive_rate, true_positive_rate, _ = roc_curve(
            test_target, probabilities
        )
        roc_payload[model_name] = (
            false_positive_rate,
            true_positive_rate,
            model_metrics["AUC"],
        )

        slug = _slugify(model_name)
        artifact_name = f"{slug}.joblib"
        joblib.dump(model, paths.model_dir / artifact_name, compress=3)
        artifact_map[model_name] = artifact_name

        model_predictions = test_frame.copy()
        model_predictions["predicted_diagnosis"] = predictions
        model_predictions["predicted_label"] = pd.Series(predictions).map(CLASS_LABELS)
        model_predictions["malignant_probability"] = probabilities
        model_predictions["correct_prediction"] = predictions == test_target.to_numpy()
        model_predictions.to_csv(
            paths.predictions_dir / f"{slug}_predictions.csv", index=False
        )

        combined_predictions[f"{slug}_prediction"] = predictions
        combined_predictions[f"{slug}_malignant_probability"] = probabilities

        save_confusion_matrix(
            test_target,
            predictions,
            model_name,
            paths.confusion_dir / f"{slug}_confusion_matrix.png",
        )

    metrics_frame = pd.DataFrame(metrics_rows)
    metric_order = [
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
    metrics_frame = metrics_frame[metric_order]
    metrics_frame.to_csv(paths.results_dir / "model_comparison.csv", index=False)
    combined_predictions.to_csv(
        paths.results_dir / "all_model_predictions.csv", index=False
    )

    winner_row = metrics_frame.sort_values(
        by=["MCC", "F1", "AUC", "Accuracy"],
        ascending=False,
    ).iloc[0]
    winner = str(winner_row["ML Model Name"])

    metrics_json = {
        row["ML Model Name"]: {
            metric: float(row[metric])
            for metric in ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
        }
        for row in metrics_rows
    }
    with (paths.model_dir / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2)

    with (paths.model_dir / "classification_reports.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(reports, handle, indent=2)

    with (paths.model_dir / "confusion_matrices.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(confusion_payload, handle, indent=2)

    with (paths.model_dir / "feature_names.json").open("w", encoding="utf-8") as handle:
        json.dump(feature_names, handle, indent=2)

    metadata = {
        "project_title": "Breast Cancer Diagnostic Classification",
        "dataset_name": "Breast Cancer Wisconsin (Diagnostic)",
        "dataset_source": "UCI Machine Learning Repository via scikit-learn",
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "class_labels": {str(key): value for key, value in CLASS_LABELS.items()},
        "source_instances": int(len(full_features)),
        "feature_count": int(len(feature_names)),
        "training_instances": int(len(train_features)),
        "test_instances": int(len(test_frame)),
        "test_source_indices": test_source_indices,
        "random_state": RANDOM_STATE,
        "winner": winner,
        "winner_selection_rule": "Highest MCC, then F1, AUC, and Accuracy as tie-breakers",
        "model_artifacts": artifact_map,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    with (paths.model_dir / "training_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)

    report_lines: list[str] = []
    for model_name in models:
        report_lines.append("=" * 78)
        report_lines.append(model_name)
        report_lines.append("=" * 78)
        report_lines.append(reports[model_name]["text"])
        report_lines.append("")
    (paths.results_dir / "classification_reports.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    save_metric_comparison_chart(
        metrics_frame,
        paths.results_dir / "model_metric_comparison.png",
    )
    save_roc_curve_chart(
        roc_payload,
        paths.results_dir / "roc_curves.png",
    )

    sample_upload = test_frame[feature_names].head(12)
    sample_upload.to_csv(paths.root / "sample_upload_without_target.csv", index=False)

    return metrics_frame


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate all Machine Learning Assignment 2 classifiers."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing test_data.csv (default: inferred from script path)",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    paths = ProjectPaths.from_root(arguments.project_root)
    metrics_frame = train_and_evaluate(paths)

    print("Training and evaluation completed successfully.")
    print(f"Project root: {paths.root}")
    print(f"Training rows: 455 | Test rows: {len(pd.read_csv(paths.test_csv))}")
    print("\nModel comparison:")
    print(metrics_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
