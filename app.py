"""Interactive Streamlit application for Machine Learning Assignment 2.

The application loads five pre-trained classifiers, presents the saved model
comparison, accepts a CSV test file, validates its schema, and displays
predictions plus assignment-required evaluation metrics whenever labels are
included.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
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

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data.csv"
TARGET_COLUMN = "diagnosis"
CLASS_LABELS = {0: "Benign", 1: "Malignant"}
METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

st.set_page_config(
    page_title="ML Assignment 2 - Breast Cancer Classification",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    :root {
        --navy: #132238;
        --blue: #2f6fed;
        --soft-blue: #eef4ff;
        --teal: #168a88;
        --ink: #1d2733;
        --muted: #555f6d;
        --line: #dfe5ec;
        --surface: #ffffff;
    }

    /* ── Global text reset handled by config.toml theme ── */
    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1320px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2,
    .stMarkdown h3, .stMarkdown h4 {
        color: var(--navy) !important;
        font-weight: 700;
    }

    /* ── Body text and captions ── */
    p, li, span, label, div {
        color: var(--ink);
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
    }

    /* ── Hero banner ── */
    .hero {
        background: linear-gradient(125deg, #132238 0%, #244a76 62%, #168a88 125%);
        border-radius: 18px;
        padding: 1.8rem 2rem;
        color: white;
        box-shadow: 0 12px 32px rgba(19,34,56,0.15);
        margin-bottom: 1.4rem;
    }

    .hero *, .hero p, .hero h1, .hero-kicker {
        color: white !important;
    }

    .hero-kicker {
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        opacity: 0.88;
        font-weight: 700;
    }

    .hero h1 {
        margin: 0.3rem 0 0.4rem 0;
        font-size: 2.1rem;
        line-height: 1.15;
    }

    .hero p {
        margin: 0;
        font-size: 1rem;
        opacity: 0.95;
        line-height: 1.55;
    }

    /* ── Cards ── */
    .info-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(27,39,51,0.05);
        height: 100%;
        color: var(--ink) !important;
    }

    .info-card strong { color: var(--navy) !important; }

    .winner-card {
        background: linear-gradient(135deg, #ecfff9, #eff6ff);
        border: 1px solid #a9ded5;
        border-left: 5px solid var(--teal);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin: 0.8rem 0 1.2rem 0;
        color: var(--ink) !important;
    }

    .winner-card strong { color: var(--navy) !important; }

    .upload-note {
        background: #fffaf0;
        border: 1px solid #f3d9a7;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #5a4010 !important;
        margin-bottom: 0.9rem;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        padding: 0.85rem;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(27,39,51,0.05);
    }

    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricValue"] > div {
        color: var(--ink) !important;
    }

    /* ── Tabs ── */
    [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 4px;
    }

    [data-baseweb="tab"],
    button[data-baseweb="tab"],
    [role="tab"] {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
        background: transparent !important;
    }

    [data-baseweb="tab"]:hover,
    button[data-baseweb="tab"]:hover {
        color: var(--blue) !important;
        background: #eef4ff !important;
        border-radius: 6px 6px 0 0;
    }

    [data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"],
    [role="tab"][aria-selected="true"] {
        color: var(--blue) !important;
        border-bottom-color: var(--blue) !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #f1f5fb;
        border-right: 1px solid #dce4ee;
    }

    [data-testid="stSidebar"] * {
        color: var(--ink) !important;
    }

    /* ── Inputs and dropdowns ── */
    div[data-baseweb="select"] > div,
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 10px;
    }

    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] *,
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *,
    [role="option"],
    [role="listbox"] {
        color: var(--ink) !important;
        background-color: white !important;
    }

    div[data-baseweb="select"] > div {
        border-color: var(--line) !important;
    }

    /* ── Pills / model selector buttons ── */
    [data-testid="stPills"] button,
    [data-baseweb="button-group"] button {
        color: var(--ink) !important;
        background: white !important;
        border: 1.5px solid var(--line) !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.3rem 0.9rem !important;
    }

    [data-testid="stPills"] button[aria-pressed="true"],
    [data-baseweb="button-group"] button[aria-pressed="true"] {
        color: white !important;
        background: var(--blue) !important;
        border-color: var(--blue) !important;
    }

    [data-testid="stPills"] button:hover {
        border-color: var(--blue) !important;
        color: var(--blue) !important;
    }

    /* ── Static pills (info tags) ── */
        display: inline-block;
        background: var(--soft-blue);
        color: #214eab !important;
        border: 1px solid #cadbff;
        border-radius: 999px;
        padding: 0.26rem 0.62rem;
        margin: 0.12rem 0.2rem 0.12rem 0;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .small-muted {
        color: var(--muted) !important;
        font-size: 0.86rem;
    }

    /* ── Inline code ── */
    code, .stMarkdown code {
        background: #eef4ff !important;
        color: #1a4eb8 !important;
        border-radius: 4px;
        padding: 0.1em 0.35em;
        font-size: 0.9em;
    }

    /* ── File uploader button ── */
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzoneInstructions"] button {
        background-color: var(--blue) !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzoneInstructions"] button * {
        color: white !important;
    }

    /* ── Buttons ── */
    .stDownloadButton > button,
    .stButton > button {
        background-color: var(--blue) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton > button:hover,
    .stButton > button:hover {
        background-color: #1a56d6 !important;
        color: white !important;
    }

    .stDownloadButton > button *,
    .stButton > button * {
        color: white !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"],
    details {
        background: white !important;
        border: 1px solid var(--line) !important;
        border-radius: 10px !important;
    }

    [data-testid="stExpander"] summary,
    details summary,
    [data-testid="stExpander"] summary *,
    details summary * {
        color: var(--ink) !important;
        font-weight: 600 !important;
        background: transparent !important;
    }

    [data-testid="stExpander"] summary:hover,
    details summary:hover {
        color: var(--blue) !important;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data
def load_json(path: Path) -> dict[str, Any] | list[Any]:
    """Load a JSON artifact once per Streamlit session."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    """Load the precomputed metrics generated by the training script."""

    return pd.read_csv(RESULTS_DIR / "model_comparison.csv")


@st.cache_resource
def load_models() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load all model pipelines and their project metadata."""

    metadata = load_json(MODEL_DIR / "training_metadata.json")
    models: dict[str, Any] = {}
    for model_name, artifact_name in metadata["model_artifacts"].items():
        models[model_name] = joblib.load(MODEL_DIR / artifact_name)
    return models, metadata


def compute_metrics(
    y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Calculate the six metrics required by the assignment."""

    auc_value = (
        float(roc_auc_score(y_true, probabilities))
        if pd.Series(y_true).nunique() == 2
        else float("nan")
    )
    return {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "AUC": auc_value,
        "Precision": float(precision_score(y_true, predictions, zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, zero_division=0)),
        "F1": float(f1_score(y_true, predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predictions)),
    }


def validate_uploaded_frame(
    frame: pd.DataFrame, expected_features: list[str]
) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
    """Validate CSV columns and return model-ready features plus optional labels."""

    if frame.empty:
        raise ValueError("The uploaded CSV contains no data rows.")

    missing_features = [name for name in expected_features if name not in frame.columns]
    if missing_features:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing_features)
        )

    allowed_columns = set(expected_features + [TARGET_COLUMN])
    extra_columns = [name for name in frame.columns if name not in allowed_columns]

    feature_frame = frame[expected_features].copy()
    for column in expected_features:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="raise")

    if feature_frame.isna().any().any():
        missing_locations = np.argwhere(feature_frame.isna().to_numpy())
        first_row, first_column = missing_locations[0]
        raise ValueError(
            "Missing numeric value found at data row "
            f"{int(first_row) + 2}, column '{expected_features[int(first_column)]}'."
        )

    labels: pd.Series | None = None
    if TARGET_COLUMN in frame.columns:
        labels = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").astype(int)
        invalid_values = sorted(set(labels) - {0, 1})
        if invalid_values:
            raise ValueError(
                "The diagnosis column must contain only 0 (Benign) or 1 "
                f"(Malignant). Invalid values: {invalid_values}"
            )
        if labels.isna().any():
            raise ValueError("The diagnosis column contains missing values.")

    return feature_frame, labels, extra_columns


def prediction_table(
    original_frame: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: pd.Series | None,
) -> pd.DataFrame:
    """Build a downloadable results table."""

    results = original_frame.copy()
    results["predicted_diagnosis"] = predictions.astype(int)
    results["predicted_label"] = pd.Series(predictions).map(CLASS_LABELS)
    results["benign_probability"] = 1.0 - probabilities
    results["malignant_probability"] = probabilities
    if labels is not None:
        results["correct_prediction"] = predictions == labels.to_numpy()
    return results


def render_metric_cards(metrics: dict[str, float]) -> None:
    columns = st.columns(6)
    for column, metric_name in zip(columns, METRIC_COLUMNS):
        column.metric(metric_name, f"{metrics[metric_name]:.4f}")


def render_confusion_matrix(
    labels: pd.Series, predictions: np.ndarray, model_name: str
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
    )
    figure, axis = plt.subplots(figsize=(5.8, 4.6))
    display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d")
    axis.set_title(f"{model_name} - Confusion Matrix", pad=12, fontweight="bold")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_roc_curve(
    labels: pd.Series, probabilities: np.ndarray, model_name: str
) -> None:
    if labels.nunique() < 2:
        st.info("ROC curve and AUC require both Benign and Malignant rows.")
        return
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, probabilities)
    auc_value = roc_auc_score(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.8, 4.6))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2.4,
        label=f"{model_name} (AUC={auc_value:.4f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random baseline")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Receiver Operating Characteristic", pad=12, fontweight="bold")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)


def render_classification_report(
    labels: pd.Series, predictions: np.ndarray
) -> None:
    report = classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        output_dict=True,
        zero_division=0,
    )
    report_frame = pd.DataFrame(report).transpose()
    st.dataframe(
        report_frame.style.format(
            {
                "precision": "{:.4f}",
                "recall": "{:.4f}",
                "f1-score": "{:.4f}",
                "support": "{:.0f}",
            }
        ),
        use_container_width=True,
    )


def evaluate_all_models(
    models: dict[str, Any], features: pd.DataFrame, labels: pd.Series
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]
        rows.append({"ML Model Name": model_name, **compute_metrics(labels, predictions, probabilities)})
    return pd.DataFrame(rows)[["ML Model Name", *METRIC_COLUMNS]]


# Auto-generate artifacts if they are missing (e.g. fresh clone without results/).
if not (RESULTS_DIR / "model_comparison.csv").exists() or not (MODEL_DIR / "feature_names.json").exists():
    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT))
    from model.train_models import ProjectPaths as _ProjectPaths, train_and_evaluate as _train_and_evaluate  # noqa: E402
    with st.spinner("First run: training models and generating artifacts — this takes about 30 seconds…"):
        _train_and_evaluate(_ProjectPaths.from_root(PROJECT_ROOT))

models, metadata = load_models()
feature_names = load_json(MODEL_DIR / "feature_names.json")
comparison = load_comparison_table()

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">BITS WILP · Machine Learning · Assignment 2</div>
        <h1>Breast Cancer Classification</h1>
        <p>
            Compare five classification algorithms and test them on Breast Cancer
            Wisconsin diagnostic records. Upload a labelled CSV to reproduce all
            required metrics, or upload feature-only data to generate predictions.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Assignment Profile")
    st.write("**Student:** Divyendu Shekhar")
    st.write("**BITS ID:** 2025AC05281")
    st.write("**Programme:** M.Tech (AIML)")
    st.write("**Section:** 2")
    st.divider()
    st.markdown("### Dataset Snapshot")
    st.write(f"**Source instances:** {metadata['source_instances']}")
    st.write(f"**Training instances:** {metadata['training_instances']}")
    st.write(f"**Test instances:** {metadata['test_instances']}")
    st.write(f"**Features:** {metadata['feature_count']}")
    st.write("**Positive class:** Malignant (1)")

summary_columns = st.columns(4)
summary_columns[0].markdown(
    "<div class='info-card'><strong>Dataset</strong><br>Breast Cancer Wisconsin (Diagnostic)</div>",
    unsafe_allow_html=True,
)
summary_columns[1].markdown(
    "<div class='info-card'><strong>Task</strong><br>Binary classification</div>",
    unsafe_allow_html=True,
)
summary_columns[2].markdown(
    "<div class='info-card'><strong>Models</strong><br>5 prescribed classifiers</div>",
    unsafe_allow_html=True,
)
summary_columns[3].markdown(
    "<div class='info-card'><strong>Evaluation</strong><br>6 required metrics</div>",
    unsafe_allow_html=True,
)

upload_tab, overview_tab = st.tabs(
    ["Upload & Evaluate", "Model Comparison"]
)

with overview_tab:
    st.subheader("Saved experiment results")
    st.caption(
        "These values were produced from the 114-row supplied test_data.csv. "
        "All rows were excluded from model training."
    )

    display_comparison = comparison.copy()
    for metric in METRIC_COLUMNS:
        display_comparison[metric] = display_comparison[metric].round(4)
    st.dataframe(display_comparison, use_container_width=True, hide_index=True)

    chart_frame = comparison.set_index("ML Model Name")[METRIC_COLUMNS]
    st.bar_chart(chart_frame, height=430)

    st.markdown("#### Model-specific observations")
    observation_cards = {
        "Logistic Regression": (
            "Highest AUC (0.9970) and perfect malignant-class precision on the supplied test data. "
            "Its five false negatives reduced recall compared with Random Forest."
        ),
        "Decision Tree": (
            "Easy to interpret, but it produced the lowest AUC and MCC. Its single-tree structure "
            "was more sensitive to the training partition."
        ),
        "kNN": (
            "Standardisation allowed distance-based learning to work well. It achieved high AUC "
            "and precision, although six malignant records were missed."
        ),
        "Naive Bayes": (
            "Produced a strong AUC despite its conditional-independence assumption, but correlated "
            "cell-nucleus measurements reduced its classification accuracy and recall."
        ),
        "Random Forest (Ensemble)": (
            "Best overall balance: highest accuracy, recall, F1 and MCC, with only one false positive "
            "and three false negatives."
        ),
    }
    for model_name, observation in observation_cards.items():
        st.markdown(
            f"<div class='info-card'><strong>{model_name}</strong><br>{observation}</div><br>",
            unsafe_allow_html=True,
        )

with upload_tab:
    st.subheader("Upload CSV and Evaluate Model")

    # Assignment-prescribed order
    MODEL_ORDER = [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest (Ensemble)",
    ]
    MODEL_LABELS = {
        "Logistic Regression": "Logistic Regression",
        "Decision Tree": "Decision Tree Classifier",
        "kNN": "K-Nearest Neighbor Classifier",
        "Naive Bayes": "Naive Bayes Classifier",
        "Random Forest (Ensemble)": "Ensemble Model - Random Forest",
    }

    st.markdown("#### Select classification model(s)")
    selected_model_names = st.pills(
        "Model",
        options=MODEL_ORDER,
        format_func=lambda x: MODEL_LABELS[x],
        default=MODEL_ORDER,
        selection_mode="multi",
        label_visibility="collapsed",
    )
    if not selected_model_names:
        selected_model_names = MODEL_ORDER

    st.markdown("#### Upload test data (CSV)")
    st.markdown(
        "<div class='upload-note'>Upload a CSV with all 30 feature columns. "
        "Include the <code>diagnosis</code> column (0=Benign, 1=Malignant) to see "
        "Accuracy, AUC, Precision, Recall, F1 and MCC. Omit it for prediction-only use.</div>",
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="The file must contain all 30 numeric feature columns. Include the diagnosis column (0/1) to see evaluation metrics.",
    )

    use_supplied = st.checkbox(
        "Use the included test_data.csv (114 labelled rows)",
        value=True,
    )

    input_frame: pd.DataFrame | None = None
    if use_supplied:
        input_frame = pd.read_csv(DEFAULT_TEST_DATA)
    elif uploaded_file is not None:
        try:
            input_frame = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"The CSV could not be read: {error}")

    if input_frame is None:
        st.info("Upload a CSV file above, or tick the checkbox to use the supplied test data.")
    else:
        st.markdown("#### Input preview")
        st.caption(f"Rows: {len(input_frame):,} · Columns: {len(input_frame.columns):,}")
        st.dataframe(input_frame.head(12), use_container_width=True, hide_index=True)

        try:
            features, labels, extra_columns = validate_uploaded_frame(
                input_frame, feature_names
            )
            if extra_columns:
                st.warning(
                    "The following extra columns were ignored: " + ", ".join(extra_columns)
                )

            for selected_model_name in selected_model_names:
                st.divider()
                st.markdown(f"### {selected_model_name}")

                selected_model = models[selected_model_name]
                predictions = selected_model.predict(features).astype(int)
                probabilities = selected_model.predict_proba(features)[:, 1]
                results = prediction_table(
                    input_frame, predictions, probabilities, labels
                )

                st.success(
                    f"{selected_model_name} processed {len(features):,} rows successfully."
                )

                if labels is not None:
                    st.markdown("#### Evaluation metrics")
                    current_metrics = compute_metrics(labels, predictions, probabilities)
                    render_metric_cards(current_metrics)

                    st.markdown("#### Diagnostic views")
                    confusion_column, roc_column = st.columns(2)
                    with confusion_column:
                        render_confusion_matrix(labels, predictions, selected_model_name)
                    with roc_column:
                        render_roc_curve(labels, probabilities, selected_model_name)

                    st.markdown("#### Classification report")
                    render_classification_report(labels, predictions)
                else:
                    st.info(
                        "No diagnosis column was supplied, so evaluation metrics cannot be calculated. "
                        "Predictions and probabilities are shown below."
                    )

                st.markdown("#### Prediction results")
                compact_columns = [
                    column
                    for column in [
                        TARGET_COLUMN,
                        "predicted_diagnosis",
                        "predicted_label",
                        "benign_probability",
                        "malignant_probability",
                        "correct_prediction",
                    ]
                    if column in results.columns
                ]
                st.dataframe(
                    results[compact_columns].style.format(
                        {
                            "benign_probability": "{:.4f}",
                            "malignant_probability": "{:.4f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
                st.download_button(
                    f"Download {selected_model_name} predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_model_name.lower().replace(' ', '_')}_predictions.csv",
                    mime="text/csv",
                    key=f"download_{selected_model_name}",
                )

        except Exception as error:
            st.error(f"CSV validation or prediction failed: {error}")

        # ── Model comparison for selected models ──────────────────────────
        if labels is not None and len(selected_model_names) > 1:
            st.divider()
            st.markdown("### Model Comparison (selected models)")
            comparison_rows = []
            for name in selected_model_names:
                preds = models[name].predict(features).astype(int)
                probs = models[name].predict_proba(features)[:, 1]
                row = {"ML Model Name": MODEL_LABELS.get(name, name)}
                row.update(compute_metrics(labels, preds, probs))
                comparison_rows.append(row)
            comp_df = pd.DataFrame(comparison_rows)[["ML Model Name", *METRIC_COLUMNS]]
            for m in METRIC_COLUMNS:
                comp_df[m] = comp_df[m].round(4)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            st.bar_chart(comp_df.set_index("ML Model Name")[METRIC_COLUMNS], height=380)

            # Dynamic winner from selected models
            winner_row = comp_df.sort_values(
                by=["MCC", "F1", "AUC", "Accuracy"], ascending=False
            ).iloc[0]
            st.markdown(
                f"""
                <div class="winner-card">
                    <strong>Winner among selected models: {winner_row['ML Model Name']}</strong><br>
                    Selected using highest MCC, then F1, AUC, and Accuracy as tie-breakers.
                </div>
                """,
                unsafe_allow_html=True,
            )



st.markdown(
    "<p class='small-muted'>Dataset: UCI Breast Cancer Wisconsin (Diagnostic), loaded through "
    "scikit-learn. Target convention used by this project: 1 = Malignant, 0 = Benign.</p>",
    unsafe_allow_html=True,
)
