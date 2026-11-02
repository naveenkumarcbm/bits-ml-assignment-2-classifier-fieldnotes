"""Streamlit interface for evaluating the saved classifiers on labeled CSV data."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import TARGET_COLUMN, calculate_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Classifier Fieldnotes",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root { --ink: #202b2b; --leaf: #146b59; --paper: #f5f7f2; --line: #dce5dc; }
    .stApp { background: var(--paper); color: var(--ink); }
    h1, h2, h3 { color: var(--ink); font-family: Georgia, 'Times New Roman', serif; }
    [data-testid="stMetric"] { background: white; border: 1px solid var(--line); padding: 14px; }
    [data-testid="stSidebar"] { background: #e8efe7; }
    .intro { border-left: 4px solid var(--leaf); padding: 4px 0 4px 16px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts() -> dict[str, dict[str, object]]:
    artifacts = {}
    for model_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path.name}. Run `python train_models.py` to create the models."
            )
        artifacts[model_name] = joblib.load(path)
    return artifacts


st.title("Classifier Fieldnotes")
st.markdown(
    '<div class="intro">Five fitted classifiers, one held-out dataset. '
    "Upload labeled measurements to compare their behavior.</div>",
    unsafe_allow_html=True,
)
st.write("")

try:
    artifacts = load_artifacts()
except (FileNotFoundError, OSError) as error:
    st.error(str(error))
    st.stop()

first_artifact = next(iter(artifacts.values()))
feature_names = list(first_artifact["features"])
class_labels = list(first_artifact["classes"])

with st.sidebar:
    st.subheader("Evaluation setup")
    selected_model = st.selectbox("Inspect model", list(artifacts))
    uploaded_file = st.file_uploader("Upload labeled test CSV", type=["csv"])
    st.caption(
        f"Expected: {len(feature_names)} numeric measurements and a "
        f"`{TARGET_COLUMN}` column. No fitting happens in this app."
    )
    sample_path = ROOT / "test_data.csv"
    if sample_path.exists():
        st.download_button(
            "Download sample test data",
            data=sample_path.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            width="stretch",
        )

if uploaded_file is None:
    st.info("Upload the included labeled test CSV to see the model comparison.")
    st.stop()

try:
    evaluation_data = pd.read_csv(uploaded_file)
except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as error:
    st.error(f"The uploaded file could not be read as CSV: {error}")
    st.stop()

required_columns = feature_names + [TARGET_COLUMN]
missing_columns = [column for column in required_columns if column not in evaluation_data]
if missing_columns:
    st.error(f"Missing required columns: {', '.join(missing_columns)}")
    st.stop()

evaluation_data = evaluation_data[required_columns].copy()
if evaluation_data.empty or evaluation_data.isna().any().any():
    st.error("The CSV must contain at least one complete row with no missing values.")
    st.stop()

observed_classes = set(evaluation_data[TARGET_COLUMN].astype(str))
if not observed_classes.issubset(set(class_labels)):
    st.error(f"Target values must be drawn from: {', '.join(class_labels)}")
    st.stop()
if observed_classes != set(class_labels):
    st.error(f"Include at least one row for each target class: {', '.join(class_labels)}")
    st.stop()

try:
    evaluation_data[feature_names] = evaluation_data[feature_names].apply(
        pd.to_numeric, errors="raise"
    )
except (TypeError, ValueError):
    st.error("All model input features must contain numeric values.")
    st.stop()

x_test = evaluation_data[feature_names]
y_test = evaluation_data[TARGET_COLUMN].astype(str)
comparison_rows = []
all_predictions = {}
for model_name, artifact in artifacts.items():
    estimator = artifact["estimator"]
    metrics = calculate_metrics(estimator, x_test, y_test)
    all_predictions[model_name] = estimator.predict(x_test)
    comparison_rows.append({"ML Model Name": model_name, **metrics})

comparison = pd.DataFrame(comparison_rows).set_index("ML Model Name")
sample_count = len(evaluation_data)
st.subheader("Held-out evaluation")
summary_columns = st.columns(3)
summary_columns[0].metric("Rows scored", f"{sample_count:,}")
summary_columns[1].metric("Input features", f"{len(feature_names)}")
summary_columns[2].metric("Models compared", f"{len(artifacts)}")
st.dataframe(
    comparison,
    column_config={
        column: st.column_config.NumberColumn(format="%.3f")
        for column in comparison.columns
    },
    width="stretch",
)

st.subheader(selected_model)
selected_artifact = artifacts[selected_model]
selected_predictions = all_predictions[selected_model]
selected_metrics = comparison.loc[selected_model]
metric_columns = st.columns(6)
for column, (metric_name, value) in zip(metric_columns, selected_metrics.items()):
    column.metric(metric_name, f"{value:.3f}")

details_left, details_right = st.columns([1, 1])
with details_left:
    st.markdown("#### Confusion matrix")
    matrix = confusion_matrix(y_test, selected_predictions, labels=class_labels)
    st.dataframe(
        pd.DataFrame(
            matrix,
            index=[f"Actual: {label}" for label in class_labels],
            columns=[f"Predicted: {label}" for label in class_labels],
        ),
        width="stretch",
    )
with details_right:
    st.markdown("#### Classification report")
    report = classification_report(
        y_test,
        selected_predictions,
        labels=class_labels,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")

st.caption(
    "Educational classification exercise using the Wisconsin Diagnostic Breast Cancer dataset. "
    "This output is not medical advice or a diagnostic tool."
)