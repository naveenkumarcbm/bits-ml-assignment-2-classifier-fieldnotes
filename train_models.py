"""Train and save the five assignment classifiers on the UCI WDBC dataset."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import TARGET_COLUMN, calculate_metrics

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_DATA_PATH = ROOT / "test_data.csv"
COMPARISON_PATH = ROOT / "model_comparison.csv"
RANDOM_STATE = 42


def build_classifiers() -> dict[str, object]:
    """Return the assignment's five classifiers with fixed, documented settings."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, random_state=RANDOM_STATE)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=7)
        ),
        "Gaussian Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(random_state=RANDOM_STATE), method="sigmoid", cv=5
            ),
        ),
    }


def main() -> None:
    dataset = load_breast_cancer(as_frame=True)
    features = dataset.data
    target_names = list(dataset.target_names)
    target = dataset.target.map(dict(enumerate(target_names))).rename(TARGET_COLUMN)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    x_test.join(y_test).to_csv(TEST_DATA_PATH, index=False)

    rows: list[dict[str, float | str]] = []
    for model_name, estimator in build_classifiers().items():
        estimator.fit(x_train, y_train)
        scores = calculate_metrics(estimator, x_test, y_test)
        rows.append({"ML Model Name": model_name, **scores})
        artifact = {
            "estimator": estimator,
            "features": list(features.columns),
            "classes": list(estimator.classes_),
            "target_column": TARGET_COLUMN,
            "positive_label": "malignant",
            "model_name": model_name,
        }
        filename = model_name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(artifact, MODEL_DIR / filename)
        print(f"\n{model_name}\n{classification_report(y_test, estimator.predict(x_test))}")

    comparison = pd.DataFrame(rows).set_index("ML Model Name")
    comparison.to_csv(COMPARISON_PATH, float_format="%.4f")
    print("\nHeld-out test metrics:")
    print(comparison.round(4).to_string())
    print(f"\nSaved labeled test rows to {TEST_DATA_PATH}")
    print(f"Saved model artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()