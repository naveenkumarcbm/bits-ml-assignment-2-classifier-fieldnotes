"""Shared dataset constants and evaluation helpers for the demo."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

TARGET_COLUMN = "diagnosis"
POSITIVE_LABEL = "malignant"


def calculate_metrics(
    estimator: Any, features: pd.DataFrame, target: pd.Series
) -> dict[str, float]:
    """Score a fitted binary classifier, treating malignant as the positive class."""
    predictions = estimator.predict(features)
    class_labels = list(estimator.classes_)
    positive_index = class_labels.index(POSITIVE_LABEL)
    positive_scores = estimator.predict_proba(features)[:, positive_index]
    positive_truth = target == POSITIVE_LABEL
    positive_predictions = pd.Series(predictions, index=target.index) == POSITIVE_LABEL

    return {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": float(roc_auc_score(positive_truth, positive_scores)),
        "Precision": float(
            precision_score(positive_truth, positive_predictions, zero_division=0)
        ),
        "Recall": float(
            recall_score(positive_truth, positive_predictions, zero_division=0)
        ),
        "F1": float(f1_score(positive_truth, positive_predictions, zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }