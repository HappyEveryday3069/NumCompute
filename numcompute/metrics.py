"""
Classification and regression evaluation metrics.
"""

from typing import Tuple
import numpy as np


# --- Classification Metrics ---

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy score."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = None
) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    if num_classes is None:
        max_val = max(
            np.max(y_true) if y_true.size > 0 else 0,
            np.max(y_pred) if y_pred.size > 0 else 0,
        )
        num_classes = max_val + 1

    # Vectorized indexing to build matrix
    indices = y_true * num_classes + y_pred
    cm = np.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Tuple[float, float, float]:
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = np.sum(cm, axis=0) - tp
    fn = np.sum(cm, axis=1) - tp

    if average == "micro":
        tp_sum, fp_sum, fn_sum = np.sum(tp), np.sum(fp), np.sum(fn)
        p = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        r = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1_score = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
        return float(p), float(r), float(f1_score)

    elif average == "macro":
        # Safe division to prevent 0/0 division errors
        p_per_class = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
        r_per_class = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
        
        pr_sum = p_per_class + r_per_class
        f1_per_class = np.where(pr_sum > 0, (2 * p_per_class * r_per_class) / pr_sum, 0.0)

        return (
            float(np.mean(p_per_class)),
            float(np.mean(r_per_class)),
            float(np.mean(f1_per_class)),
        )
    else:
        raise ValueError("Average must be either 'macro' or 'micro'.")


# --- Regression Metrics ---

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error (MSE)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


# --- Bonus: Binary ROC Curve & AUC ---

def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()

    # Sort thresholds in descending order
    desc_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_indices]
    y_true = y_true[desc_indices]

    # Distinct score thresholds
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Cumulative True Positives and False Positives
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps

    # Add origin (0, 0)
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[y_score[0] + 1.0, y_score[threshold_idxs]]

    total_positives = tps[-1]
    total_negatives = fps[-1]

    tpr = tps / total_positives if total_positives > 0 else np.zeros_like(tps)
    fpr = fps / total_negatives if total_negatives > 0 else np.zeros_like(fps)

    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x), np.asarray(y)
    # NumPy's trapezoidal integration
    return float(np.abs(np.trapezoid(y, x)))