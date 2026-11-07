"""
Reusable matplotlib plots for streaming experiments.

Every function accepts ``save_path`` (write the figure to a file) and ``show``
(call ``plt.show()``; in a notebook this displays the figure inline). Both can
be combined, and ``show=False`` with no ``save_path`` simply returns the
figure. Every function returns the ``matplotlib.figure.Figure``; in a notebook
assign it (``_ = plot_...(...)``) to avoid the figure being displayed twice.
"""

import matplotlib.pyplot as plt
import numpy as np


def _series(values, name):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D sequence of numbers.")
    return arr


def _finish(fig, save_path, show):
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metric_over_time(metric_values, title="Metric over time", ylabel="Metric",
                          xlabel="Chunk", retrain_points=None, save_path=None,
                          show=True):
    """Plot a metric (e.g. accuracy) across chunks.

    Args:
        metric_values: 1D sequence, one value per chunk (NaN leaves a gap).
        title, ylabel, xlabel: text labels.
        retrain_points: optional 1-based chunk numbers drawn as dashed vertical
            lines (e.g. ``trainer.retrain_chunks()``).
        save_path: file to write (png, pdf, ...), or None.
        show: call ``plt.show()``.
    Returns:
        matplotlib Figure.
    Raises:
        ValueError: metric_values is not a non-empty 1D sequence.
    """
    values = _series(metric_values, "metric_values")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, values.size + 1), values, marker="o", markersize=3, label=ylabel)
    if retrain_points is not None:
        for i, x in enumerate(retrain_points):
            ax.axvline(x, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                       label="retrain" if i == 0 else None)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def compare_models(metric1, metric2, labels=("Model 1", "Model 2"),
                   title="Model comparison", ylabel="Metric", xlabel="Chunk",
                   save_path=None, show=True):
    """Compare two models on a streaming metric (e.g. different ``buffer_size``).

    Args:
        metric1, metric2: 1D sequences of per-chunk values (may differ in length).
        labels: two legend labels.
        title, ylabel, xlabel: text labels.
        save_path, show: see module docstring.
    Returns:
        matplotlib Figure.
    Raises:
        ValueError: invalid series or labels is not of length 2.
    """
    a, b = _series(metric1, "metric1"), _series(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries.")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, a.size + 1), a, marker="o", markersize=3, label=str(labels[0]))
    ax.plot(np.arange(1, b.size + 1), b, marker="s", markersize=3, label=str(labels[1]))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, save_path, show)


def plot_predictions_vs_ground_truth(y_true, y_pred, title="Predictions vs ground truth",
                                     save_path=None, show=True):
    """Visualise predictions against actual labels (e.g. the latest chunk).

    Left panel: true label (circles) and predicted label (crosses) per sample,
    wrong predictions in red. Right panel: confusion matrix with counts.

    Args:
        y_true, y_pred: 1D label sequences of equal, non-zero length (numbers or strings).
        title: figure title.
        save_path, show: see module docstring.
    Returns:
        matplotlib Figure.
    Raises:
        ValueError: empty input or length mismatch.
    """
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if yt.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred have inconsistent lengths: {yt.size} vs {yp.size}."
        )
    classes = np.unique(np.concatenate([yt, yp]))
    k = classes.size
    ti, pi = np.searchsorted(classes, yt), np.searchsorted(classes, yp)
    wrong = ti != pi
    x = np.arange(yt.size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax1.scatter(x, ti, facecolors="none", edgecolors="tab:blue", s=30, label="true")
    ax1.scatter(x[~wrong], pi[~wrong], marker="x", color="tab:green", s=25, label="predicted (correct)")
    ax1.scatter(x[wrong], pi[wrong], marker="x", color="tab:red", s=40, label="predicted (wrong)")
    ax1.set_yticks(range(k))
    ax1.set_yticklabels([str(c) for c in classes])
    ax1.set_xlabel("Sample in chunk")
    ax1.set_ylabel("Class")
    ax1.set_title(f"Accuracy {1 - wrong.mean():.3f} ({int(wrong.sum())} errors)")
    ax1.set_ylim(-0.5, k + 0.3)  # headroom for the legend
    ax1.legend(loc="upper center", ncol=3, fontsize=8, frameon=False)

    cm = np.bincount(ti * k + pi, minlength=k * k).reshape(k, k)
    image = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(k))
    ax2.set_yticks(range(k))
    ax2.set_xticklabels([str(c) for c in classes])
    ax2.set_yticklabels([str(c) for c in classes])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("True")
    ax2.set_title("Confusion matrix")
    if k <= 12:
        for r in range(k):
            for c in range(k):
                ax2.text(c, r, str(cm[r, c]), ha="center", va="center",
                         color="white" if cm[r, c] > cm.max() / 2 else "black")
    fig.colorbar(image, ax=ax2, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    return _finish(fig, save_path, show)
