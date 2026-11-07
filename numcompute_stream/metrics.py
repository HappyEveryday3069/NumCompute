"""
Classification and regression metrics: batch functions and streaming classes.

The batch functions (``accuracy``, ``confusion_matrix``, ...) evaluate a whole
array at once. The ``Streaming*`` classes accumulate over chunks through
``update(y_true_chunk, y_pred_chunk)`` / ``result()`` / ``reset()`` and can
optionally report a rolling window over the most recent samples.
"""

from typing import Tuple
import numpy as np

from .utils import has_missing


# ---------------------------------------------------------------------------
# Input validation helpers shared by the batch metrics
# ---------------------------------------------------------------------------

def _label_vector(values, name):
    """Return ``values`` as a 1D array (scalars and (n, 1) / (1, n) arrays are flattened).

    Raises:
        ValueError: more than one non-trivial dimension, or NaN/None (also inside object arrays).
    """
    arr = np.asarray(values)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim == 2 and 1 in arr.shape:
        arr = arr.ravel()
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D (n_samples,); got shape {arr.shape}.")
    if has_missing(arr):
        raise ValueError(f"{name} contains missing values (NaN/None).")
    return arr


def _check_pair(y_true, y_other, other_name="y_pred"):
    """Validate two label/score vectors and make sure they have the same length."""
    yt = _label_vector(y_true, "y_true")
    yo = _label_vector(y_other, other_name)
    if yt.shape[0] != yo.shape[0]:
        raise ValueError(
            f"y_true and {other_name} have inconsistent lengths: {yt.shape[0]} vs {yo.shape[0]}."
        )
    return yt, yo


def _class_indices(values, name):
    """Convert labels to non-negative int64 class indices (rejects floats like 1.5, negatives)."""
    arr = np.asarray(values)
    if arr.size == 0:
        return arr.astype(np.int64)
    try:
        as_float = arr.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain integer class labels.") from None
    if (not np.isfinite(as_float).all() or np.any(as_float != np.floor(as_float))
            or np.any(as_float < 0)):
        raise ValueError(f"{name} must contain non-negative integer class labels.")
    return as_float.astype(np.int64)


def _safe_divide(num, den, zero_division=0.0):
    """Elementwise ``num / den`` that yields ``zero_division`` where ``den == 0``.

    Unlike ``np.where(den > 0, num / den, 0)`` the division is only evaluated where the
    denominator is non-zero, so no ``RuntimeWarning`` is raised for 0/0.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full(np.broadcast(num, den).shape, float(zero_division))
    np.divide(num, den, out=out, where=den != 0)
    return out


# --- Classification Metrics (batch) ---

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions equal to the true labels.

    Args:
        y_true, y_pred: label vectors of equal length (1D, or (n, 1)).
    Returns:
        accuracy in [0, 1]; NaN for empty input.
    Raises:
        ValueError: shape/length mismatch or NaN labels.
    """
    yt, yp = _check_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan")
    return float(np.mean(yt == yp))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = None
) -> np.ndarray:
    """Confusion matrix of non-negative integer class labels (rows = true, cols = predicted).

    Args:
        y_true, y_pred: integer label vectors of equal length.
        num_classes: matrix size; default ``max(label) + 1``.
    Returns:
        int array ``(num_classes, num_classes)``.
    Raises:
        ValueError: shape/length mismatch, non-integer or negative labels, or
            ``num_classes`` that is not a positive integer larger than every label.
    """
    yt, yp = _check_pair(y_true, y_pred)
    yt, yp = _class_indices(yt, "y_true"), _class_indices(yp, "y_pred")
    max_label = int(max(yt.max() if yt.size else 0, yp.max() if yp.size else 0))
    if num_classes is None:
        num_classes = max_label + 1
    else:
        if isinstance(num_classes, (bool, np.bool_)) or int(num_classes) != num_classes or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")
        num_classes = int(num_classes)
        if max_label >= num_classes:
            raise ValueError(
                f"num_classes={num_classes} is too small for label {max_label}."
            )
    indices = yt * num_classes + yp
    return np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro"
) -> Tuple[float, float, float]:
    """Precision, recall and F1 from the confusion matrix.

    Args:
        y_true, y_pred: integer label vectors of equal length.
        average: ``"macro"`` (class mean), ``"micro"`` (global counts) or
            ``"weighted"`` (class mean weighted by support).
    Returns:
        ``(precision, recall, f1)``; a ratio with a zero denominator counts as 0.0
        (no warning is emitted); all NaN for empty input.
    Raises:
        ValueError: invalid ``average`` or invalid labels (see ``confusion_matrix``).
    """
    if average not in ("macro", "micro", "weighted"):
        raise ValueError("Average must be 'macro', 'micro' or 'weighted'.")
    yt, _ = _check_pair(y_true, y_pred)
    if yt.size == 0:
        return float("nan"), float("nan"), float("nan")
    cm = confusion_matrix(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    if average == "micro":
        tp_s, fp_s, fn_s = tp.sum(), fp.sum(), fn.sum()
        return (float(_safe_divide(tp_s, tp_s + fp_s)),
                float(_safe_divide(tp_s, tp_s + fn_s)),
                float(_safe_divide(2 * tp_s, 2 * tp_s + fp_s + fn_s)))
    p = _safe_divide(tp, tp + fp)
    r = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f1.mean())
    support = cm.sum(axis=1).astype(float)
    w = support / support.sum()
    return float((p * w).sum()), float((r * w).sum()), float((f1 * w).sum())


# --- Regression Metrics (batch) ---

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error.

    Args:
        y_true, y_pred: numeric arrays of *identical* shape (no broadcasting, so an
            (n,) vs (n, 1) mix-up is reported instead of silently producing an n x n result).
    Returns:
        the mean of the squared differences; NaN for empty input.
    Raises:
        ValueError: non-numeric input or different shapes.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape; got {yt.shape} and {yp.shape}."
        )
    if yt.size == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


# --- Binary ROC curve and AUC (batch) ---

def roc_curve(
    y_true: np.ndarray, y_score: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC curve of a binary classifier.

    Args:
        y_true: labels in {0, 1}.
        y_score: finite scores, higher = more likely class 1; same length as y_true.
    Returns:
        ``(fpr, tpr, thresholds)``; the first point is (0, 0) with threshold ``inf``.
        If a class is absent its rate is all zeros.
    Raises:
        ValueError: empty input, length mismatch, labels not in {0, 1}, NaN/inf scores.
    """
    yt, ys = _check_pair(y_true, y_score, "y_score")
    if yt.size == 0:
        raise ValueError("roc_curve needs at least one sample.")
    if not np.isin(yt, (0, 1)).all():
        raise ValueError("y_true must contain only the labels 0 and 1.")
    try:
        ys = ys.astype(float)
    except (TypeError, ValueError):
        raise ValueError("y_score must be numeric.") from None
    if not np.isfinite(ys).all():
        raise ValueError("y_score must be finite.")
    yt = yt.astype(np.int64)

    order = np.argsort(ys, kind="stable")[::-1]
    ys, yt = ys[order], yt[order]
    distinct = np.flatnonzero(np.diff(ys))
    threshold_idxs = np.r_[distinct, yt.size - 1]
    tps = np.cumsum(yt)[threshold_idxs]
    fps = (1 + threshold_idxs) - tps
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    thresholds = np.r_[np.inf, ys[threshold_idxs]]
    tpr = _safe_divide(tps, tps[-1])
    fpr = _safe_divide(fps, fps[-1])
    return fpr, tpr, thresholds


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """Area under a curve by the trapezoidal rule.

    Args:
        x, y: coordinates of equal length (>= 2 points); ``x`` must be monotonic.
    Raises:
        ValueError: shape mismatch, fewer than two points, or non-monotonic x.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length; got {x.size} and {y.size}.")
    if x.size < 2:
        raise ValueError("auc needs at least two points.")
    dx = np.diff(x)
    if not (np.all(dx >= 0) or np.all(dx <= 0)):
        raise ValueError("x must be monotonic (all increasing or all decreasing).")
    _trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz
    return float(np.abs(_trapezoid(y, x)))


# ---------------------------------------------------------------------------
# Streaming metrics
# ---------------------------------------------------------------------------

def _check_window(window):
    if window is None:
        return None
    if isinstance(window, (bool, np.bool_)) or int(window) != window or window < 1:
        raise ValueError("window must be None or a positive integer.")
    return int(window)


def _check_label_pair(y_true, y_pred):
    """Return 1D arrays of equal length; reject NaN labels."""
    yt, yp = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred have inconsistent lengths: {yt.shape[0]} vs {yp.shape[0]}."
        )
    for name, arr in (("y_true", yt), ("y_pred", yp)):
        if arr.dtype.kind == "f" and np.isnan(arr).any():
            raise ValueError(f"{name} contains NaN labels.")
    return yt, yp


def _confusion_from_labels(yt, yp, classes):
    """Confusion matrix (rows = true, cols = predicted) over sorted ``classes``."""
    k = classes.shape[0]
    ti = np.searchsorted(classes, yt)
    pi = np.searchsorted(classes, yp)
    return np.bincount(ti * k + pi, minlength=k * k).reshape(k, k).astype(np.int64)


class StreamingMetric:
    """Base class of streaming metrics.

    Subclasses implement ``update`` and ``result``; ``reset`` clears all state.

    Class attributes:
        uses_scores: True for metrics that need class probabilities / scores
            (they are fed through ``update_proba`` by ``StreamTrainer``).
        is_scalar: False for metrics whose ``result()`` is not a single number
            (e.g. the confusion matrix); such metrics are not written to the trainer log.

    Args:
        window: None for a cumulative metric, otherwise the number of most
            recent *samples* the metric is computed over (rolling window).
    """

    uses_scores = False
    is_scalar = True

    def __init__(self, window=None):
        self.window = _check_window(window)
        self.reset()

    def reset(self):
        """Forget everything seen so far."""
        self.n_samples_seen_ = 0
        return self

    def update(self, y_true, y_pred):  # pragma: no cover - abstract
        raise NotImplementedError

    def result(self):  # pragma: no cover - abstract
        raise NotImplementedError


class _LabelMetric(StreamingMetric):
    """Metrics derived from a (growing) confusion matrix of class labels."""

    def reset(self):
        super().reset()
        self._classes = None
        self._cm = None
        self._wt = None  # rolling window storage (true labels)
        self._wp = None  # rolling window storage (predicted labels)
        return self

    def update(self, y_true, y_pred):
        """Accumulate one chunk of true and predicted labels; returns self.

        Raises:
            ValueError: length mismatch or NaN labels.
        """
        yt, yp = _check_label_pair(y_true, y_pred)
        if yt.size == 0:
            return self
        self.n_samples_seen_ += yt.size
        if self.window is not None:
            self._wt = yt if self._wt is None else np.concatenate([self._wt, yt])
            self._wp = yp if self._wp is None else np.concatenate([self._wp, yp])
            self._wt, self._wp = self._wt[-self.window:], self._wp[-self.window:]
            return self
        labels = np.unique(np.concatenate([yt, yp]))
        if self._classes is None:
            self._classes = labels
            self._cm = np.zeros((labels.size, labels.size), dtype=np.int64)
        else:
            merged = np.union1d(self._classes, labels)
            if merged.size != self._classes.size:  # new classes: expand matrix
                grown = np.zeros((merged.size, merged.size), dtype=np.int64)
                old = np.searchsorted(merged, self._classes)
                grown[np.ix_(old, old)] = self._cm
                self._classes, self._cm = merged, grown
        self._cm += _confusion_from_labels(yt, yp, self._classes)
        return self

    def _state(self):
        """Return ``(classes, confusion_matrix)`` for the current window/total."""
        if self.window is None:
            if self._cm is None:
                return np.array([]), np.zeros((0, 0), dtype=np.int64)
            return self._classes, self._cm
        if self._wt is None:
            return np.array([]), np.zeros((0, 0), dtype=np.int64)
        classes = np.unique(np.concatenate([self._wt, self._wp]))
        return classes, _confusion_from_labels(self._wt, self._wp, classes)

    @property
    def classes_(self):
        """Sorted array of the class labels seen (in the window, if rolling)."""
        return self._state()[0].copy()


class StreamingConfusionMatrix(_LabelMetric):
    """Confusion matrix that accumulates over chunks and grows with new classes.

    ``result()`` returns an int array ``(k, k)``: rows are true classes, columns
    predicted classes, both ordered like ``classes_``.
    """

    is_scalar = False

    def result(self):
        return self._state()[1].copy()


class StreamingAccuracy(_LabelMetric):
    """Fraction of correct predictions; NaN before any sample is seen."""

    def result(self):
        _, cm = self._state()
        total = cm.sum()
        return float(np.trace(cm) / total) if total > 0 else float("nan")


class _PRFMetric(_LabelMetric):
    """Shared logic of precision / recall / F1 with averaging.

    Args:
        average: ``"macro"`` (unweighted class mean), ``"micro"`` (global
            counts), ``"weighted"`` (class mean weighted by support) or
            ``"binary"`` (only class ``pos_label``).
        pos_label: positive class for ``average="binary"``.
        zero_division: value returned when a ratio has a zero denominator.
        window: see ``StreamingMetric``.
    """

    _AVERAGES = ("macro", "micro", "weighted", "binary")

    def __init__(self, average="macro", pos_label=1, zero_division=0.0, window=None):
        if average not in self._AVERAGES:
            raise ValueError(f"average must be one of {self._AVERAGES}; got {average!r}.")
        self.average = average
        self.pos_label = pos_label
        self.zero_division = float(zero_division)
        super().__init__(window=window)

    def _ratio(self, num, den):
        num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
        safe = np.where(den > 0, den, 1.0)
        return np.where(den > 0, num / safe, self.zero_division)

    def _score(self, tp, fp, fn):  # pragma: no cover - abstract
        raise NotImplementedError

    def result(self):
        classes, cm = self._state()
        if cm.sum() == 0:
            return float("nan")
        tp = np.diag(cm).astype(float)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp
        if self.average == "micro":
            return float(self._score(tp.sum(), fp.sum(), fn.sum()))
        per_class = self._score(tp, fp, fn)
        if self.average == "macro":
            return float(per_class.mean())
        if self.average == "weighted":
            support = cm.sum(axis=1).astype(float)
            return float((per_class * support).sum() / support.sum())
        hit = np.flatnonzero(classes == self.pos_label)  # binary
        return float(per_class[hit[0]]) if hit.size else self.zero_division


class StreamingPrecision(_PRFMetric):
    """Precision = TP / (TP + FP), averaged as configured."""

    def _score(self, tp, fp, fn):
        return self._ratio(tp, tp + fp)


class StreamingRecall(_PRFMetric):
    """Recall = TP / (TP + FN), averaged as configured."""

    def _score(self, tp, fp, fn):
        return self._ratio(tp, tp + fn)


class StreamingF1(_PRFMetric):
    """F1 = 2TP / (2TP + FP + FN), averaged as configured."""

    def _score(self, tp, fp, fn):
        return self._ratio(2 * tp, 2 * tp + fp + fn)


class StreamingMSE(StreamingMetric):
    """Mean squared error over a stream (regression); NaN before any sample."""

    def reset(self):
        super().reset()
        self._sse = 0.0
        self._n = 0
        self._sq = None
        return self

    def update(self, y_true, y_pred):
        """Accumulate one chunk of targets and predictions; returns self."""
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
        if yt.shape != yp.shape:
            raise ValueError(
                f"y_true and y_pred have inconsistent lengths: {yt.size} vs {yp.size}."
            )
        if yt.size == 0:
            return self
        sq = (yt - yp) ** 2
        self.n_samples_seen_ += yt.size
        if self.window is None:
            self._sse += float(sq.sum())
            self._n += yt.size
        else:
            self._sq = sq if self._sq is None else np.concatenate([self._sq, sq])
            self._sq = self._sq[-self.window:]
        return self

    def result(self):
        if self.window is None:
            return self._sse / self._n if self._n else float("nan")
        return float(self._sq.mean()) if self._sq is not None else float("nan")


def _average_ranks(values):
    """1-based ranks with ties replaced by their average rank (vectorised)."""
    uniq, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts
    return (first + (counts + 1) / 2.0)[inverse.ravel()]


class StreamingAUC(StreamingMetric):
    """Binary ROC-AUC accumulated over a stream.

    Two modes:

    * exact (default, ``n_bins=None``): stores all scores (or the last
      ``window`` samples) and evaluates the Mann-Whitney U statistic with
      tie-averaged ranks.
    * approximate (``n_bins=B``): keeps only two histograms of scores in
      [0, 1] (O(B) memory); ties inside a bin count as half. Does not support
      ``window``.

    ``update(y_true, y_score)`` takes labels in {0, 1} and finite scores.
    ``update_proba(y_true, proba, classes)`` takes arbitrary labels plus the
    ``predict_proba`` matrix and evaluates class ``pos_label`` against the rest
    (one-vs-rest for multi-class models); ``StreamTrainer`` uses this entry point.
    ``result()`` is NaN while only one class has been seen.

    Args:
        window: None or the number of most recent samples (exact mode only).
        n_bins: None for the exact AUC, or an int >= 2 for the histogram approximation.
        pos_label: the positive class used by ``update_proba``.

    Raises:
        ValueError: invalid labels/scores, ``window`` combined with ``n_bins``.
    """

    uses_scores = True

    def __init__(self, window=None, n_bins=None, pos_label=1):
        if n_bins is not None:
            if isinstance(n_bins, (bool, np.bool_)) or int(n_bins) != n_bins or n_bins < 2:
                raise ValueError("n_bins must be None or an integer >= 2.")
            if window is not None:
                raise ValueError("window is not supported together with n_bins.")
            n_bins = int(n_bins)
        self.n_bins = n_bins
        self.pos_label = pos_label
        super().__init__(window=window)

    def reset(self):
        super().reset()
        self._labels = None
        self._scores = None
        if self.n_bins is not None:
            self._pos = np.zeros(self.n_bins)
            self._neg = np.zeros(self.n_bins)
        return self

    def update(self, y_true, y_score):
        """Accumulate one chunk; returns self."""
        yt = np.asarray(y_true).ravel()
        ys = np.asarray(y_score, dtype=float).ravel()
        if yt.shape[0] != ys.shape[0]:
            raise ValueError(
                f"y_true and y_score have inconsistent lengths: {yt.shape[0]} vs {ys.shape[0]}."
            )
        if yt.size == 0:
            return self
        if not np.isin(yt, (0, 1)).all():
            raise ValueError("y_true must contain only the labels 0 and 1.")
        if not np.isfinite(ys).all():
            raise ValueError("y_score must be finite.")
        yt = yt.astype(int)
        self.n_samples_seen_ += yt.size
        if self.n_bins is not None:
            if ys.min() < 0.0 or ys.max() > 1.0:
                raise ValueError("y_score must lie in [0, 1] when n_bins is used.")
            b = np.minimum((ys * self.n_bins).astype(int), self.n_bins - 1)
            self._pos += np.bincount(b[yt == 1], minlength=self.n_bins)
            self._neg += np.bincount(b[yt == 0], minlength=self.n_bins)
            return self
        self._labels = yt if self._labels is None else np.concatenate([self._labels, yt])
        self._scores = ys if self._scores is None else np.concatenate([self._scores, ys])
        if self.window is not None:
            self._labels, self._scores = self._labels[-self.window:], self._scores[-self.window:]
        return self

    def update_proba(self, y_true, proba, classes):
        """Accumulate a chunk from a ``predict_proba`` matrix (``pos_label`` vs rest).

        Args:
            y_true: true labels, shape ``(n,)``.
            proba: probabilities, shape ``(n, len(classes))``.
            classes: class label of each ``proba`` column. If ``pos_label`` is not among
                them (the model has never seen it) its score is 0 for every sample.
        Raises:
            ValueError: inconsistent shapes, missing labels, or a ``proba`` matrix with
                non-finite entries, values outside [0, 1] or rows that do not sum to 1.
                (``update(y_true, y_score)`` still accepts any finite ranking score.)
        """
        y = _label_vector(y_true, "y_true")
        proba = np.asarray(proba, dtype=float)
        classes = np.asarray(classes)
        if proba.ndim != 2 or proba.shape[0] != y.shape[0] or proba.shape[1] != classes.shape[0]:
            raise ValueError(
                f"proba must have shape (n_samples, n_classes) = ({y.shape[0]}, {classes.shape[0]}); "
                f"got {proba.shape}."
            )
        if not np.isfinite(proba).all():
            raise ValueError("proba contains NaN or infinite values.")
        if proba.size and (proba.min() < 0.0 or proba.max() > 1.0):
            raise ValueError("proba values must lie in [0, 1].")
        if proba.size and not np.allclose(proba.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("each row of proba must sum to 1 (within 1e-6).")
        hit = np.flatnonzero(classes == self.pos_label)
        score = proba[:, hit[0]] if hit.size else np.zeros(y.shape[0])
        return self.update((y == self.pos_label).astype(int), score)

    def result(self):
        if self.n_bins is not None:
            n_pos, n_neg = self._pos.sum(), self._neg.sum()
            if n_pos == 0 or n_neg == 0:
                return float("nan")
            neg_below = np.cumsum(self._neg) - self._neg
            return float((self._pos * (neg_below + 0.5 * self._neg)).sum() / (n_pos * n_neg))
        if self._labels is None:
            return float("nan")
        n_pos = int((self._labels == 1).sum())
        n_neg = self._labels.size - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        ranks = _average_ranks(self._scores)
        u = ranks[self._labels == 1].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))
