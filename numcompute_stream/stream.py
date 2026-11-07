"""
StreamTrainer: run a model or pipeline over a data stream with per-chunk logging.

The trainer implements *prequential* (test-then-train) evaluation: every new
chunk is first scored with the current model and only then used for training.
That is the standard way to measure an online learner, because each sample is
predicted before the model has seen it. While the model cannot predict yet
(e.g. a buffered tree before its first retrain) the chunk is logged as
``scored=False`` and its metrics are NaN.

Metrics that need probabilities (``StreamingAUC``) are fed from the model's
``predict_proba``; all other metrics receive the hard predictions of ``predict``.
"""

import logging
import sys
import time
import types
from collections import deque

import numpy as np

from .metrics import StreamingAccuracy, StreamingF1, StreamingMetric
from .utils import NotFittedError, check_y, iter_chunks  # noqa: F401 (iter_chunks is re-exported)

_LOGGER = logging.getLogger(__name__)


def estimate_memory(obj):
    """Approximate the memory footprint of an object graph in bytes.

    Walks attributes, dicts, lists, tuples, sets and deques and counts every object once
    through ``sys.getsizeof``. NumPy arrays need care:

    * an array that owns its buffer: ``sys.getsizeof`` already includes the data
      (``nbytes``), so nothing is added on top (adding ``nbytes`` would count it twice);
    * a *view*: ``sys.getsizeof`` only covers the small array header, so the walk continues
      into ``.base`` and the owning buffer is counted exactly once, however many views of it
      (or the owner itself) are reachable.

    Classes, functions and modules are ignored. This is an estimate, not an allocator
    measurement (fragmentation and interpreter-wide shared objects are not modelled).
    """
    seen, total, stack = set(), 0, [obj]
    skip = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType)
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, skip):
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                stack.append(item.base)
            if item.dtype == object:
                stack.extend(item.ravel().tolist())
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset, deque)):
            stack.extend(item)
        elif hasattr(item, "__dict__"):
            stack.append(vars(item))
    return int(total)


def _final_estimator(model):
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


def _classes_of(model):
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(_final_estimator(model), "classes_", None)
    return classes


def _check_chunk(X, y):
    """Validate chunk shape/length only; dtype checks are left to the model.

    This keeps the trainer usable with categorical (string) features that a
    ``OneHotEncoder`` step turns into numbers.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2D with shape (n_samples, n_features); got shape {X.shape}."
        )
    return X, check_y(y, X.shape[0])


class StreamTrainer:
    """Manage a model (or ``Pipeline``), metrics and logging over a stream of chunks.

    Args:
        model: object with ``partial_fit(X, y)`` and ``predict(X)`` (e.g. a
            ``Pipeline`` ending in ``DecisionTreeClassifier``). It must also provide
            ``predict_proba(X)`` if a probability-based metric is used.
        metrics: dict ``name -> StreamingMetric``. Default: cumulative ``accuracy``,
            ``rolling_accuracy`` (last ``rolling_window`` samples) and ``macro_f1``.
            Hard-label metrics get ``update(y_true, y_pred)``; metrics with
            ``uses_scores`` (``StreamingAUC``) get ``update_proba(y_true, proba, classes)``.
            Metrics whose result is not a single number (``StreamingConfusionMatrix``) are
            updated but not written to the log; read them from ``trainer.metrics[name]``.
        rolling_window: window (samples) of the default rolling accuracy.
        log_memory: record ``estimate_memory(model)`` after each fit.
        logger: ``logging.Logger`` to use (default: this module's logger); one
            INFO line is emitted per processed chunk.
        verbose: also print each log line.

    Attributes:
        history: list of per-chunk dict records (see ``process_chunk``).
        last_y_true, last_y_pred: labels / predictions of the latest scored chunk.

    Raises:
        ValueError: model without ``partial_fit``/``predict`` (or without ``predict_proba``
            when a probability metric is requested), or a metric that is not a StreamingMetric.
    """

    def __init__(self, model, metrics=None, rolling_window=500, log_memory=True,
                 logger=None, verbose=False):
        for method in ("partial_fit", "predict"):
            if not callable(getattr(model, method, None)):
                raise ValueError(f"model must implement {method}().")
        if metrics is None:
            metrics = {"accuracy": StreamingAccuracy(),
                       "rolling_accuracy": StreamingAccuracy(window=rolling_window),
                       "macro_f1": StreamingF1(average="macro")}
        for name, metric in metrics.items():
            if not isinstance(metric, StreamingMetric):
                raise ValueError(f"metric {name!r} must be a StreamingMetric.")
        self._needs_proba = any(m.uses_scores for m in metrics.values())
        if self._needs_proba and not callable(getattr(model, "predict_proba", None)):
            raise ValueError("model must implement predict_proba() to use probability metrics (e.g. StreamingAUC).")
        self.model = model
        self.metrics = metrics
        self.log_memory = log_memory
        self.logger = logger or _LOGGER
        self.verbose = verbose
        self.reset()

    def reset(self):
        """Clear history and metrics (the model itself is left untouched)."""
        for metric in self.metrics.values():
            metric.reset()
        self.history = []
        self.n_chunks_ = 0
        self.n_samples_ = 0
        self.last_y_true = None
        self.last_y_pred = None
        return self

    def _logged_metrics(self):
        return {name: m for name, m in self.metrics.items() if m.is_scalar}

    # -- single-chunk operations ---------------------------------------------

    def score_chunk(self, X, y):
        """Score a chunk with the *current* model and update the streaming metrics.

        Returns:
            dict with ``scored`` (bool), ``chunk_accuracy``, ``score_time_s`` and
            the current value of each scalar metric. If the model cannot predict yet,
            ``scored`` is False and all values are NaN.
        Raises:
            ValueError: invalid X / y (shape mismatch, NaN labels, ...).
        """
        X, y = _check_chunk(X, y)
        out = {"scored": False, "chunk_accuracy": float("nan"), "score_time_s": 0.0}
        out.update({name: float("nan") for name in self._logged_metrics()})
        if X.shape[0] == 0:
            return out
        start = time.perf_counter()
        try:
            pred = np.asarray(self.model.predict(X))
            proba = np.asarray(self.model.predict_proba(X)) if self._needs_proba else None
        except NotFittedError:
            return out
        out["score_time_s"] = time.perf_counter() - start
        classes = _classes_of(self.model) if self._needs_proba else None
        for metric in self.metrics.values():
            if metric.uses_scores:
                metric.update_proba(y, proba, classes)
            else:
                metric.update(y, pred)
        out["scored"] = True
        out["chunk_accuracy"] = float(np.mean(pred == y))
        out.update({name: float(m.result()) for name, m in self._logged_metrics().items()})
        self.last_y_true, self.last_y_pred = y, pred
        return out

    def fit_chunk(self, X, y):
        """Train the model on one chunk with ``partial_fit``.

        Returns:
            dict with ``fit_time_s``, ``memory_bytes`` (0 if ``log_memory`` is False),
            ``retrained`` (bool) and ``n_retrains``.
        """
        X, y = _check_chunk(X, y)
        estimator = _final_estimator(self.model)
        before = getattr(estimator, "n_retrains_", 0)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        elapsed = time.perf_counter() - start
        after = getattr(estimator, "n_retrains_", 0)
        return {"fit_time_s": elapsed,
                "memory_bytes": estimate_memory(self.model) if self.log_memory else 0,
                "retrained": after > before,
                "n_retrains": after}

    def process_chunk(self, X, y):
        """Prequential step: ``score_chunk`` then ``fit_chunk``; log and return the record.

        The record contains: ``chunk`` (1-based), ``n_samples``, ``total_samples``,
        everything returned by ``score_chunk`` and ``fit_chunk``.
        """
        X, y = _check_chunk(X, y)
        record = {"chunk": self.n_chunks_ + 1, "n_samples": int(X.shape[0])}
        self.n_chunks_ += 1
        self.n_samples_ += X.shape[0]
        record["total_samples"] = self.n_samples_
        record.update(self.score_chunk(X, y))
        record.update(self.fit_chunk(X, y))
        self.history.append(record)
        line = ("chunk %(chunk)d n=%(n_samples)d scored=%(scored)s "
                "acc=%(chunk_accuracy).3f fit=%(fit_time_s).4fs mem=%(memory_bytes)dB "
                "retrained=%(retrained)s" % record)
        self.logger.info(line)
        if self.verbose:
            print(line)
        return record

    def run(self, stream):
        """Process every ``(X_chunk, y_chunk)`` pair of an iterable; returns self."""
        for X, y in stream:
            self.process_chunk(X, y)
        return self

    # -- reading the log -------------------------------------------------------

    def history_arrays(self):
        """Return the log as a dict of NumPy arrays (one entry per chunk)."""
        if not self.history:
            return {}
        return {key: np.array([rec[key] for rec in self.history]) for key in self.history[0]}

    def metric_series(self, name):
        """Per-chunk values of a logged field (e.g. ``"accuracy"``, ``"chunk_accuracy"``)."""
        if not self.history:
            return np.array([])
        if name not in self.history[0]:
            raise KeyError(f"Unknown field {name!r}; available: {sorted(self.history[0])}.")
        return np.array([rec[name] for rec in self.history], dtype=float)

    def retrain_chunks(self):
        """1-based chunk numbers at which the model retrained."""
        return [rec["chunk"] for rec in self.history if rec["retrained"]]

    def summary(self):
        """Aggregate numbers of the run as a dict (scalar metrics only)."""
        arr = self.history_arrays()
        if not arr:
            return {"chunks": 0, "samples": 0}
        scored = arr["scored"].astype(bool)
        return {"chunks": self.n_chunks_, "samples": self.n_samples_,
                "scored_chunks": int(scored.sum()),
                "n_retrains": int(arr["n_retrains"][-1]),
                "total_fit_time_s": float(arr["fit_time_s"].sum()),
                "final_memory_bytes": int(arr["memory_bytes"][-1]),
                **{name: float(m.result()) for name, m in self._logged_metrics().items()}}
