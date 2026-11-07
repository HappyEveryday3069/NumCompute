"""
Streaming statistics: running moments, histograms and quantiles.

Batch helpers (``compute_basic_stats``, ``histogram``, ``quantiles``) are kept
for reference; the streaming classes below consume data chunk by chunk through
``update_stats(X_chunk)`` and never need the full dataset in memory (the
quantile estimator keeps a bounded sample).
"""

from collections import deque
from operator import index
import warnings

import numpy as np

from .utils import check_X


# ---------------------------------------------------------------------------
# Batch helpers (reference implementations)
# ---------------------------------------------------------------------------

def compute_basic_stats(data, axis=None, keepdims=False, ignore_nan=True):
    """Mean, median, std, min and max of an array (NaN-aware by default).

    Args:
        data: array-like.
        axis: axis to reduce (None = all).
        keepdims: keep reduced axes with size 1.
        ignore_nan: use the ``nan*`` reductions when True.
    Returns:
        dict with keys ``mean, median, std, min, max``.
    """
    data = np.asarray(data)
    if ignore_nan:
        funcs = (np.nanmean, np.nanmedian, np.nanstd, np.nanmin, np.nanmax)
    else:
        funcs = (np.mean, np.median, np.std, np.min, np.max)
    names = ("mean", "median", "std", "min", "max")
    return {n: f(data, axis=axis, keepdims=keepdims) for n, f in zip(names, funcs)}


def histogram(data, bins=10, range_bounds=None):
    """Vectorised histogram that ignores NaNs and clips outliers into edge bins.

    Args:
        data: array-like (flattened).
        bins: number of equal-width bins (>= 1).
        range_bounds: optional ``(low, high)`` with low < high.
    Returns:
        ``(counts, bin_edges)`` with shapes ``(bins,)`` and ``(bins + 1,)``.
    Raises:
        ValueError: bins < 1 or invalid bounds.
    """
    if bins < 1:
        raise ValueError("Number of bins must be at least 1.")
    flat = np.asarray(data, dtype=float).ravel()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return np.zeros(bins, dtype=int), np.linspace(0, 1, bins + 1)
    if range_bounds is None:
        low, high = float(flat.min()), float(flat.max())
        if low == high:
            low, high = low - 0.5, high + 0.5
    else:
        low, high = range_bounds
        if low >= high:
            raise ValueError("Lower bound must be strictly less than upper bound.")
    edges = np.linspace(low, high, bins + 1)
    idx = np.clip(np.digitize(flat, edges) - 1, 0, bins - 1)
    return np.bincount(idx, minlength=bins), edges


def quantiles(data, q, axis=None, keepdims=False, ignore_nan=True):
    """Quantiles with ``q`` in [0, 1].

    Raises:
        ValueError: q outside [0, 1].
    """
    q_arr = np.asarray(q)
    if np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("Quantiles q must be between 0.0 and 1.0 inclusive.")
    data = np.asarray(data)
    fn = np.nanquantile if ignore_nan else np.quantile
    return fn(data, q=q, axis=axis, keepdims=keepdims)


# ---------------------------------------------------------------------------
# Streaming statistics
# ---------------------------------------------------------------------------

class RunningStats:
    """Per-feature running count, mean, variance, min and max (NaN-aware).

    Chunks are merged with the parallel (Chan et al.) update, a batch
    generalisation of Welford's algorithm, which is numerically stable and
    exactly reproduces the full-batch result up to rounding:

        delta = mean_b - mean_a
        mean  = mean_a + delta * n_b / n
        M2    = M2_a + M2_b + delta**2 * n_a * n_b / n

    NaNs are ignored feature by feature (each feature has its own count).
    Features that have never been observed report NaN statistics.

    The variance squares deviations, so magnitudes beyond about 1e154 overflow. Rather than
    silently returning inf/NaN, ``update_stats`` raises ``ValueError`` for ``|x| > max_abs``.

    Args:
        max_abs: largest accepted ``|x|`` (default 1e100); None disables the check
            (use it when only min/max are needed).

    Attributes:
        n_features_: number of features (None until the first update).
        count_: per-feature number of non-NaN observations, shape ``(f,)``.
        n_samples_seen_: total number of rows seen.
        n_chunks_: number of non-empty chunks consumed.
    """

    def __init__(self, max_abs=1e100):
        if max_abs is not None and not (np.isfinite(max_abs) and max_abs > 0):
            raise ValueError("max_abs must be None or a positive finite number.")
        self.max_abs = max_abs
        self.reset()

    def reset(self):
        """Forget all data and the learned number of features."""
        self.n_features_ = None
        self.count_ = None
        self.n_samples_seen_ = 0
        self.n_chunks_ = 0
        self._mean = None
        self._m2 = None
        self._min = None
        self._max = None
        return self

    def _init_state(self, n_features):
        self.n_features_ = n_features
        self.count_ = np.zeros(n_features)
        self._mean = np.zeros(n_features)
        self._m2 = np.zeros(n_features)
        self._min = np.full(n_features, np.inf)
        self._max = np.full(n_features, -np.inf)

    def update_stats(self, X_chunk):
        """Merge a chunk of shape ``(n, f)`` (or 1D = one feature) into the state.

        Raises:
            ValueError: non-numeric / infinite input, ``|x| > max_abs``, or a feature-count change.
        """
        X = check_X(X_chunk, "X_chunk", allow_1d=True)
        if self.max_abs is not None and X.size and np.fmax.reduce(np.abs(X), axis=None) > self.max_abs:
            raise ValueError(
                f"Values with |x| > {self.max_abs:g} are not supported: the running variance "
                "would overflow. Rescale the data (or create RunningStats(max_abs=None))."
            )
        if self.n_features_ is None:
            self._init_state(X.shape[1])
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, expected {self.n_features_}."
            )
        if X.shape[0] == 0:
            return self
        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0).astype(float)
        safe_n_b = np.where(n_b > 0, n_b, 1.0)
        n_a = self.count_
        n = n_a + n_b
        safe_n = np.where(n > 0, n, 1.0)
        # With max_abs=None huge values may overflow the variance (min/max stay exact);
        # that must not emit warnings, so the moment arithmetic runs under errstate.
        with np.errstate(over="ignore", invalid="ignore"):
            mean_b = np.where(valid, X, 0.0).sum(axis=0) / safe_n_b
            m2_b = (np.where(valid, X - mean_b, 0.0) ** 2).sum(axis=0)
            delta = mean_b - self._mean
            self._mean = self._mean + delta * (n_b / safe_n)
            self._m2 = self._m2 + m2_b + delta ** 2 * (n_a * n_b / safe_n)
        self.count_ = n
        self._min = np.fmin(self._min, np.fmin.reduce(X, axis=0, initial=np.inf))
        self._max = np.fmax(self._max, np.fmax.reduce(X, axis=0, initial=-np.inf))
        self.n_samples_seen_ += X.shape[0]
        self.n_chunks_ += 1
        return self

    update = update_stats  # alias: ``update`` is the generic streaming verb

    def _require_data(self):
        if self.n_features_ is None:
            raise ValueError("RunningStats has not seen any data yet.")

    @property
    def mean(self):
        """Per-feature mean, shape ``(f,)``; NaN where no value was observed."""
        self._require_data()
        return np.where(self.count_ > 0, self._mean, np.nan)

    def variance(self, ddof=0):
        """Per-feature variance with ``ddof`` delta degrees of freedom.

        NaN where ``count - ddof <= 0``.
        """
        self._require_data()
        denom = self.count_ - ddof
        safe = np.where(denom > 0, denom, 1.0)
        return np.where(denom > 0, np.maximum(self._m2, 0.0) / safe, np.nan)

    @property
    def std(self):
        """Per-feature population standard deviation (ddof = 0)."""
        return np.sqrt(self.variance(0))

    @property
    def min(self):
        """Per-feature minimum; NaN where no value was observed."""
        self._require_data()
        return np.where(self.count_ > 0, self._min, np.nan)

    @property
    def max(self):
        """Per-feature maximum; NaN where no value was observed."""
        self._require_data()
        return np.where(self.count_ > 0, self._max, np.nan)

    def merge(self, other):
        """Merge another ``RunningStats`` (e.g. from a parallel worker) into this one.

        Raises:
            TypeError / ValueError: other is not RunningStats or has another width.
        """
        if not isinstance(other, RunningStats):
            raise TypeError("other must be a RunningStats instance.")
        if other.n_features_ is None:
            return self
        if self.n_features_ is None:
            self._init_state(other.n_features_)
        elif self.n_features_ != other.n_features_:
            raise ValueError("Cannot merge statistics with different feature counts.")
        n = self.count_ + other.count_
        safe_n = np.where(n > 0, n, 1.0)
        delta = other._mean - self._mean
        self._mean = self._mean + delta * (other.count_ / safe_n)
        self._m2 = (self._m2 + other._m2
                    + delta ** 2 * (self.count_ * other.count_ / safe_n))
        self.count_ = n
        self._min = np.fmin(self._min, other._min)
        self._max = np.fmax(self._max, other._max)
        self.n_samples_seen_ += other.n_samples_seen_
        self.n_chunks_ += other.n_chunks_
        return self

    def summary(self):
        """Return a dict with count, mean, std, min and max arrays."""
        return {"count": self.count_.copy(), "mean": self.mean, "std": self.std,
                "min": self.min, "max": self.max}


class StreamingHistogram:
    """Fixed-edge histogram updated chunk by chunk, optionally over a sliding window.

    Values outside the range are clipped into the first/last bin (as in
    ``histogram``); NaNs are ignored. If ``range_bounds`` is None the range is
    inferred from the first chunk that contains finite values (and then fixed).

    Args:
        bins: number of equal-width bins (>= 1).
        range_bounds: optional ``(low, high)`` with low < high.
        window: None for a cumulative histogram, else the number of most
            recent (non-empty) chunks the histogram covers.
    """

    def __init__(self, bins=10, range_bounds=None, window=None):
        if isinstance(bins, (bool, np.bool_)) or index(bins) < 1:
            raise ValueError("bins must be a positive integer.")
        if window is not None and (isinstance(window, (bool, np.bool_)) or index(window) < 1):
            raise ValueError("window must be None or a positive integer.")
        if range_bounds is not None:
            low, high = range_bounds
            if not (np.isfinite(low) and np.isfinite(high) and low < high):
                raise ValueError("range_bounds must be finite with low < high.")
        self.bins = index(bins)
        self.window = None if window is None else index(window)
        self.bin_edges_ = (None if range_bounds is None else
                           np.linspace(float(range_bounds[0]), float(range_bounds[1]),
                                       self.bins + 1))
        self._counts = np.zeros(self.bins, dtype=np.int64)
        self._recent = deque()

    def update_stats(self, X_chunk):
        """Add the values in ``X_chunk`` (any shape, flattened) to the histogram."""
        x = np.asarray(X_chunk, dtype=float).ravel()
        x = x[~np.isnan(x)]
        if x.size == 0:
            return self
        if self.bin_edges_ is None:
            finite = x[np.isfinite(x)]
            if finite.size == 0:
                return self  # cannot infer a range yet
            low, high = float(finite.min()), float(finite.max())
            if low == high:
                low, high = low - 0.5, high + 0.5
            self.bin_edges_ = np.linspace(low, high, self.bins + 1)
        idx = np.clip(np.digitize(x, self.bin_edges_) - 1, 0, self.bins - 1)
        counts = np.bincount(idx, minlength=self.bins).astype(np.int64)
        self._counts += counts
        if self.window is not None:
            self._recent.append(counts)
            while len(self._recent) > self.window:
                self._counts -= self._recent.popleft()
        return self

    update = update_stats

    def reset(self):
        """Clear counts (keeps the bin edges once they are known)."""
        self._counts = np.zeros(self.bins, dtype=np.int64)
        self._recent.clear()
        return self

    @property
    def counts(self):
        """Current bin counts, shape ``(bins,)``."""
        return self._counts.copy()

    @property
    def n_samples(self):
        """Number of values currently counted (within the window if one is set)."""
        return int(self._counts.sum())

    def density(self):
        """Counts normalised to sum to 1 (all zeros if empty)."""
        total = self._counts.sum()
        return self._counts / total if total > 0 else np.zeros(self.bins)

    def result(self):
        """Return ``(counts, bin_edges)``; edges are None until a range is known."""
        return self.counts, (None if self.bin_edges_ is None else self.bin_edges_.copy())


class StreamingQuantiles:
    """Approximate (or exact) per-feature quantiles over a stream.

    * ``window`` set: exact quantiles over the most recent ``window`` rows.
    * ``window`` None: a uniform reservoir sample of at most ``max_samples``
      rows (exact while fewer rows than ``max_samples`` have been seen).

    Rows are stored whole; NaNs are ignored per feature when computing quantiles.

    Args:
        window: None or positive int (rows).
        max_samples: reservoir capacity (positive int).
        random_state: seed for the reservoir sampling.
    """

    def __init__(self, window=None, max_samples=100_000, random_state=None):
        if window is not None and (isinstance(window, (bool, np.bool_)) or index(window) < 1):
            raise ValueError("window must be None or a positive integer.")
        if isinstance(max_samples, (bool, np.bool_)) or index(max_samples) < 1:
            raise ValueError("max_samples must be a positive integer.")
        self.window = None if window is None else index(window)
        self.max_samples = index(max_samples)
        self._seed = random_state
        self.reset()

    def reset(self):
        """Forget all stored rows."""
        self._rng = np.random.default_rng(self._seed)
        self._buf = None
        self.n_features_ = None
        self.n_samples_seen_ = 0
        return self

    def update_stats(self, X_chunk):
        """Add a chunk ``(n, f)`` (or 1D = one feature)."""
        X = check_X(X_chunk, "X_chunk", allow_1d=True)
        if self.n_features_ is None:
            self.n_features_ = X.shape[1]
            self._buf = np.empty((0, X.shape[1]))
        elif X.shape[1] != self.n_features_:
            raise ValueError(
                f"X_chunk has {X.shape[1]} features, expected {self.n_features_}."
            )
        m = X.shape[0]
        if m == 0:
            return self
        if self.window is not None:
            self._buf = np.concatenate([self._buf, X])[-self.window:]
            self.n_samples_seen_ += m
            return self
        free = self.max_samples - self._buf.shape[0]
        take = min(m, max(free, 0))
        if take > 0:
            self._buf = np.concatenate([self._buf, X[:take]])
            self.n_samples_seen_ += take
            X = X[take:]
        rest = X.shape[0]
        if rest > 0:
            # Algorithm R, vectorised: row with 0-based stream index t replaces a
            # random slot with probability max_samples / (t + 1). For repeated
            # slots the last assignment wins, matching sequential processing.
            t = self.n_samples_seen_ + np.arange(rest)
            slot = (self._rng.random(rest) * (t + 1)).astype(np.int64)
            keep = slot < self.max_samples
            self._buf[slot[keep]] = X[keep]
            self.n_samples_seen_ += rest
        return self

    update = update_stats

    @property
    def n_stored(self):
        """Number of rows currently stored."""
        return 0 if self._buf is None else self._buf.shape[0]

    def quantile(self, q):
        """Return quantile(s) ``q`` in [0, 1] per feature.

        Returns:
            shape ``(f,)`` for scalar q, ``(len(q), f)`` for a 1D q.
        Raises:
            ValueError: q outside [0, 1] or no data stored.
        """
        q_arr = np.asarray(q, dtype=float)
        if np.any((q_arr < 0.0) | (q_arr > 1.0)) or np.isnan(q_arr).any():
            raise ValueError("Quantiles q must be between 0.0 and 1.0 inclusive.")
        if self.n_stored == 0:
            raise ValueError("StreamingQuantiles has not seen any data yet.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns -> NaN
            return np.nanquantile(self._buf, q_arr, axis=0)
