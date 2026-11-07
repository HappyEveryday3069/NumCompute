"""
Streaming-capable preprocessing: scalers, imputer and one-hot encoder.

Every transformer offers ``fit`` (train from scratch), ``partial_fit``
(incremental update from one chunk), ``transform`` and ``fit_transform``.
All inputs are 2D arrays of shape ``(n_samples, n_features)``; the number of
features is fixed by the first chunk (except for ``OneHotEncoder``, whose
*output* width grows as new categories appear).
"""

import numpy as np

from .stats import RunningStats
from .utils import NotFittedError, check_X


def _check_n_features(X, expected, name):
    if X.shape[1] != expected:
        raise ValueError(
            f"{name} was fitted with {expected} features but X has {X.shape[1]}."
        )


class StandardScaler:
    """Standardise features: ``z = (x - mean) / std``, with streaming updates.

    ``partial_fit`` maintains running mean/variance per feature through
    ``RunningStats`` (Chan/Welford merge), so fitting chunk by chunk gives the
    same parameters as fitting all data at once. NaNs are ignored when fitting
    and propagate through ``transform``. Zero-variance features get scale 1.

    Note:
        The parameters change with every ``partial_fit``. Do not place this scaler in
        front of a *buffered* estimator (e.g. ``DecisionTreeClassifier(buffer_size=...)``):
        chunks of one buffer would be scaled differently and the model's thresholds would
        go stale between retrains. Tree models do not need scaling anyway.

    Args:
        with_mean: subtract the mean.
        with_std: divide by the standard deviation.

    Attributes:
        mean_, var_, scale_: arrays ``(n_features,)``.
        n_samples_seen_: per-feature number of non-NaN observations.
    """

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._stats = RunningStats()
        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    def _refresh(self):
        s = self._stats
        mean = np.nan_to_num(s.mean, nan=0.0)
        var = np.nan_to_num(s.variance(0), nan=0.0)
        std = np.sqrt(var)
        tiny = 10 * np.finfo(float).eps * np.maximum(1.0, np.abs(mean))
        self.mean_ = mean
        self.var_ = var
        self.scale_ = np.where(std > tiny, std, 1.0)
        self.n_samples_seen_ = s.count_.copy()

    def partial_fit(self, X, y=None):
        """Update the running statistics with one chunk; returns self."""
        X = check_X(X, "X")
        self._stats.update_stats(X)
        self._refresh()
        return self

    def fit(self, X, y=None):
        """Fit from scratch on X (discarding earlier state); returns self."""
        self._stats = RunningStats()
        return self.partial_fit(X)

    def _check_fitted(self):
        if self.mean_ is None:
            raise NotFittedError("StandardScaler must be fitted before transform.")

    def transform(self, X):
        """Standardise X using the current parameters."""
        self._check_fitted()
        X = check_X(X, "X")
        _check_n_features(X, self.mean_.shape[0], "StandardScaler")
        out = X
        if self.with_mean:
            out = out - self.mean_
        if self.with_std:
            out = out / self.scale_
        return out

    def inverse_transform(self, X):
        """Undo ``transform``."""
        self._check_fitted()
        X = check_X(X, "X")
        _check_n_features(X, self.mean_.shape[0], "StandardScaler")
        out = X
        if self.with_std:
            out = out * self.scale_
        if self.with_mean:
            out = out + self.mean_
        return out

    def fit_transform(self, X, y=None):
        """``fit`` then ``transform`` the same data."""
        return self.fit(X).transform(X)


class MinMaxScaler:
    """Scale each feature to ``feature_range`` using running min/max.

    Constant features map to the lower bound of the range. Values outside the
    range seen so far are extrapolated (not clipped).

    Args:
        feature_range: ``(low, high)`` with low < high.
    Raises:
        ValueError: low >= high.
    """

    def __init__(self, feature_range=(0, 1)):
        low, high = feature_range
        if low >= high:
            raise ValueError("min_value must be less than max_value")
        self.feature_range = (low, high)
        self._stats = RunningStats(max_abs=None)  # only min/max are used, no variance overflow
        self.data_min_ = None
        self.data_max_ = None

    def partial_fit(self, X, y=None):
        """Update running min/max with one chunk; returns self."""
        X = check_X(X, "X")
        self._stats.update_stats(X)
        self.data_min_ = np.nan_to_num(self._stats.min, nan=0.0)
        self.data_max_ = np.nan_to_num(self._stats.max, nan=1.0)
        return self

    def fit(self, X, y=None):
        """Fit from scratch on X; returns self."""
        self._stats = RunningStats(max_abs=None)
        return self.partial_fit(X)

    def transform(self, X):
        """Scale X to the target range."""
        if self.data_min_ is None:
            raise NotFittedError("MinMaxScaler must be fitted before transform.")
        X = check_X(X, "X")
        _check_n_features(X, self.data_min_.shape[0], "MinMaxScaler")
        span = self.data_max_ - self.data_min_
        span = np.where(span == 0, 1.0, span)
        low, high = self.feature_range
        return (X - self.data_min_) / span * (high - low) + low

    def fit_transform(self, X, y=None):
        """``fit`` then ``transform`` the same data."""
        return self.fit(X).transform(X)


class SimpleImputer:
    """Replace NaNs, updating the fill statistics on the fly.

    Strategies:
        ``"constant"``: always ``fill_value``.
        ``"mean"``: running per-feature mean of the observed values.
        ``"most_frequent"``: running per-feature mode (ties -> smallest value).
    A feature that has not yet had any observed value is filled with ``fill_value``.

    Note:
        ``"mean"`` and ``"most_frequent"`` are adaptive (the fill values change with every
        chunk); in front of a buffered estimator prefer ``"constant"`` (see ``StandardScaler``).

    Args:
        strategy: one of the strategies above.
        fill_value: constant / fallback value.
    Raises:
        ValueError: unknown strategy.
    """

    _STRATEGIES = ("constant", "mean", "most_frequent")

    def __init__(self, strategy="constant", fill_value=0.0):
        if strategy not in self._STRATEGIES:
            raise ValueError(
                f"Strategy must be one of {self._STRATEGIES}; got {strategy!r}."
            )
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None
        self._stats = RunningStats()
        self._freq = None

    def partial_fit(self, X, y=None):
        """Update fill statistics with one chunk; returns self."""
        X = check_X(X, "X")
        n_features = X.shape[1]
        if self.statistics_ is not None:
            _check_n_features(X, self.statistics_.shape[0], "SimpleImputer")
        if self.strategy == "mean":
            self._stats.update_stats(X)
            self.statistics_ = np.where(self._stats.count_ > 0,
                                        np.nan_to_num(self._stats.mean),
                                        float(self.fill_value))
        elif self.strategy == "most_frequent":
            if self._freq is None:
                self._freq = [dict() for _ in range(n_features)]
            for j in range(n_features):  # per-feature dictionaries: loop is inherent
                col = X[:, j]
                values, counts = np.unique(col[~np.isnan(col)], return_counts=True)
                table = self._freq[j]
                for v, c in zip(values.tolist(), counts.tolist()):
                    table[v] = table.get(v, 0) + c
            self.statistics_ = np.array([
                min(t, key=lambda v: (-t[v], v)) if t else float(self.fill_value)
                for t in self._freq
            ], dtype=float)
        else:
            self.statistics_ = np.full(n_features, float(self.fill_value))
        return self

    def fit(self, X, y=None):
        """Fit from scratch on X; returns self."""
        self._stats = RunningStats()
        self._freq = None
        self.statistics_ = None
        return self.partial_fit(X)

    def transform(self, X):
        """Return a copy of X with NaNs replaced by the current statistics."""
        if self.statistics_ is None:
            raise NotFittedError("SimpleImputer must be fitted before transform.")
        X = check_X(X, "X")
        _check_n_features(X, self.statistics_.shape[0], "SimpleImputer")
        return np.where(np.isnan(X), self.statistics_, X)

    def fit_transform(self, X, y=None):
        """``fit`` then ``transform`` the same data."""
        return self.fit(X).transform(X)


Imputer = SimpleImputer  # name used in the assignment specification


class OneHotEncoder:
    """One-hot encode categorical columns; new categories expand the output.

    Output layout: columns are numbered in the global order in which categories were first
    seen (within one chunk: feature by feature, categories sorted). A category that appears
    later is **appended as a new last column**, so no existing output column ever moves,
    even for inputs with several features. This matters for models trained on earlier
    chunks, whose splits refer to column indices (see ``DecisionTreeClassifier(
    allow_feature_growth=True)``).

    Args:
        handle_unknown: ``"ignore"`` encodes an unseen category as all zeros,
            ``"error"`` raises ``ValueError``.
    Attributes:
        categories_: list with the known categories of every input feature (object arrays).
        n_output_features_: current width of the encoded output.
    Raises:
        ValueError: invalid ``handle_unknown``; missing values in the data;
            input that is not 2D.
    """

    def __init__(self, handle_unknown="ignore"):
        if handle_unknown not in ("ignore", "error"):
            raise ValueError("handle_unknown must be 'ignore' or 'error'.")
        self.handle_unknown = handle_unknown
        self.categories_ = None
        self._columns = None   # per feature: output column index of each known category
        self._width = 0

    @staticmethod
    def _check(X):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError(
                f"X must be 2D (n_samples, n_features); got shape {X.shape}."
            )
        missing = (X != X) | (X == None)  # noqa: E711  (elementwise on object arrays)
        if np.any(missing):
            raise ValueError("OneHotEncoder does not accept missing values; impute first.")
        return X

    def _check_width(self, X):
        if X.shape[1] != len(self.categories_):
            raise ValueError(
                f"OneHotEncoder was fitted with {len(self.categories_)} features "
                f"but X has {X.shape[1]}."
            )

    def partial_fit(self, X, y=None):
        """Register categories found in the chunk; returns self."""
        X = self._check(X)
        if self.categories_ is None:
            self.categories_ = [np.array([], dtype=object) for _ in range(X.shape[1])]
            self._columns = [np.array([], dtype=np.int64) for _ in range(X.shape[1])]
            self._width = 0
        else:
            self._check_width(X)
        if X.shape[0] == 0:
            return self
        for j in range(X.shape[1]):  # one category list per column
            try:
                found = np.unique(X[:, j])
            except TypeError as exc:
                raise ValueError(f"Column {j} mixes uncomparable category types.") from exc
            known = self.categories_[j]
            new = found[~np.isin(found, known)] if known.size else found
            if new.size:
                self.categories_[j] = np.concatenate([known, new.astype(object)])
                added = np.arange(self._width, self._width + new.size, dtype=np.int64)
                self._columns[j] = np.concatenate([self._columns[j], added])
                self._width += new.size
        return self

    def fit(self, X, y=None):
        """Fit from scratch on X; returns self."""
        self.categories_ = None
        return self.partial_fit(X)

    @property
    def n_output_features_(self):
        """Current width of the encoded output."""
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder must be fitted first.")
        return int(self._width)

    def transform(self, X):
        """Encode X; shape ``(n_samples, n_output_features_)`` with 0/1 floats."""
        if self.categories_ is None:
            raise NotFittedError("OneHotEncoder must be fitted before transform.")
        X = self._check(X)
        self._check_width(X)
        out = np.zeros((X.shape[0], self._width))
        for j, cats in enumerate(self.categories_):
            block = (X[:, j:j + 1] == cats[None, :])
            if self.handle_unknown == "error" and block.shape[0] and not block.any(axis=1).all():
                bad = X[~block.any(axis=1), j][0]
                raise ValueError(f"Unknown category {bad!r} in column {j}.")
            out[:, self._columns[j]] = block
        return out

    def fit_transform(self, X, y=None):
        """``fit`` then ``transform`` the same data."""
        return self.fit(X).transform(X)
