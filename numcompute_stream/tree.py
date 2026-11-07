"""
Decision tree classifier built from scratch with buffered streaming retraining.

Design summary
--------------
* Binary splits on numeric features: a sample goes left if ``x[f] <= threshold``.
* Impurity: Gini or entropy (bits). Split quality = impurity decrease
  (information gain) of the weighted child impurities.
* The best threshold of a feature is found in one vectorised pass: sort the
  feature, take cumulative one-hot class counts, and evaluate the impurity of
  *every* candidate split at once. A loop-based reference implementation
  (``best_split_loop``) is kept for tests and benchmarks.
* The tree is stored in flat arrays (no node objects) and prediction advances
  all samples one tree level at a time (vectorised).
* Streaming: ``partial_fit`` appends chunks to a buffer; once the buffer holds
  at least ``buffer_size`` samples the tree is retrained on the buffered data
  and the buffer is cleared. Between retrains ``predict`` uses the most
  recently trained tree.

Tie handling and determinism
----------------------------
* Thresholds are only placed between *distinct* consecutive feature values and
  are the midpoint of the two values.
* Among thresholds / features whose gain is equal within 1e-12 the first one
  wins (smallest feature index, then smallest threshold).
* A leaf predicts the majority class; ties go to the smallest class label.

Design decision: zero-gain splits
---------------------------------
The course material (``_best_feature``) starts with ``best_gain = 0.0`` and only accepts
a feature whose information gain is *strictly greater* than zero, so a node whose every
split has gain 0 becomes a leaf. In XOR-like data each single feature has gain exactly 0
although two splits together separate the classes; the course rule stops at the root and
predicts the majority class. This implementation deliberately differs by default
(``allow_zero_gain_splits=True``): when ``min_impurity_decrease`` is 0 the best split is
taken even if its gain is 0 (it never makes the tree worse on the training data, and
``max_depth`` / ``min_samples_*`` still bound the growth). Set
``allow_zero_gain_splits=False`` for the strict course behaviour, or use
``min_impurity_decrease > 0``, which always requires a positive gain.

NaNs and infinities
-------------------
NaNs in X are replaced by the per-feature median of the training data (stored
as ``nan_fill_`` and also applied at prediction time). Infinite values and missing (NaN/None)
labels are rejected with ``ValueError``.
"""

import time
import warnings

import numpy as np

from .utils import NotFittedError, check_X, check_y

_TOL = 1e-12  # gains closer than this are treated as ties


# ---------------------------------------------------------------------------
# Impurity and split search
# ---------------------------------------------------------------------------

def impurity_from_counts(counts, criterion="gini"):
    """Impurity of class-count vectors.

    Args:
        counts: array ``(..., K)`` of non-negative class counts.
        criterion: ``"gini"`` (1 - sum p^2) or ``"entropy"`` (-sum p log2 p).
    Returns:
        array with the last axis reduced; 0 for empty groups.
    Raises:
        ValueError: unknown criterion.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=-1, keepdims=True)
    p = counts / np.where(n > 0, n, 1.0)
    if criterion == "gini":
        out = 1.0 - (p ** 2).sum(axis=-1)
    elif criterion == "entropy":
        out = -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=-1)
    else:
        raise ValueError("criterion must be 'gini' or 'entropy'.")
    return np.maximum(out, 0.0)


def _midpoint(a, b):
    """Split threshold between two consecutive distinct values ``a < b``.

    Computed as ``a / 2 + b / 2``: unlike ``a + (b - a) / 2`` it cannot overflow when ``a`` and
    ``b`` are huge finite numbers of opposite sign (``b - a`` would become ``inf``). Rounding can
    push the result onto ``b`` (adjacent floats) or, for subnormals, below ``a``; in that case
    ``a`` is returned, so ``a <= threshold < b`` always holds and ``x <= threshold`` still
    separates the two values.
    """
    thr = a / 2.0 + b / 2.0
    return thr if a <= thr < b else a


def best_split_vectorised(Xn, Yn, features, criterion="gini", min_samples_leaf=1):
    """Find the best split among ``features`` with vectorised threshold evaluation.

    Args:
        Xn: float array ``(n, F)`` of the node's samples (no NaN).
        Yn: one-hot labels ``(n, K)``.
        features: iterable of feature indices to consider (ascending).
        criterion: ``"gini"`` or ``"entropy"``.
        min_samples_leaf: minimum samples on each side.
    Returns:
        ``(feature, threshold, gain)`` or None if no valid split exists.
    Cost: O(|features| * (n log n + n K)).
    """
    n = Xn.shape[0]
    total = Yn.sum(axis=0)
    parent = float(impurity_from_counts(total, criterion))
    n_left = np.arange(1, n, dtype=float)
    n_right = n - n_left
    size_ok = (n_left >= min_samples_leaf) & (n_right >= min_samples_leaf)
    best = None
    for f in features:
        x = Xn[:, f]
        order = np.argsort(x, kind="stable")
        xs = x[order]
        left = np.cumsum(Yn[order], axis=0)[:-1]
        valid = (xs[1:] > xs[:-1]) & size_ok
        if not valid.any():
            continue
        weighted = (n_left * impurity_from_counts(left, criterion)
                    + n_right * impurity_from_counts(total - left, criterion)) / n
        gain = np.where(valid, parent - weighted, -np.inf)
        i = int(np.flatnonzero(gain >= gain.max() - _TOL)[0])
        if best is None or gain[i] > best[2] + _TOL:
            best = (int(f), _midpoint(xs[i], xs[i + 1]), float(gain[i]))
    return best


def best_split_loop(Xn, Yn, features, criterion="gini", min_samples_leaf=1):
    """Loop-based reference of ``best_split_vectorised`` (same result, O(n^2) per feature).

    Evaluates every candidate threshold separately with a Python loop. Used to
    validate the vectorised search and as the baseline in the benchmarks.
    """
    n = Xn.shape[0]
    total = Yn.sum(axis=0)
    parent = float(impurity_from_counts(total, criterion))
    best = None
    for f in features:
        x = Xn[:, f]
        values = np.unique(x)
        gains, thresholds = [], []
        for a, b in zip(values[:-1], values[1:]):
            thr = _midpoint(a, b)
            mask = x <= thr
            n_l = int(mask.sum())
            if n_l < min_samples_leaf or n - n_l < min_samples_leaf:
                continue
            left = Yn[mask].sum(axis=0)
            weighted = (n_l * impurity_from_counts(left, criterion)
                        + (n - n_l) * impurity_from_counts(total - left, criterion)) / n
            gains.append(parent - weighted)
            thresholds.append(thr)
        if not gains:
            continue
        gains = np.array(gains)
        i = int(np.flatnonzero(gains >= gains.max() - _TOL)[0])
        if best is None or gains[i] > best[2] + _TOL:
            best = (int(f), thresholds[i], float(gains[i]))
    return best


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _is_real(value):
    """A real number that is neither a bool nor NaN/inf."""
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_)) and np.isfinite(value))


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


class DecisionTreeClassifier:
    """Depth-limited decision tree with Gini/entropy splits and buffered streaming.

    Args:
        criterion: ``"gini"`` or ``"entropy"``.
        max_depth: maximum depth (root = 0); None for unlimited. 0 gives a single leaf.
        min_samples_split: minimum samples a node needs to be split (>= 2).
        min_samples_leaf: minimum samples in each child (>= 1).
        max_features: features examined per split: None (all), int, float
            fraction in (0, 1], ``"sqrt"`` or ``"log2"``. If no valid split is
            found among the sampled features the remaining ones are tried.
        buffer_size: ``partial_fit`` retrains once at least this many samples
            are buffered (>= 1).
        min_impurity_decrease: finite number >= 0; a split is made only if its gain is at
            least this.
        allow_zero_gain_splits: design decision (see module docstring). True (default)
            accepts a best split whose gain is 0 when ``min_impurity_decrease`` is 0, so
            interaction patterns such as XOR can be learned. False follows the course
            ``_best_feature`` rule: split only if the gain is strictly positive.
        random_state: seed for feature sub-sampling.
        allow_feature_growth: if True the number of columns may grow between
            chunks (e.g. after a streaming ``OneHotEncoder``); narrower chunks
            are zero-padded, extra unseen columns are ignored at prediction.

    Attributes (after training):
        classes_, n_features_, feature_, threshold_, children_left_,
        children_right_, value_ (class counts per node), n_node_samples_,
        node_depth_, n_retrains_, n_samples_trained_, last_fit_time_.

    Raises:
        ValueError: invalid hyper-parameters or data (see each method).
    """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features=None, buffer_size=500,
                 min_impurity_decrease=0.0, random_state=None,
                 allow_feature_growth=False, allow_zero_gain_splits=True):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'.")
        if max_depth is not None and (not _is_int(max_depth) or max_depth < 0):
            raise ValueError("max_depth must be None or a non-negative integer.")
        if not _is_int(min_samples_split) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        if not _is_int(min_samples_leaf) or min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1.")
        if not _is_int(buffer_size) or buffer_size < 1:
            raise ValueError("buffer_size must be an integer >= 1.")
        if not _is_real(min_impurity_decrease) or min_impurity_decrease < 0:
            raise ValueError("min_impurity_decrease must be a finite number >= 0 (not a bool).")
        self._validate_max_features(max_features)
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = int(min_samples_split)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.buffer_size = int(buffer_size)
        self.min_impurity_decrease = float(min_impurity_decrease)
        self.random_state = random_state
        self.allow_feature_growth = bool(allow_feature_growth)
        self.allow_zero_gain_splits = bool(allow_zero_gain_splits)
        self._reset_state()

    # -- parameter helpers -------------------------------------------------

    @staticmethod
    def _validate_max_features(mf):
        ok = (mf is None or mf in ("sqrt", "log2")
              or (_is_int(mf) and mf >= 1)
              or (isinstance(mf, float) and 0.0 < mf <= 1.0))
        if not ok:
            raise ValueError(
                "max_features must be None, 'sqrt', 'log2', an int >= 1 or a float in (0, 1]."
            )

    def get_params(self):
        """Return the constructor hyper-parameters as a dict."""
        return {"criterion": self.criterion, "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features, "buffer_size": self.buffer_size,
                "min_impurity_decrease": self.min_impurity_decrease,
                "random_state": self.random_state,
                "allow_feature_growth": self.allow_feature_growth,
                "allow_zero_gain_splits": self.allow_zero_gain_splits}

    def _n_candidate_features(self, n_features):
        mf = self.max_features
        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))
        return min(int(mf), n_features)

    # -- state -------------------------------------------------------------

    def _reset_state(self):
        self.classes_ = None
        self.n_features_ = None          # width the current tree was trained on
        self.n_features_in_ = None       # width expected from incoming data
        self.feature_ = None
        self.threshold_ = None
        self.children_left_ = None
        self.children_right_ = None
        self.value_ = None
        self.n_node_samples_ = None
        self.node_depth_ = None
        self.nan_fill_ = None
        self.n_retrains_ = 0
        self.n_samples_trained_ = 0
        self.last_fit_time_ = 0.0
        self._importance = None
        self._buf_X = []
        self._buf_y = []
        self._buf_n = 0

    @property
    def is_fitted(self):
        """True once a tree has been trained."""
        return self.feature_ is not None

    @property
    def n_buffered_samples(self):
        """Number of samples waiting in the buffer for the next retrain."""
        return self._buf_n

    # -- input handling ----------------------------------------------------

    def _check_chunk(self, X, y):
        X = check_X(X, "X")
        y = check_y(y, X.shape[0])
        if X.shape[0] == 0:
            return X, y
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]
        elif X.shape[1] != self.n_features_in_:
            if self.allow_feature_growth and X.shape[1] > self.n_features_in_:
                self.n_features_in_ = X.shape[1]
            elif not self.allow_feature_growth:
                raise ValueError(
                    f"X has {X.shape[1]} features, but the tree expects {self.n_features_in_}."
                )
        return X, y

    @staticmethod
    def _align(X, width):
        """Zero-pad or truncate columns to ``width``."""
        if X.shape[1] < width:
            return np.hstack([X, np.zeros((X.shape[0], width - X.shape[1]))])
        return X[:, :width]

    # -- training ----------------------------------------------------------

    def fit(self, X, y):
        """Train a new tree from scratch on (X, y); clears the streaming buffer.

        Args:
            X: ``(n, f)`` float array (NaN allowed, inf not).
            y: ``(n,)`` labels (numbers or strings).
        Returns:
            self.
        Raises:
            ValueError: empty data, shape/length mismatch, NaN labels, inf in X.
        """
        self._reset_state()
        X, y = self._check_chunk(X, y)
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a tree on zero samples.")
        start = time.perf_counter()
        self._build(X, y)
        self.last_fit_time_ = time.perf_counter() - start
        self.n_retrains_ = 1
        return self

    def partial_fit(self, X, y):
        """Add a chunk to the buffer; retrain when ``buffer_size`` is reached.

        The new tree is trained on the buffered samples only (not on earlier
        chunks) and the buffer is then cleared. An empty chunk is a no-op.

        Returns:
            self.
        Raises:
            ValueError: shape/length mismatch, NaN labels, inf in X, or a
                feature-count change (unless ``allow_feature_growth``).
        """
        X, y = self._check_chunk(X, y)
        if X.shape[0] == 0:
            return self
        self._buf_X.append(X)
        self._buf_y.append(y)
        self._buf_n += X.shape[0]
        if self._buf_n >= self.buffer_size:
            self.flush()
        return self

    def flush(self):
        """Retrain immediately on whatever is buffered (no-op if the buffer is empty)."""
        if self._buf_n == 0:
            return self
        width = max(x.shape[1] for x in self._buf_X)
        X = np.vstack([self._align(x, width) for x in self._buf_X])
        y = np.concatenate(self._buf_y)
        start = time.perf_counter()
        self._build(X, y)
        self.last_fit_time_ = time.perf_counter() - start
        self.n_retrains_ += 1
        self._buf_X, self._buf_y, self._buf_n = [], [], 0
        return self

    def _build(self, X, y):
        """Grow a tree on validated arrays and store it in flat arrays."""
        n, n_features = X.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns
            # The median of an even count averages the two middle values, which overflows for
            # values near the float maximum; halving first keeps every intermediate finite.
            fill = 2.0 * np.nanmedian(X / 2.0, axis=0)
        fill = np.where(np.isnan(fill), 0.0, fill)
        X = np.where(np.isnan(X), fill, X)

        classes, y_idx = np.unique(y, return_inverse=True)
        y_idx = y_idx.ravel()
        k = classes.shape[0]
        Y = np.zeros((n, k))
        Y[np.arange(n), y_idx] = 1.0

        rng = np.random.default_rng(self.random_state)
        n_cand = self._n_candidate_features(n_features)
        all_features = np.arange(n_features)

        feature, threshold, left, right = [], [], [], []
        value, depth_of, gains = [], [], []
        stack = [(np.arange(n), 0, -1, True)]
        while stack:
            idx, depth, parent, is_left = stack.pop()
            node = len(feature)
            counts = Y[idx].sum(axis=0)
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(counts)
            depth_of.append(depth)
            gains.append(0.0)
            if parent >= 0:
                (left if is_left else right)[parent] = node

            size = idx.size
            if ((self.max_depth is not None and depth >= self.max_depth)
                    or size < self.min_samples_split
                    or size < 2 * self.min_samples_leaf
                    or counts.max() == size):
                continue
            Xn, Yn = X[idx], Y[idx]
            if n_cand >= n_features:
                candidates = all_features
            else:
                candidates = np.sort(rng.choice(n_features, n_cand, replace=False))
            best = best_split_vectorised(Xn, Yn, candidates, self.criterion,
                                         self.min_samples_leaf)
            if best is None and n_cand < n_features:
                rest = np.setdiff1d(all_features, candidates)
                best = best_split_vectorised(Xn, Yn, rest, self.criterion,
                                             self.min_samples_leaf)
            if best is None or best[2] + _TOL < self.min_impurity_decrease:
                continue
            if not self.allow_zero_gain_splits and best[2] <= _TOL:
                continue  # course rule: the gain must be strictly positive
            f, thr, gain = best
            feature[node], threshold[node], gains[node] = f, thr, gain
            go_left = Xn[:, f] <= thr
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        self.classes_ = classes
        self.n_features_ = n_features
        self.feature_ = np.array(feature, dtype=np.int64)
        self.threshold_ = np.array(threshold, dtype=float)
        self.children_left_ = np.array(left, dtype=np.int64)
        self.children_right_ = np.array(right, dtype=np.int64)
        self.value_ = np.array(value)
        self.n_node_samples_ = self.value_.sum(axis=1).astype(np.int64)
        self.node_depth_ = np.array(depth_of, dtype=np.int64)
        self.nan_fill_ = fill
        self.n_samples_trained_ = n
        weighted = np.array(gains) * self.n_node_samples_ / n
        importance = np.bincount(self.feature_[self.feature_ >= 0],
                                 weights=weighted[self.feature_ >= 0],
                                 minlength=n_features)
        self._importance = importance

    # -- prediction --------------------------------------------------------

    def _check_fitted(self):
        if self.feature_ is None:
            raise NotFittedError(
                "The tree has not been trained yet: call fit(), or call "
                "partial_fit() until the buffer reaches buffer_size (or flush())."
            )

    def _prepare(self, X):
        X = check_X(X, "X")
        if X.shape[1] != self.n_features_:
            if not self.allow_feature_growth:
                raise ValueError(
                    f"X has {X.shape[1]} features, but the tree was trained on {self.n_features_}."
                )
            X = self._align(X, self.n_features_)
        return np.where(np.isnan(X), self.nan_fill_, X)

    def _leaf_ids(self, X):
        """Leaf index of every row, advancing all rows one level per iteration."""
        node = np.zeros(X.shape[0], dtype=np.int64)
        while True:
            f = self.feature_[node]
            rows = np.flatnonzero(f >= 0)
            if rows.size == 0:
                return node
            cur = node[rows]
            go_left = X[rows, f[rows]] <= self.threshold_[cur]
            node[rows] = np.where(go_left, self.children_left_[cur], self.children_right_[cur])

    def predict_proba(self, X):
        """Class probabilities from the leaf class counts, shape ``(n, n_classes)``.

        Raises:
            NotFittedError: no tree trained yet; ValueError: wrong feature count.
        """
        self._check_fitted()
        X = self._prepare(X)
        counts = self.value_[self._leaf_ids(X)]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predict class labels, shape ``(n,)``, using the most recently trained tree.

        Raises:
            NotFittedError: no tree trained yet; ValueError: wrong feature count.
        """
        self._check_fitted()
        X = self._prepare(X)
        leaf = self._leaf_ids(X)
        return self.classes_[np.argmax(self.value_[leaf], axis=1)]

    def predict_loop(self, X):
        """Loop-based per-sample prediction (reference for tests and benchmarks)."""
        self._check_fitted()
        X = self._prepare(X)
        out = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            node = 0
            while self.feature_[node] >= 0:
                if X[i, self.feature_[node]] <= self.threshold_[node]:
                    node = self.children_left_[node]
                else:
                    node = self.children_right_[node]
            out[i] = self.classes_[int(np.argmax(self.value_[node]))]
        return out

    def score(self, X, y):
        """Accuracy of ``predict(X)`` against y."""
        pred = self.predict(X)
        y = check_y(y, pred.shape[0])
        return float(np.mean(pred == y)) if y.size else float("nan")

    # -- inspection --------------------------------------------------------

    def get_depth(self):
        """Depth of the trained tree (a single leaf has depth 0)."""
        self._check_fitted()
        return int(self.node_depth_.max())

    def get_n_leaves(self):
        """Number of leaves of the trained tree."""
        self._check_fitted()
        return int((self.feature_ < 0).sum())

    @property
    def feature_importances_(self):
        """Normalised total impurity decrease per feature (sums to 1, or all 0 for a stump)."""
        self._check_fitted()
        total = self._importance.sum()
        return self._importance / total if total > 0 else np.zeros_like(self._importance)

    def export_text(self, feature_names=None):
        """Return a readable text rendering of the trained tree."""
        self._check_fitted()
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_)]
        if len(feature_names) != self.n_features_:
            raise ValueError(f"feature_names must have {self.n_features_} entries.")
        lines = []
        stack = [("node", 0, 0)]
        while stack:
            item = stack.pop()
            if item[0] == "line":
                lines.append(item[1])
                continue
            _, node, depth = item
            pad = "|   " * depth
            if self.feature_[node] < 0:
                label = self.classes_[int(np.argmax(self.value_[node]))]
                lines.append(f"{pad}class: {label} (n={self.n_node_samples_[node]})")
                continue
            name, thr = feature_names[self.feature_[node]], self.threshold_[node]
            stack.append(("node", self.children_right_[node], depth + 1))
            stack.append(("line", f"{pad}{name} > {thr:.4g}"))
            stack.append(("node", self.children_left_[node], depth + 1))
            stack.append(("line", f"{pad}{name} <= {thr:.4g}"))
        return "\n".join(lines)
