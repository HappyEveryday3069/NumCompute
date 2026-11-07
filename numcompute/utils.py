"""
Vectorised distances, activations, stable reductions and batching.
"""

import numpy as np
from operator import index


def euclidean_distance(a, b, axis=-1):
    """Compute Euclidean distances with scaled squares.

    Args: a, b are broadcastable real arrays; axis selects coordinates.
    Returns: distances with axis reduced; NaNs propagate, empty norms are 0.
    Raises: ValueError for incompatible shapes; NumPy axis/conversion errors.
    Cost: O(N) time/space for N broadcast elements.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.abs(a - b)
    scale = np.max(delta, axis=axis, keepdims=True, initial=0.0)
    divisor = np.where(np.isfinite(scale) & (scale != 0), scale, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        out = divisor * np.sqrt(np.sum((delta / divisor) ** 2, axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def sigmoid(x):
    """Compute a numerically stable elementwise sigmoid.

    Args: x is a real scalar or array.
    Returns: float array of the same shape; +/-inf map to 1/0, NaNs propagate.
    Raises: TypeError/ValueError on invalid numeric conversion.
    Cost: O(N) time/space.
    """
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def relu(x):
    """Replace negative entries with zero.

    Args: x is a real scalar or array.
    Returns: float array of the same shape; NaNs propagate.
    Raises: TypeError/ValueError on invalid numeric conversion.
    Cost: O(N) time/space.
    """
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0)


def logsumexp(x, axis=None, keepdims=False):
    """Compute log(sum(exp(x))) without direct exponential overflow.

    Args: x is real; axis is None/int/tuple; keepdims retains reduced axes.
    Returns: reduced float array/scalar; empty/all -inf sums return -inf;
        +inf is supported and NaNs propagate.
    Raises: NumPy axis errors or TypeError/ValueError on invalid conversion.
    Cost: O(N) time, O(N) space including possible input conversion.
    """
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        out = np.logaddexp.reduce(x, axis=axis, keepdims=keepdims, initial=-np.inf)
    return float(out) if np.ndim(out) == 0 else out


def softmax(x, axis=-1):
    """Normalise exponentials using a maximum shift.

    Args: x is finite real data; axis selects the normalisation axes.
    Returns: float array with x's shape, summing to 1 over axis.
    Raises: ValueError for non-finite input or empty reduction axes;
        invalid axes/conversions raise NumPy errors.
    Cost: O(N) time/space.
    """
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("softmax requires finite values.")
    if axis is None:
        empty_axis = x.size == 0
    elif isinstance(axis, tuple):
        empty_axis = any(x.shape[a] == 0 for a in axis)
    elif x.ndim == 0:
        empty_axis = False
    else:
        empty_axis = x.shape[axis] == 0
    if empty_axis:
        raise ValueError("softmax cannot reduce an empty axis.")
    with np.errstate(over="ignore", under="ignore"):
        shifted = x - np.max(x, axis=axis, keepdims=True)
        exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def topk_indices(values, k, largest=True):
    """Select sorted extrema with deterministic ties.

    Args: values is real (n,) without NaNs; integer k is in [1,n];
        largest=True selects maxima, otherwise minima.
    Returns: original indices (k,); ties favour smaller original indices.
    Raises: TypeError for noninteger k; ValueError for invalid data/range.
    Cost: O(n+k*log(k)) typical time, O(n+k) space.
    """
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "biuf":
        raise ValueError("values must be a 1D real numeric array.")
    if np.isnan(values).any():
        raise ValueError("values must not contain NaN.")
    if isinstance(k, (bool, np.bool_)):
        raise TypeError("k must be an integer.")
    k = index(k)
    if k < 1 or k > values.size:
        raise ValueError("k must be between 1 and len(values).")
    position = values.size - k if largest else k - 1
    partition = np.argpartition(values, position)
    boundary = values[partition[position]]
    better = np.flatnonzero(values > boundary if largest else values < boundary)
    tied = np.flatnonzero(values == boundary)[:k - better.size]
    idx = np.sort(np.concatenate((better, tied)))
    if largest:
        idx = idx[::-1]
        return idx[np.argsort(values[idx], kind="stable")[::-1]]
    return idx[np.argsort(values[idx], kind="stable")]


def batched(iterable, batch_size):
    """Yield consecutive slices, including the final short batch.

    Args: iterable is sized and sliceable; batch_size is a positive integer.
    Returns: iterator of slices of length <= batch_size; empty input yields none.
    Raises: TypeError for noninteger size; ValueError for nonpositive size.
    Cost: O(b) dispatch for b batches; array slices are views, list slices copy
        up to batch_size elements. Slicing cost depends on the input type.
    """
    if isinstance(batch_size, (bool, np.bool_)):
        raise TypeError("batch_size must be an integer.")
    batch_size = index(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]
