import numpy as np


def stable_sort(values, axis=-1):
    values = np.asarray(values)

    return np.sort(values, axis=axis, kind="stable")


def multi_key_sort(X, keys):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    if len(keys) == 0:
        return X.copy()

    sort_indices = np.lexsort([X[:, k] for k in reversed(keys)])
    return X[sort_indices]


def topk(values, k, largest=True, return_indices=True):
    values = np.asarray(values)

    if not isinstance(k, int):
        raise TypeError("k must be an integer")

    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")

    if k == len(values):
        idx = np.arange(len(values))
    elif largest:
        idx = np.argpartition(values, -k)[-k:]
    else:
        idx = np.argpartition(values, k - 1)[:k]

    order = np.argsort(values[idx])

    if largest:
        order = order[::-1]

    idx = idx[order]
    top_vals = values[idx]

    return (top_vals, idx) if return_indices else top_vals


def quickselect(values, k):
    values = np.asarray(values)

    if k <= 0 or k > len(values):
        raise ValueError("k must be between 1 and len(values)")

    return np.partition(values, k - 1)[k - 1]


def binary_search(sorted_array, x):
    arr = np.asarray(sorted_array)

    if arr.ndim != 1:
        raise ValueError("sorted_array must be a 1D array")

    idx = np.searchsorted(arr, x)
    found = (idx < len(arr)) and (arr[idx] == x)

    return idx, bool(found)
