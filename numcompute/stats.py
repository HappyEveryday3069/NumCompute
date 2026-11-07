"""
Descriptive statistics, histograms, and quantiles with NaN handling.
"""

from typing import Dict, Tuple, Union
import numpy as np


def compute_basic_stats(
    data: np.ndarray,
    axis: Union[int, None] = None,
    keepdims: bool = False,
    ignore_nan: bool = True
) -> Dict[str, Union[float, np.ndarray]]:
    data = np.asarray(data)

    if ignore_nan:
        mean_val = np.nanmean(data, axis=axis, keepdims=keepdims)
        median_val = np.nanmedian(data, axis=axis, keepdims=keepdims)
        std_val = np.nanstd(data, axis=axis, keepdims=keepdims)
        min_val = np.nanmin(data, axis=axis, keepdims=keepdims)
        max_val = np.nanmax(data, axis=axis, keepdims=keepdims)
    else:
        mean_val = np.mean(data, axis=axis, keepdims=keepdims)
        median_val = np.median(data, axis=axis, keepdims=keepdims)
        std_val = np.std(data, axis=axis, keepdims=keepdims)
        min_val = np.min(data, axis=axis, keepdims=keepdims)
        max_val = np.max(data, axis=axis, keepdims=keepdims)

    return {
        "mean": mean_val,
        "median": median_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
    }


def histogram(
    data: np.ndarray,
    bins: int = 10,
    range_bounds: Union[Tuple[float, float], None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if bins < 1:
        raise ValueError("Number of bins must be at least 1.")

    data_flat = np.asarray(data).ravel()
    # Mask NaNs
    data_flat = data_flat[~np.isnan(data_flat)]

    if len(data_flat) == 0:
        edges = np.linspace(0, 1, bins + 1)
        return np.zeros(bins, dtype=int), edges

    if range_bounds is None:
        min_val, max_val = float(np.min(data_flat)), float(np.max(data_flat))
        if min_val == max_val:
            min_val -= 0.5
            max_val += 0.5
    else:
        min_val, max_val = range_bounds
        if min_val >= max_val:
            raise ValueError("Lower bound must be strictly less than upper bound.")

    bin_edges = np.linspace(min_val, max_val, bins + 1)
    
    # Fully vectorized binning
    indices = np.digitize(data_flat, bin_edges) - 1
    # Handle maximum value landing exactly on the right-most edge
    indices = np.clip(indices, 0, bins - 1)
    counts = np.bincount(indices, minlength=bins)

    return counts, bin_edges


def quantiles(
    data: np.ndarray,
    q: Union[float, np.ndarray],
    axis: Union[int, None] = None,
    keepdims: bool = False,
    ignore_nan: bool = True
) -> Union[float, np.ndarray]:
    
    q_arr = np.asarray(q)
    if np.any((q_arr < 0.0) | (q_arr > 1.0)):
        raise ValueError("Quantiles q must be between 0.0 and 1.0 inclusive.")

    data = np.asarray(data)

    if ignore_nan:
        return np.nanquantile(data, q=q, axis=axis, keepdims=keepdims)
    return np.quantile(data, q=q, axis=axis, keepdims=keepdims)