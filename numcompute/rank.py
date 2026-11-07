"""
Ranking and percentile operations for NumPy arrays.
"""

from typing import Union
import numpy as np


def compute_rank(
    data: np.ndarray,
    method: str = "average",
    axis: int = -1
) -> np.ndarray:
    data = np.asarray(data)
    
    # Ensure working on a 1D slice or vectorized format
    if data.ndim > 1:
        return np.apply_along_axis(compute_rank, axis, data, method=method)

    n = len(data)
    if n == 0:
        return np.array([], dtype=float)

    # Obtain sorted indices
    sorter = np.argsort(data, kind="mergesort")
    inv_sorter = np.empty(n, dtype=int)
    inv_sorter[sorter] = np.arange(n)

    sorted_data = data[sorter]

    if method == "ordinal":
        # Ordinal ranks are simply 1 to N based on sorted order
        return (inv_sorter + 1).astype(float)

    # Identify where values change in the sorted array
    obs = np.pad(sorted_data[1:] != sorted_data[:-1], (1, 0), constant_values=True)

    if method == "dense":
        # Cumulative sum of distinct observations gives dense rank
        dense_ranks = np.cumsum(obs)
        return dense_ranks[inv_sorter].astype(float)

    elif method == "average":
        # Find indices where unique blocks start
        obs_idx = np.flatnonzero(obs)
        # Compute count of repeated elements
        counts = np.diff(np.append(obs_idx, n))
        
        # Calculate 1-based start and end indices for average
        start_ranks = obs_idx + 1
        end_ranks = obs_idx + counts
        avg_ranks = (start_ranks + end_ranks) / 2.0
        
        # Repeat the average ranks for each element in the group
        group_ranks = np.repeat(avg_ranks, counts)
        return group_ranks[inv_sorter]

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Choose from 'average', 'dense', 'ordinal'."
        )


def percentile(
    data: np.ndarray,
    q: Union[float, np.ndarray],
    interpolation: str = "linear",
    axis: Union[int, None] = None,
    keepdims: bool = False
) -> Union[float, np.ndarray]:
    
    q_arr = np.asarray(q)
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("Percentile q must be between 0 and 100 inclusive.")

    # Delegate to NumPy's optimized implementation
    return np.percentile(
        data, q=q, method=interpolation, axis=axis, keepdims=keepdims
    )