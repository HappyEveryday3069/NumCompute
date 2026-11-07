"""
Synthetic streaming datasets used by the demo, tests and benchmarks.
"""

import csv

import numpy as np


def make_drifting_data(n_samples=6000, n_features=6, drift_at=0.5, noise=0.05,
                       missing_rate=0.02, seed=0):
    """Generate a 3-class dataset whose decision rule changes part-way through.

    Before the drift point the label depends on features 0 and 1; afterwards it
    depends on features 2 and 3. A few entries are set to NaN and a fraction
    ``noise`` of labels is replaced by a random class.

    Args:
        n_samples: number of rows (>= 1).
        n_features: number of features (>= 4).
        drift_at: fraction in [0, 1] of the stream after which the concept changes.
        noise: label-noise probability in [0, 1].
        missing_rate: probability in [0, 1] that a feature value is NaN.
        seed: random seed (deterministic output).
    Returns:
        ``(X, y)``: float array ``(n_samples, n_features)`` and int array ``(n_samples,)``.
    Raises:
        ValueError: invalid arguments.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")
    if n_features < 4:
        raise ValueError("n_features must be >= 4.")
    if not (0.0 <= drift_at <= 1.0):
        raise ValueError("drift_at must be in [0, 1].")
    if not (0.0 <= noise <= 1.0) or not (0.0 <= missing_rate <= 1.0):
        raise ValueError("noise and missing_rate must be in [0, 1].")
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features))
    before = np.where(X[:, 0] < -0.5, 0, np.where(X[:, 1] < 0.3, 1, 2))
    after = np.where(X[:, 2] < 0.5, 2, np.where(X[:, 3] < -0.3, 0, 1))
    is_after = np.arange(n_samples) >= int(round(drift_at * n_samples))
    y = np.where(is_after, after, before).astype(int)
    flip = rng.random(n_samples) < noise
    y = np.where(flip, rng.integers(0, 3, size=n_samples), y)
    X = np.where(rng.random(X.shape) < missing_rate, np.nan, X)
    return X, y


def write_csv(path, X, y, header=True):
    """Write features and labels to a CSV file; NaN is written as an empty cell.

    Args:
        path: output file path.
        X: float array ``(n, f)``.
        y: label array ``(n,)``.
        header: write a header row ``f0,...,f{k-1},label``.
    Raises:
        ValueError: X is not 2D or lengths differ.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X must be 2D and have the same number of rows as y.")
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        if header:
            writer.writerow([f"f{i}" for i in range(X.shape[1])] + ["label"])
        for row, label in zip(X, y):
            writer.writerow(["" if np.isnan(v) else repr(float(v)) for v in row] + [int(label)])
