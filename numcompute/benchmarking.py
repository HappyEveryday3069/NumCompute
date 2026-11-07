"""
Reproducible timing comparisons between Python loops and NumPy.
"""

import argparse
import contextlib
import io
import json
import os
import platform
import sys
import time
import warnings
from operator import index

import numpy as np


def _positive_integer(value, name):
    """Validate integer value > 0; return int or raise TypeError/ValueError. O(1) cost."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer.")
    value = index(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def benchmark(fn, *args, repeat=5, **kwargs):
    """Time repeat calls after one untimed warm-up.

    Args: fn is callable; args/kwargs are forwarded; repeat is a positive int.
        fn must accept repeated calls on the same inputs without harmful mutation.
    Returns: dict with result, times (repeat seconds), mean/min/max/median.
    Raises: TypeError/ValueError for invalid fn/repeat; fn errors propagate.
    Cost: O((repeat+1)*C_fn) time, O(repeat) timing storage plus fn output/workspace.
    """
    if not callable(fn):
        raise TypeError("fn must be callable.")
    repeat = _positive_integer(repeat, "repeat")
    fn(*args, **kwargs)
    times = []
    result = None
    for _ in range(repeat):
        start = time.perf_counter_ns()
        result = fn(*args, **kwargs)
        end = time.perf_counter_ns()
        times.append((end - start) / 1e9)
    return {
        "result": result,
        "times": times,
        "mean": float(np.mean(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "median": float(np.median(times)),
    }


def python_sum(x):
    """Sum a real vector using a Python loop.

    Args: x is a 1D numeric iterable; empty input is supported.
    Returns: scalar sum (0 for empty input).
    Raises: input iteration/arithmetic errors propagate.
    Cost: O(n) time, O(1) auxiliary space.
    """
    total = 0.0
    for v in x:
        total += v
    return total


def numpy_sum(x):
    """Sum numeric values with NumPy.

    Args: x is numeric array-like; all axes are reduced.
    Returns: scalar sum; NumPy determines dtype and empty-sum behaviour.
    Raises: NumPy conversion/arithmetic errors propagate.
    Cost: O(N) time, O(N) possible conversion space; O(1) for an existing array.
    """
    return np.sum(x)


def compare_sum(n=100000, repeat=5, seed=42):
    """Compare seeded float64 vector sums after verifying agreement.

    Args: n and repeat are positive integers; seed initialises a local RNG.
    Returns: dict with python_sum/numpy_sum timing records from benchmark.
    Raises: TypeError/ValueError for invalid settings; AssertionError on mismatch.
    Cost: O((repeat+1)*n) time, O(n+repeat) space.
    """
    n = _positive_integer(n, "n")
    repeat = _positive_integer(repeat, "repeat")
    rng = np.random.default_rng(seed)
    x = rng.random(n)
    np.testing.assert_allclose(python_sum(x), numpy_sum(x), rtol=1e-10, atol=1e-12)
    return {
        "python_sum": benchmark(python_sum, x, repeat=repeat),
        "numpy_sum": benchmark(numpy_sum, x, repeat=repeat),
    }


def python_distances(x, reference):
    """Calculate row-wise distances with nested Python loops.

    Args: x has shape (n,d); reference has length d, with moderate finite values.
    Returns: float array (n,).
    Raises: indexing/conversion/arithmetic errors propagate.
    Cost: O(n*d) time, O(n) output space.
    """
    distances = np.empty(len(x), dtype=float)
    for i, row in enumerate(x):
        total = 0.0
        for j in range(len(reference)):
            total += (row[j] - reference[j]) ** 2
        distances[i] = total ** 0.5
    return distances


def compare_distances(n=10000, repeat=5, seed=42, features=8):
    """Compare loop distances with the vectorised utility.

    Args: positive n/repeat/features; seed controls local float64 inputs (n,d).
    Returns: python_distances/numpy_distances timing records from benchmark.
    Raises: TypeError/ValueError for settings; AssertionError before timing on mismatch.
    Cost: O((repeat+1)*n*d) time, O(n*d+repeat) space, d=features.
    """
    from .utils import euclidean_distance

    n = _positive_integer(n, "n")
    repeat = _positive_integer(repeat, "repeat")
    features = _positive_integer(features, "features")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, features))
    reference = rng.normal(size=features)
    np.testing.assert_allclose(
        python_distances(x, reference), euclidean_distance(x, reference),
        rtol=1e-10, atol=1e-12,
    )
    return {
        "python_distances": benchmark(python_distances, x, reference, repeat=repeat),
        "numpy_distances": benchmark(euclidean_distance, x, reference, repeat=repeat),
    }


def environment_info():
    """Collect reproducibility metadata without changing thread settings.

    Args: none.
    Returns: dict of Python/NumPy/platform, timer, thread settings and build config.
    Raises: errors from platform or NumPy metadata retrieval propagate.
    Cost: O(B) time/space for B characters of environment/build metadata.
    """
    configuration = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(configuration):
        warnings.filterwarnings("ignore", message="Install `pyyaml` for better output")
        np.show_config()
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "timer": "perf_counter_ns",
        "time_unit": "seconds",
        "thread_settings": {
            name: os.environ.get(name, "unset")
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "numpy_configuration": configuration.getvalue(),
    }


def format_results(results):
    """Format timing records as a Markdown table in milliseconds.

    Args: results maps names to benchmark records; two rows also show mean speedup.
    Returns: str; a zero NumPy mean produces an unavailable speedup.
    Raises: KeyError/TypeError for malformed records.
    Cost: O(R) time/space for R rows, assuming fixed-size records.
    """
    lines = [
        "| Implementation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, timing in results.items():
        lines.append(
            f"| {name} | {timing['mean'] * 1000:.4f} | "
            f"{timing['median'] * 1000:.4f} | {timing['min'] * 1000:.4f} | "
            f"{timing['max'] * 1000:.4f} |"
        )
    timings = list(results.values())
    if len(timings) == 2:
        loop_time = timings[0]["mean"]
        numpy_time = timings[1]["mean"]
        speedup = f"{loop_time / numpy_time:.2f}x" if numpy_time > 0 else "unavailable"
        lines.extend(["", f"Speedup (mean time): {speedup}"])
    return "\n".join(lines)


def _serializable(results):
    """Copy timing records without result arrays for JSON. O(R) time/space for R records."""
    return {
        name: {key: value for key, value in timing.items() if key != "result"}
        for name, timing in results.items()
    }


def main():
    """Run command-line comparisons and optionally save JSON.

    Args: CLI --n/--distance-n/--features/--repeat/--seed and optional --output.
    Returns: None; prints tables and writes timing/environment metadata if requested.
    Raises: SystemExit for invalid CLI syntax; validation/assertion/I/O errors propagate.
    Cost: sum of both comparisons and report formatting/storage costs.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100000)
    parser.add_argument("--distance-n", type=int, default=10000)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    args = parser.parse_args()
    metadata = environment_info()
    sums = compare_sum(n=args.n, repeat=args.repeat, seed=args.seed)
    distances = compare_distances(
        n=args.distance_n, repeat=args.repeat, seed=args.seed, features=args.features,
    )
    print("Python:", metadata["python"])
    print("NumPy:", metadata["numpy"])
    print("Platform:", metadata["platform"])
    print("Thread settings:", metadata["thread_settings"])
    print(f"dtype: float64; repeat: {args.repeat}; seed: {args.seed}; warm-up: 1")
    print("Times include function calls and output allocation; input creation is excluded.")
    print("Correctness checks run before timing. Speedups depend on size, hardware and load.")
    print(f"\nSum: {args.n} values")
    print(format_results(sums))
    print(f"\nDistances: {args.distance_n} rows x {args.features} features")
    print(format_results(distances))
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "environment": metadata,
            "settings": {
                "n": args.n, "distance_n": args.distance_n, "features": args.features,
                "repeat": args.repeat, "seed": args.seed, "dtype": "float64",
                "warmup": 1, "input_creation_timed": False, "allocation_timed": True,
            },
            "sum": _serializable(sums),
            "distances": _serializable(distances),
        }
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print("\nSaved:", path)


if __name__ == "__main__":
    main()
