"""
Benchmarks for NumCompute-Stream.

(a) loop-based vs vectorised implementations
      * best-split search of the decision tree (per-threshold loop vs cumulative counts)
      * prediction (per-sample Python loop vs level-wise vectorised traversal)
      * running mean/variance (per-row Welford loop vs chunked Chan merge)
(b) effect of ``buffer_size`` on a drifting stream
      prequential accuracy, number of retrains, retraining time, memory.

Usage (from the repository root):
    python benchmark/stream_benchmarks.py                  # full run
    python benchmark/stream_benchmarks.py --quick          # small sizes, a few seconds
    python benchmark/stream_benchmarks.py --repeat 5 --seed 1 --output-dir benchmark

Results are written to ``stream_results.json`` and ``stream_results.md`` in
``--output-dir``. Times are wall-clock seconds measured with ``time.perf_counter``
after one untimed warm-up call; speedups depend on hardware and load.
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from numcompute_stream.datasets import make_drifting_data  # noqa: E402
from numcompute_stream.stats import RunningStats  # noqa: E402
from numcompute_stream.stream import estimate_memory  # noqa: E402
from numcompute_stream.tree import (DecisionTreeClassifier, best_split_loop,  # noqa: E402
                                    best_split_vectorised)


def time_call(fn, repeat):
    """Median/mean/min/max seconds of ``repeat`` calls after one warm-up; returns (stats, last result)."""
    result = fn()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return {"median": float(np.median(times)), "mean": float(np.mean(times)),
            "min": float(np.min(times)), "max": float(np.max(times)), "times": times}, result


def _comparison(name, size, loop_fn, vec_fn, repeat, equal):
    loop_stats, loop_out = time_call(loop_fn, repeat)
    vec_stats, vec_out = time_call(vec_fn, repeat)
    if not equal(loop_out, vec_out):
        raise AssertionError(f"{name}: loop and vectorised results differ")
    return {"task": name, "size": size, "loop_median_s": loop_stats["median"],
            "vectorised_median_s": vec_stats["median"],
            "speedup": loop_stats["median"] / vec_stats["median"],
            "loop_times": loop_stats["times"], "vectorised_times": vec_stats["times"]}


def welford_loop(X):
    """Reference per-row Welford update of mean and population variance."""
    n, mean, m2 = 0, np.zeros(X.shape[1]), np.zeros(X.shape[1])
    for row in X:
        n += 1
        delta = row - mean
        mean += delta / n
        m2 += delta * (row - mean)
    return mean, m2 / n


def chunked_stats(X, chunk=100):
    rs = RunningStats()
    for start in range(0, len(X), chunk):
        rs.update_stats(X[start:start + chunk])
    return rs.mean, rs.variance(0)


def bench_loop_vs_vectorised(quick, repeat, seed):
    sizes_split = [200, 500] if quick else [500, 1000, 2000]
    rows = []
    for n in sizes_split:
        X, y = make_drifting_data(n, missing_rate=0, seed=seed)
        Y = np.eye(3)[y]
        rows.append(_comparison(
            "best split (6 features)", n,
            lambda: best_split_loop(X, Y, range(X.shape[1]), "gini", 1),
            lambda: best_split_vectorised(X, Y, range(X.shape[1]), "gini", 1),
            repeat, lambda a, b: a == b))
    n_train, n_pred = (1000, 5000) if quick else (3000, 50000)
    Xtr, ytr = make_drifting_data(n_train, seed=seed)
    tree = DecisionTreeClassifier(max_depth=8).fit(Xtr, ytr)
    Xp, _ = make_drifting_data(n_pred, seed=seed + 1)
    rows.append(_comparison("predict (depth<=8 tree)", n_pred,
                            lambda: tree.predict_loop(Xp), lambda: tree.predict(Xp),
                            repeat, np.array_equal))
    n_stat = 20000 if quick else 200000
    Xs = np.random.default_rng(seed).normal(size=(n_stat, 6))
    rows.append(_comparison("running mean/var (chunks of 100)", n_stat,
                            lambda: welford_loop(Xs), lambda: chunked_stats(Xs),
                            repeat,
                            lambda a, b: np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])))
    return rows


def run_stream(X, y, buffer_size, chunk_size, max_depth, drift_index):
    """Prequential run of a buffered tree; returns accuracy and timing details."""
    tree = DecisionTreeClassifier(max_depth=max_depth, buffer_size=buffer_size)
    hits = total = hits_after = total_after = 0
    fit_time = 0.0
    for start in range(0, len(X), chunk_size):
        xb, yb = X[start:start + chunk_size], y[start:start + chunk_size]
        if tree.is_fitted:
            ok = int((tree.predict(xb) == yb).sum())
            hits, total = hits + ok, total + len(yb)
            if start >= drift_index:
                hits_after, total_after = hits_after + ok, total_after + len(yb)
        before = tree.n_retrains_
        t0 = time.perf_counter()
        tree.partial_fit(xb, yb)
        if tree.n_retrains_ > before:
            fit_time += time.perf_counter() - t0
    return {"accuracy": hits / total if total else float("nan"),
            "accuracy_after_drift": hits_after / total_after if total_after else float("nan"),
            "scored_fraction": total / len(X), "n_retrains": tree.n_retrains_,
            "retrain_time_total_s": fit_time,
            "retrain_time_mean_s": fit_time / tree.n_retrains_ if tree.n_retrains_ else float("nan"),
            "memory_bytes": estimate_memory(tree)}


def bench_buffer_sizes(quick, repeat, seed):
    n = 4000 if quick else 12000
    drift_at, chunk, depth = 0.5, 100, 8
    X, y = make_drifting_data(n, drift_at=drift_at, seed=seed)
    sizes = [100, 500, 2000] if quick else [100, 250, 500, 1000, 2000, 4000]
    rows = []
    for size in sizes:
        runs = [run_stream(X, y, size, chunk, depth, int(drift_at * n)) for _ in range(repeat)]
        best = dict(runs[0])  # accuracy is deterministic; take median timings
        best["retrain_time_total_s"] = float(np.median([r["retrain_time_total_s"] for r in runs]))
        best["retrain_time_mean_s"] = float(np.median([r["retrain_time_mean_s"] for r in runs]))
        best["buffer_size"] = size
        rows.append(best)
    return {"n_samples": n, "chunk_size": chunk, "max_depth": depth, "drift_at": drift_at,
            "rows": rows}


def to_markdown(report):
    env, s = report["environment"], report["settings"]
    lines = ["# NumCompute-Stream benchmark results", "",
             f"Python {env['python']}, NumPy {env['numpy']}, {env['platform']}.",
             f"repeat={s['repeat']}, seed={s['seed']}, quick={s['quick']}. "
             "Median of repeated runs after one warm-up; speedup = loop / vectorised.", "",
             "## (a) Loop vs vectorised", "",
             "| Task | Size | Loop (ms) | Vectorised (ms) | Speedup |", "|---|---:|---:|---:|---:|"]
    for r in report["loop_vs_vectorised"]:
        lines.append(f"| {r['task']} | {r['size']} | {r['loop_median_s'] * 1e3:.3f} | "
                     f"{r['vectorised_median_s'] * 1e3:.3f} | {r['speedup']:.1f}x |")
    b = report["buffer_sizes"]
    lines += ["", f"## (b) Effect of buffer_size ({b['n_samples']} samples, chunk={b['chunk_size']}, "
              f"max_depth={b['max_depth']}, concept drift at {int(b['drift_at'] * 100)}%)", "",
              "| buffer_size | Prequential acc. | Acc. after drift | Scored fraction | Retrains | "
              "Mean retrain (ms) | Total retrain (ms) | Model memory (KB) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in b["rows"]:
        lines.append(f"| {r['buffer_size']} | {r['accuracy']:.4f} | {r['accuracy_after_drift']:.4f} | "
                     f"{r['scored_fraction']:.3f} | {r['n_retrains']} | "
                     f"{r['retrain_time_mean_s'] * 1e3:.2f} | {r['retrain_time_total_s'] * 1e3:.1f} | "
                     f"{r['memory_bytes'] / 1024:.1f} |")
    lines += ["", "Prequential accuracy counts only samples predicted by an already trained tree "
              "(each chunk is scored before it is used for training).", ""]
    return "\n".join(lines)


def run(quick=False, repeat=3, seed=42):
    return {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "platform": platform.platform()},
            "settings": {"quick": quick, "repeat": repeat, "seed": seed},
            "loop_vs_vectorised": bench_loop_vs_vectorised(quick, repeat, seed),
            "buffer_sizes": bench_buffer_sizes(quick, repeat, seed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--quick", action="store_true", help="use small workloads")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    report = run(args.quick, args.repeat, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stream_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    (args.output_dir / "stream_results.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return report


if __name__ == "__main__":
    main()
