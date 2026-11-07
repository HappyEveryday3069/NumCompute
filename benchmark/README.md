# Benchmarks

Run from the repository root:

```powershell
python benchmark/run_benchmarks.py
```

This entry point calls `numcompute.benchmarking` without changing its
implementations. It verifies agreement, performs one untimed warm-up, and
compares Python loops with NumPy for sums and Euclidean distances.

The default run writes both files from the same measurement:

- `benchmark/results.json`: raw timings, settings and environment metadata.
- `benchmark/results.md`: formatted tables generated from that JSON.

The stored results were measured using Python 3.10.11 and NumPy 2.2.6.
A later run overwrites the files in this directory with new measurements.

Custom workloads use the existing benchmark options:

```powershell
python benchmark/run_benchmarks.py --n 100000 --distance-n 10000 --features 8 --repeat 7 --seed 42
```

To preserve the current results, choose another output name:

```powershell
python benchmark/run_benchmarks.py --output benchmark/results_new.json
```

The matching summary is saved as `benchmark/results_new.md`. Use JSON and
Markdown from the same run in the report. Timing values are seconds in JSON
and milliseconds in the table; speedups use unrounded mean times. Input
construction is excluded, while function calls and output allocation are
included. No claim is made that vectorisation is faster for every workload.

The original command remains available:

```powershell
python -m numcompute.benchmarking --output benchmark/results.json
```

That original command writes JSON only; use `run_benchmarks.py` when you also
want the matching Markdown summary regenerated.
