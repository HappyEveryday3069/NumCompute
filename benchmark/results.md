# Benchmark Results

Python: 3.10.11; NumPy: 2.2.6.
Platform: Windows-10-10.0.26200-SP0.
dtype: float64; repeat: 5; seed: 42; warm-up: 1.
Thread settings: {'OMP_NUM_THREADS': 'unset', 'OPENBLAS_NUM_THREADS': 'unset', 'MKL_NUM_THREADS': 'unset'}.

## Sum: 100000 values

| Implementation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| python_sum | 4.6493 | 4.5091 | 4.3068 | 5.0381 |
| numpy_sum | 0.0308 | 0.0268 | 0.0170 | 0.0581 |

Speedup (mean time): 150.85x

## Distances: 10000 rows x 8 features

| Implementation | Mean (ms) | Median (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|---:|
| python_distances | 16.4816 | 16.5377 | 15.6805 | 17.1069 |
| numpy_distances | 0.8639 | 0.8375 | 0.7375 | 0.9725 |

Speedup (mean time): 19.08x

Times include function calls and output allocation; input creation is excluded.
Correctness checks run before timing. Speedups depend on size, hardware and load.
Raw samples and NumPy build information are stored in the matching JSON file.
