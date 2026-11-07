# NumCompute-Stream benchmark results

Python 3.13.18, NumPy 2.5.3, Linux-6.18.44-fc-v300-x86_64-with-glibc2.39.
repeat=3, seed=42, quick=False. Median of repeated runs after one warm-up; speedup = loop / vectorised.

## (a) Loop vs vectorised

| Task | Size | Loop (ms) | Vectorised (ms) | Speedup |
|---|---:|---:|---:|---:|
| best split (6 features) | 500 | 55.789 | 0.505 | 110.5x |
| best split (6 features) | 1000 | 146.166 | 0.861 | 169.8x |
| best split (6 features) | 2000 | 430.362 | 2.005 | 214.6x |
| predict (depth<=8 tree) | 50000 | 224.691 | 5.988 | 37.5x |
| running mean/var (chunks of 100) | 200000 | 529.115 | 69.776 | 7.6x |

## (b) Effect of buffer_size (12000 samples, chunk=100, max_depth=8, concept drift at 50%)

| buffer_size | Prequential acc. | Acc. after drift | Scored fraction | Retrains | Mean retrain (ms) | Total retrain (ms) | Model memory (KB) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 0.8954 | 0.8903 | 0.992 | 120 | 2.16 | 259.6 | 4.5 |
| 250 | 0.8938 | 0.8767 | 0.975 | 40 | 5.63 | 225.1 | 7.1 |
| 500 | 0.8907 | 0.8662 | 0.958 | 24 | 8.79 | 211.1 | 9.5 |
| 1000 | 0.8645 | 0.8225 | 0.917 | 12 | 13.61 | 163.3 | 11.2 |
| 2000 | 0.8043 | 0.7190 | 0.833 | 6 | 26.29 | 157.8 | 11.7 |
| 4000 | 0.5846 | 0.4665 | 0.667 | 3 | 46.17 | 138.5 | 15.4 |

Prequential accuracy counts only samples predicted by an already trained tree (each chunk is scored before it is used for training).
