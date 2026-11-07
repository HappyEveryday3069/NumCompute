# NumCompute-Stream

A modularised, **decision tree-based streaming machine-learning framework** written with plain
Python, NumPy and matplotlib (no scikit-learn / pandas). It extends the group's NumCompute toolkit
(`numcompute/`, left untouched; its README is kept as `README_numcompute_2_1.md`).

Everything supports **chunk-wise streaming updates** through `partial_fit(...)` / `update(...)` /
`update_stats(...)`, plus a decision tree that is retrained from an internal **buffer**.

## Installation

```bash
git clone <your fork url>
cd NumCompute
git switch <your-individual-branch>   # the branch that contains numcompute_stream/ (use its real name)

# Option A: conda (Windows / macOS / Linux), then install the package
conda env create -f environment.yml
conda activate numcompute
python -m pip install -e .

# Option B: any Python >= 3.10 environment
python -m pip install -e .       # installs numpy and matplotlib from pyproject.toml
```

Python >= 3.10 and NumPy >= 2.0 are required (tested with Python 3.13 / NumPy 2.5).
For the notebook also install `ipykernel` (and Jupyter or VS Code with the notebook extension).

## Repository layout

```text
numcompute_stream/
    io.py             read_csv, read_csv_chunks (CSV input; NaN for missing cells)
    utils.py          input validation (check_X, check_y), iter_chunks, NotFittedError, group helpers
    datasets.py       synthetic drifting data generator + CSV writer (used by demo/tests/benchmarks)
    stats.py          RunningStats, StreamingHistogram, StreamingQuantiles (+ batch helpers)
    preprocessing.py  StandardScaler, MinMaxScaler, SimpleImputer (= Imputer), OneHotEncoder
    metrics.py        Streaming{Accuracy,Precision,Recall,F1,ConfusionMatrix,AUC,MSE} (+ batch metrics)
    tree.py           DecisionTreeClassifier with buffered retraining, split search (vectorised + loop)
    pipeline.py       Pipeline with fit / partial_fit / predict / predict_proba / score
    stream.py         StreamTrainer (prequential evaluation incl. probability metrics, logging, memory estimate)
    visualise.py      plot_metric_over_time, compare_models, plot_predictions_vs_ground_truth
tests/                167 streaming tests (test_stream_*.py, incl. test_stream_integration.py) + the group's tests
environment.yml       portable conda environment (python, numpy, matplotlib, notebook support)
demo/                 stream_demo.ipynb, data/stream_data.csv, make_dataset.py, figures/
benchmark/            stream_benchmarks.py, stream_results.md / .json (plus the group's benchmarks)
```

## Quick start

```python
from numcompute_stream import (Pipeline, Imputer, DecisionTreeClassifier,
                               StreamTrainer, iter_chunks)
from numcompute_stream.io import read_csv
from numcompute_stream.visualise import plot_metric_over_time

data = read_csv("demo/data/stream_data.csv", dtype=float, skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)

pipe = Pipeline([
    ("impute", Imputer(strategy="constant", fill_value=0.0)),   # fixed transformation, see design note
    ("model", DecisionTreeClassifier(max_depth=5, buffer_size=500)),
])

trainer = StreamTrainer(pipe)                       # test-then-train (prequential) evaluation
trainer.run(iter_chunks(X, y, chunk_size=100))      # one log record per chunk
plot_metric_over_time(trainer.metric_series("rolling_accuracy"), "Rolling accuracy", "accuracy",
                      retrain_points=trainer.retrain_chunks(), save_path="acc.png")
```

Without the trainer, the same pipeline is used directly:
`pipe.partial_fit(X_chunk, y_chunk)`, then `pipe.predict(X_new)`.

## How streaming works

| Component | Streaming behaviour |
|---|---|
| `DecisionTreeClassifier.partial_fit` | Appends the chunk to a buffer. When the buffer holds >= `buffer_size` samples the tree is **retrained on the buffered data only** and the buffer is cleared. Between retrains `predict` uses the latest tree; before the first retrain it raises `NotFittedError`. `flush()` forces a retrain; `fit()` trains from scratch. |
| `StandardScaler`, `MinMaxScaler` | Running mean/variance (Chan/Welford merge) and min/max; identical to the batch result up to rounding. NaNs ignored per feature; zero variance -> scale 1. |
| `SimpleImputer` | Running mean / most-frequent / constant fill values. |
| `OneHotEncoder` | A new category is appended as the **last output column** (global order of first appearance), so existing columns never move - not even those of later input features. Use it with `DecisionTreeClassifier(allow_feature_growth=True)`. |
| `RunningStats`, `StreamingHistogram`, `StreamingQuantiles` | `update_stats(X_chunk)`; histogram with fixed edges and optional sliding window of chunks; quantiles exact over a sliding window or via a bounded reservoir sample. |
| `Streaming*` metrics | `update(y_true_chunk, y_pred_chunk)`, `result()`, `reset()`; confusion matrix grows with new classes; optional rolling window (last N samples); AUC exact or histogram-approximate. |
| `StreamTrainer` | Test-then-train per chunk. Hard-label metrics use `predict`; `StreamingAUC` (with `pos_label`) is fed from `predict_proba`. Non-scalar metrics (confusion matrix) are updated but not logged. Works with string categories when a `OneHotEncoder` step precedes the model. |
| `Pipeline.partial_fit` | Each transformer: `partial_fit(X)` then `transform(X)`; final estimator: `partial_fit(X_t, y)`. |

### Design note: adaptive preprocessing in front of a buffered tree
`StandardScaler`, `MinMaxScaler` and `Imputer("mean" / "most_frequent")` are *adaptive*: their parameters
change with every chunk. A buffered tree only retrains after `buffer_size` samples, so with such a step
in front of it (1) the chunks stored in one buffer would have been transformed with different parameters
(different coordinate systems), and (2) between two retrains the tree's fixed thresholds would meet data
transformed with newer parameters. Decision trees are invariant to feature scaling, so the demo and the
quick start use a **fixed** transformation (`Imputer(strategy="constant")`), or the tree's own median NaN
handling could be used instead. The adaptive transformers remain fully supported on their own (and inside
a `Pipeline` for models that retrain on every chunk); `demo/stream_demo.ipynb` section 6 shows them standalone.

### Decision tree details
* Binary splits `x[f] <= threshold`; Gini or entropy; `max_depth`, `min_samples_split`,
  `min_samples_leaf`, `max_features` (int / fraction / `"sqrt"` / `"log2"`), `min_impurity_decrease`.
* **Vectorised split search**: per feature, sort once and use cumulative class counts to score every
  candidate threshold at once. `best_split_loop` is the loop-based reference (tests + benchmark).
* Flat-array tree; prediction advances all samples level by level (`predict_loop` is the reference).
* **Ties**: thresholds only between distinct values (midpoint); equal gains (within 1e-12) resolve to the
  smallest feature index, then smallest threshold; leaf ties go to the smallest class label.
* **NaN** in X: replaced by the training median per feature (stored, reused at prediction); infinite values
  and missing labels (NaN or `None`, also inside `object` arrays) raise `ValueError`. Wrong shapes /
  feature counts give descriptive errors. `min_impurity_decrease` must be a finite number >= 0 (no bool).
* **Design decision - zero-gain splits.** The course `_best_feature` only accepts a split whose gain is
  strictly greater than 0, so it stops at the root for XOR data (each single feature has gain 0). By default
  this tree (`allow_zero_gain_splits=True`) still takes the best split when `min_impurity_decrease=0`, which
  lets it learn XOR; depth/size limits still bound growth. `allow_zero_gain_splits=False` reproduces the
  course rule exactly, and `min_impurity_decrease > 0` always demands a positive gain. Both behaviours are
  tested (`tests/test_stream_review4.py`).
* `allow_feature_growth=True` lets the number of columns grow between chunks (e.g. after a streaming
  `OneHotEncoder`).

## Running things (from the repository root)

```bash
python -m unittest discover -s tests -v                       # everything (group + stream tests)
python -m unittest discover -s tests -p "test_stream_*.py"    # only the streaming tests
python benchmark/stream_benchmarks.py                         # full benchmark (~30 s)
python benchmark/stream_benchmarks.py --quick                 # small workloads (~2 s)
python demo/make_dataset.py                                   # regenerate demo/data/stream_data.csv
```

Open `demo/stream_demo.ipynb`, choose a Python kernel with NumPy and matplotlib, then **Restart & Run All**.
The notebook loads the CSV with `io.read_csv`, streams it in chunks of 100, trains the pipeline with
`partial_fit`, logs metrics, and plots them with `visualise.py`; figures are also saved to `demo/figures/`.

## Benchmarks (see `benchmark/stream_results.md`)

Measured on one machine (Python 3.13.18, NumPy 2.5.3, median of 3 runs; absolute times vary with hardware and load, the ratios are what matters):

| Task | Loop (ms) | Vectorised (ms) | Speedup |
|---|---:|---:|---:|
| Best split, 2000 samples x 6 features | 430 | 2.0 | ~215x |
| Predict, 50 000 samples | 225 | 6.0 | ~38x |
| Running mean/var, 200 000 rows | 529 | 69.8 | ~8x |

Effect of `buffer_size` on a 12 000-sample stream with a concept drift at 50 %: smaller buffers
adapt faster (prequential accuracy 0.895 at 100 vs 0.585 at 4000) but retrain more often
(120 vs 3 retrains); each retrain on a larger buffer costs more
(2.2 ms vs 46.2 ms). The model-memory column uses `estimate_memory`
(owning arrays are counted once; views are followed to their base).

## Notes and limitations
* The tree is retrained on the buffer only (as specified), so it forgets older chunks; this helps under
  concept drift and hurts on stationary data with small buffers.
* `estimate_memory` approximates the size of the object graph (arrays counted once, views followed to their base); it is not an allocator measurement.
* Numerical limits: the tree only compares values and computes thresholds overflow-safely (tested up to +-1.7e308). Running variance squares deviations, so `RunningStats` / `StandardScaler` reject `|x| > 1e100` with a clear error (`MinMaxScaler` has no such limit).
* `StreamingAUC.update_proba` requires a finite matrix with values in [0, 1] and rows summing to 1 (tolerance 1e-6); `update(y, score)` accepts any finite ranking score.
* Batch metrics validate shapes/labels (`ValueError`) and return NaN for empty input; zero denominators count as 0.0 without warnings.
* Only NumPy and matplotlib are used; the tests use the standard library's `unittest`.
