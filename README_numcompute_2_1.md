# NumCompute

NumCompute is a modular scientific computing toolkit built with plain Python
and NumPy for Assignment 2.1. It includes CSV input, preprocessing, sorting and
searching, ranking, descriptive statistics, evaluation metrics, numerical
derivatives, transformation pipelines and performance comparisons.

## Requirements

- Python 3.10 or newer.
- NumPy 2.0 or newer. The current AUC implementation uses `numpy.trapezoid`.
- For the notebook: a Python notebook kernel (`ipykernel`) and a notebook
  interface, such as VS Code with Python/Jupyter notebook support.

The toolkit uses the Python standard library and NumPy. It does not require
pandas, scikit-learn, PyTorch or other ML/DL libraries. Tests use the standard
library's `unittest`; pytest is not required.

## Installation

Clone the repository and enter its root directory:

```powershell
git clone https://github.com/A1KESH1/NumCompute.git
cd NumCompute
```

This is a private repository, so cloning requires an authorised GitHub account.
If the repository is already cloned, open the existing `NumCompute` directory.
The repository root contains `pyproject.toml`, `numcompute/`, `tests/` and `README.md`.

Install NumCompute in editable mode from the repository root:

```powershell
python -m pip install -e .
```

This installs the package and its NumPy dependency from `pyproject.toml`.
Editable installation uses the local source files, so changes to Python
source are available without reinstalling. For the notebook, also install
its kernel support:

```powershell
python -m pip install ipykernel
```

To create a separate environment on Windows instead:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip install ipykernel
```

Select `.venv\Scripts\python.exe` as the interpreter and notebook kernel in
VS Code. The commands below use `python`; if the environment is not active,
replace `python` with `.\.venv\Scripts\python.exe`.

Verify the dependencies and project import from the repository root:

```powershell
python -c "import numpy; import numcompute; print('NumPy:', numpy.__version__); print('NumCompute import successful')"
```

Run the test, notebook and benchmark instructions below from the repository
root so their relative paths resolve correctly. After installation, the
package can also be imported outside the repository using the same Python
environment. The existing `environment.yml` contains platform-specific
packages and an absolute prefix; the pip installation above avoids relying
on that machine-specific configuration.

## Project Structure

```text
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py
│   ├── preprocessing.py
│   ├── sort_search.py
│   ├── rank.py
│   ├── stats.py
│   ├── metrics.py
│   ├── optim.py
│   ├── pipeline.py
│   ├── utils.py
│   └── benchmarking.py
├── tests/
│   ├── data/
│   ├── test_io.py
│   ├── test_preprocessing.py
│   ├── test_sort_search.py
│   ├── test_optim_pipeline.py
│   ├── test_utils.py
│   └── test_benchmarking.py
├── demo/
│   └── quickstart.ipynb
├── benchmark/
│   ├── run_benchmarks.py
│   ├── results.json
│   ├── results.md
│   └── README.md
├── environment.yml
├── pyproject.toml
└── README.md
```

## API Overview

| Module | Main interfaces |
|---|---|
| `io.py` | `read_csv`, `read_csv_chunks` |
| `preprocessing.py` | `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `SimpleImputer` |
| `sort_search.py` | `stable_sort`, `multi_key_sort`, `topk`, `quickselect`, `binary_search` |
| `rank.py` | `compute_rank`, `percentile` |
| `stats.py` | `compute_basic_stats`, `histogram`, `quantiles` |
| `metrics.py` | `accuracy`, `confusion_matrix`, `precision_recall_f1`, `mse`, `roc_curve`, `auc` |
| `optim.py` | `grad`, `jacobian` |
| `pipeline.py` | `Pipeline` |
| `utils.py` | `euclidean_distance`, `sigmoid`, `relu`, `softmax`, `logsumexp`, `topk_indices`, `batched` |
| `benchmarking.py` | `benchmark`, `compare_sum`, `compare_distances`, `environment_info`, `format_results` |

The examples import directly from submodules so that they work with the
current `__init__.py`. `precision_recall_f1` returns a tuple containing all
three metrics; `compute_basic_stats` returns a dictionary.

## Usage Examples

### CSV Input and Pipeline

```python
from io import StringIO
import numpy as np
from numcompute.io import read_csv
from numcompute.preprocessing import SimpleImputer, StandardScaler
from numcompute.pipeline import Pipeline

csv_data = StringIO("feature_1,feature_2\n1.0,\n2.0,5.0\n3.0,6.0\n")
X = read_csv(csv_data, delimiter=",", dtype=float, skip_header=1)

pipe = Pipeline([
    ("imputer", SimpleImputer(fill_value=0.0)),
    ("scaler", StandardScaler()),
])
X_transformed = pipe.fit_transform(X)
print(X_transformed)
print(pipe.transform(np.array([[2.0, np.nan]])))
```

For a file on disk, pass its path instead of `StringIO`, for example
`read_csv("data.csv", dtype=float, skip_header=1)`. For chunked input, use
`read_csv_chunks("data.csv", dtype=float, skip_header=1, chunk_size=1000)`.

Preprocessors accept two-dimensional `(samples, features)` arrays. Fit on
training data first, then transform new data with the fitted parameters.
Numeric scalers propagate NaNs, so impute missing numeric values first.
Categorical encoding also requires 2D input, such as
`OneHotEncoder().fit_transform(categories.reshape(-1, 1))` for one column.

### Selection, Ranking and Statistics

```python
import numpy as np
from numcompute.sort_search import topk, quickselect, binary_search
from numcompute.rank import compute_rank, percentile
from numcompute.stats import compute_basic_stats

values = np.array([10, 4, 8, 2, 9])
print(topk(values, 3))
print(quickselect(values, 3))
print(binary_search(np.array([1, 3, 5, 7]), 5))
print(compute_rank([10, 20, 20, 30], method="average"))
print(percentile(values, 50))
print(compute_basic_stats(values))
```

`topk` returns `(selected_values, original_indices)` by default. `quickselect`
uses a one-based `k`; the example's third smallest value is `8`.
`binary_search` requires sorted input and returns `(insertion_index, found)`.
Ranking supports `average`, `dense` and `ordinal` tie methods. Percentiles use
`q` in `[0, 100]`, while quantiles use `[0, 1]`.

### Numerical Derivatives

```python
import numpy as np
from numcompute.optim import grad, jacobian

f = lambda x: np.sum(x ** 2)
F = lambda x: np.array([x[0] + x[1], x[0] * x[1]])

print(grad(f, np.array([3.0, 4.0])))
print(jacobian(F, np.array([2.0, 3.0])))
```

The expected gradient is approximately `[6, 8]`. The expected Jacobian is
approximately `[[1, 1], [3, 2]]`. Inputs have shape `(n,)`; `grad` returns
`(n,)`, while `jacobian` returns `(m, n)` for a function with `m` outputs.
Both support `method="central"` and `method="forward"` and require a finite,
positive step size `h`.

## Running Tests

Run the full suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

The current verified suite contains 156 tests. Successful output ends with
`OK`; the test count may change as the group adds more tests. This verifies
the cases covered by those tests, rather than every possible input.

Run a specific test file using discovery:

```powershell
python -m unittest discover -s tests -p "test_utils.py" -v
python -m unittest discover -s tests -p "test_benchmarking.py" -v
python -m unittest discover -s tests -p "test_optim_pipeline.py" -v
```

Tests cover CSV input, preprocessing, selection/search, gradients, pipeline
integration, numerical utilities and benchmarking. The utility tests include
empty inputs, extreme values, NaNs/infinities, ties, invalid parameters and
non-contiguous slices. Benchmark tests check correctness, timing records,
validation, reporting and JSON output. They do not require NumPy to be faster
than Python loops on every machine.

## Running the Demo

1. Open the repository in VS Code.
2. Open `demo/quickstart.ipynb`.
3. Select the Python kernel with NumPy installed.
4. Choose **Restart**, then **Run All**.

The notebook locates the repository when started from the repository root,
its `demo/` directory, or the enclosing `OWN/` directory in the current layout.
If another working directory is used, open the project folder and restart the
kernel. A successful run completes every cell without a traceback.

The notebook demonstrates:

- Reading CSV text with a missing value using `read_csv`.
- Imputation and standardisation through a pipeline.
- One-hot encoding of categorical values.
- Top-k, quickselect, binary search, tie-aware ranks and percentiles.
- Descriptive statistics, histograms and quantiles.
- Classification/regression metrics and finite-difference derivatives.
- Python-loop versus NumPy sum timing, including a performance table and
  Python/NumPy/platform information.

CSV data is embedded in the notebook with `StringIO`, so no separate demo
CSV file needs to be downloaded. The notebook uses the group's current APIs
and does not require a trained ML model. Benchmark values saved in notebook
outputs should be regenerated in the environment used for the final report.

## Running Benchmarks

From the repository root:

```powershell
python benchmark/run_benchmarks.py
```

The command checks numerical agreement before timing, performs one untimed
warm-up per implementation, and compares:

- A Python-loop sum against `numpy.sum` on 100,000 float64 values.
- Python-loop Euclidean distances against the vectorised utility on 10,000
  rows with 8 features.

Defaults are `repeat=5` and `seed=42`. Reported times are seconds internally
and milliseconds in the table. The output contains mean, median, minimum,
maximum and the ratio of loop mean time to NumPy mean time. Input generation
is excluded from timing; function calls and output allocation are included.

Customise the workload:

```powershell
python benchmark/run_benchmarks.py --n 100000 --distance-n 10000 --features 8 --repeat 7 --seed 42
```

Save raw timing samples and environment information:

```powershell
python benchmark/run_benchmarks.py --output benchmark/results.json
```

`benchmark/results.json` contains Python/NumPy versions, platform, thread
settings, NumPy build configuration, workload settings and per-run timing
samples. `benchmark/results.md` contains the matching performance tables.
Running the command again overwrites both files. Use results from the same
run in the report. Speedups vary with hardware, input size and system load.

### Performance Results

Results are recorded in [benchmark/results.md](benchmark/results.md), with
raw timing samples in [benchmark/results.json](benchmark/results.json).
Measurements use Python 3.10.11, NumPy 2.2.6 and Windows, with float64 data,
seed 42, five repeats and one warm-up per implementation.

| Operation | Python loop mean (ms) | NumPy mean (ms) | Mean-time speedup |
|---|---:|---:|---:|
| Sum, 100,000 values | 4.6493 | 0.0308 | 150.85x |
| Euclidean distances, 10,000 x 8 | 16.4816 | 0.8639 | 19.08x |

Displayed times are rounded; speedups are calculated from unrounded means.
Performance depends on hardware, input size and system load.

## Utility Boundary Policies

- `logsumexp` returns `-inf` for an empty sum or an all-`-inf` slice, handles
  `+inf`, propagates NaNs and supports `keepdims`.
- `softmax` requires finite input and a non-empty reduction axis.
- `topk_indices` accepts 1D real numeric arrays and `1 <= k <= len(values)`;
  it rejects NaNs and breaks ties by the smaller original index.
- `batched` accepts positive integer batch sizes and includes the final short
  batch. An empty input yields no batches.
- `euclidean_distance` supports broadcasting and axis reductions, using scaled
  calculations to reduce intermediate overflow for large finite distances.
