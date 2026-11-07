"""Regenerate demo/data/stream_data.csv (6000 rows, 6 features + label, concept drift at 50%)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from numcompute_stream.datasets import make_drifting_data, write_csv  # noqa: E402

if __name__ == "__main__":
    X, y = make_drifting_data(n_samples=6000, n_features=6, drift_at=0.5, noise=0.05,
                              missing_rate=0.02, seed=42)
    out = Path(__file__).resolve().parent / "data" / "stream_data.csv"
    write_csv(out, X, y)
    print(f"wrote {out} with {len(X)} rows")
