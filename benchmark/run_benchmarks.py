import argparse
import json
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from numcompute.benchmarking import format_results, main as run_benchmark


def _summary(report):
    environment = report["environment"]
    settings = report["settings"]
    return "\n".join([
        "# Benchmark Results",
        "",
        f"Python: {environment['python']}; NumPy: {environment['numpy']}.",
        f"Platform: {environment['platform']}.",
        f"dtype: {settings['dtype']}; repeat: {settings['repeat']}; "
        f"seed: {settings['seed']}; warm-up: {settings['warmup']}.",
        f"Thread settings: {environment['thread_settings']}.",
        "",
        f"## Sum: {settings['n']} values",
        "",
        format_results(report["sum"]),
        "",
        f"## Distances: {settings['distance_n']} rows x {settings['features']} features",
        "",
        format_results(report["distances"]),
        "",
        "Times include function calls and output allocation; input creation is excluded.",
        "Correctness checks run before timing. Speedups depend on size, hardware and load.",
        "Raw samples and NumPy build information are stored in the matching JSON file.",
        "",
    ])


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    options, remaining = parser.parse_known_args()
    original_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *remaining, "--output", str(options.output)]
        run_benchmark()
    finally:
        sys.argv = original_argv
    report = json.loads(options.output.read_text(encoding="utf-8"))
    summary_path = options.output.with_suffix(".md")
    summary_path.write_text(_summary(report), encoding="utf-8")
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
