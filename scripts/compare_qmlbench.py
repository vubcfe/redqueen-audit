"""Rank our runs among the published qml-benchmarks models, per dataset.

Usage: python scripts/compare_qmlbench.py configs/qmlbench_pilot.yaml
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "src")
from redqueen.qmlbench import published  # noqa: E402

cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
out_dir = Path(cfg["out_dir"])
for bench in cfg["benchmarks"]:
    for stem in bench["stems"]:
        table = published(cfg["published_root"], bench["results_dir"], stem)
        if table.empty:  # results folder missing or names not matching
            print(f"\n(no published results found for {bench['results_dir']}/{stem})")
            table = pd.DataFrame(columns=["model", "test_mean", "test_std"])
        for method in cfg["methods"]:
            runs = [json.loads(f.read_text()) for f in sorted(out_dir.glob(f"{method}_{stem}_s*.json"))]
            if runs:
                acc = np.concatenate([r["test_acc"] for r in runs])
                table.loc[len(table)] = {"model": f"** {method.upper()} **", "test_mean": acc.mean(),
                                         "test_std": acc.std(ddof=1)}
        table = table.sort_values("test_mean", ascending=False).reset_index(drop=True)
        table.index += 1
        print(f"\n=== {stem}  ({len(table)} models)")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
