"""Retrain stored selected heads and compare test accuracy and MLP audit
agreement with the stored values (reproducibility check before re-auditing).

Usage: python scripts/revision/check_reproduce.py DATASET [N]
"""
import sys

import numpy as np

from common import DATASETS, QUANTUM, load_runs, rebuild
from redqueen import classical

ds = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
runs = load_runs(DATASETS[ds])
keys = sorted(k for k in runs if k[0] in QUANTUM)
rng = np.random.default_rng(0)
for k in [keys[i] for i in rng.choice(len(keys), n, replace=False)]:
    rec = runs[k]
    model, g, d, acc = rebuild(rec)
    T = classical.targets(model, d)
    m = 12
    a = classical.audit_mlp(model, d, classical.mlp_hidden((2 * m + 1) * d["n_classes"], g.n,
                                                            d["n_classes"]), T=T)["agree"]
    print(k, f"acc stored={rec['best']['acc']:.4f} now={acc:.4f} | "
             f"MLP agree@12 stored={rec['best']['audit']['curves']['mlp']['12']:.4f} now={a:.4f}",
          flush=True)
