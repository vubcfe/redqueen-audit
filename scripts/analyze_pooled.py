"""Decision rule of docs/DESIGN.md section 9 on the pooled pairs over all five
real datasets (section 10), next to the per-dataset verdicts.

Usage: python scripts/analyze_pooled.py

H2: one-sided Wilcoxon over every paired (dataset, budget, seed) run that
RED-QUEEN's MLP M90 exceeds GA's (and random search's), p < 0.05, with mean
test accuracy within 1 pp (scripts/rules.py). H1 is computed in paper/make_figures.py.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rules import rule_holds

DATASETS = {"eurosat": "results/paper", "so2sat": "results/paper_so2sat",
            "ucmerced": "results/paper_ucmerced", "breastmnist": "results/paper_breastmnist",
            "cifar10": "results/paper_cifar10"}

runs = {}
for name, d in DATASETS.items():
    n = 0
    for f in Path(d).glob("*.json"):
        r = json.loads(f.read_text())
        runs[(name, r["method"], r["budget"], r["seed"])] = r
        n += 1
    print(f"{name:12s} {n:3d} records")
keys = sorted({(k[0], k[2], k[3]) for k in runs})


def get(ds, method, b, s, path):
    r = runs.get((ds, method, b, s))
    if r is None:
        return None
    for k in path:
        r = r[k]
    return r


def h2(other, fam, subset):
    """M90 test over pairs with both audits; accuracy difference over every pair
    with both accuracies, as in analyze_paper.py."""
    x = [(get(d, "redqueen", b, s, ["best", "audit", "m90", fam]),
          get(d, other, b, s, ["best", "audit", "m90", fam])) for d, b, s in subset]
    x = np.array([t for t in x if None not in t], dtype=float)
    if len(x) < 5:
        return None
    acc = [(get(d, "redqueen", b, s, ["best", "acc"]), get(d, other, b, s, ["best", "acc"]))
           for d, b, s in subset]
    acc = np.array([t for t in acc if None not in t], dtype=float)
    diff = x[:, 0] - x[:, 1]
    p = wilcoxon(x[:, 0], x[:, 1], alternative="greater").pvalue if np.any(diff) else 1.0
    return len(x), int((diff > 0).sum()), int((diff < 0).sum()), p, float(np.mean(acc[:, 0] - acc[:, 1]))


print("\n=== H2 per dataset and pooled (MLP M90 primary; wins/losses, rest ties)")
for other in ("ga", "random"):
    for fam in ("mlp", "fourier", "rff"):
        tag = "PRIMARY" if fam == "mlp" else "in-family"
        print(f"\n  RED-QUEEN vs {other}, {fam} ({tag})")
        for name in list(DATASETS) + ["POOLED"]:
            sub = keys if name == "POOLED" else [k for k in keys if k[0] == name]
            res = h2(other, fam, sub)
            if res is None:
                continue
            n, w, l, p, dacc = res
            ok = rule_holds(p, dacc)
            print(f"    {name:12s} n={n:2d} wins={w:2d} losses={l:2d} p={p:.4f} "
                  f"d_acc={dacc * 100:+.2f} pp" + ("  [rule holds]" if fam == "mlp" and ok else ""))

pooled = {o: h2(o, "mlp", keys) for o in ("ga", "random")}
holds = {o: r is not None and rule_holds(r[3], r[4]) for o, r in pooled.items()}
print("\nPooled decision (DESIGN.md sections 9-10):")
if holds["ga"]:
    print("  METHOD CLAIM HOLDS vs GA" + (" and vs random search" if holds["random"]
                                          else " (not vs random search: no claim about the operator)"))
else:
    print("  Method claim NOT supported -> adversarial audit")

# H1 (per dataset, pooled, matched heads, per method) is computed in
# paper/make_figures.py, with tied validation accuracies averaged.
