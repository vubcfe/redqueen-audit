"""Apply the decision rule of docs/DESIGN.md section 9 to the main EuroSAT runs.

Usage: python scripts/analyze_paper.py [results/paper]

H2 (method claim): one-sided Wilcoxon signed-rank over paired (budget, seed)
runs that RED-QUEEN's MLP-distillation M90 (primary) exceeds GA's, p < 0.05,
and mean test accuracy within 1 pp of GA's; the same test against random
search for a claim about the search operator. Fourier and RFF M90 are
reported as in-family checks. H1: best quantum head vs best classical head,
paired differences per label budget with Holm correction.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from rules import holm, rule_holds, val_pick

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paper")
runs = {}
for f in run_dir.glob("*.json"):
    r = json.loads(f.read_text())
    runs[(r["method"], r["budget"], r["seed"])] = r

budgets = sorted({k[1] for k in runs})
seeds = sorted({k[2] for k in runs})
pairs = [(b, s) for b in budgets for s in seeds]
QUANTUM = ("redqueen", "ga", "random", "hea_L1", "hea_L2")


def get(method, b, s, path):
    r = runs.get((method, b, s))
    if r is None:
        return None
    for k in path:
        r = r[k]
    return r


print(f"runs: {len(runs)}  budgets: {budgets}  seeds: {seeds}\n")

# ---------------------------------------------------------------- summary table
print("mean over seeds: test acc | MLP M90 | Fourier M90 | RFF M90 | params")
for b in budgets:
    print(f"\nlabel budget {b:g}")
    for m in QUANTUM:
        acc = [get(m, b, s, ["best", "acc"]) for s in seeds]
        if any(a is None for a in acc):
            continue
        m90 = {fam: [get(m, b, s, ["best", "audit", "m90", fam]) for s in seeds]
               for fam in ("mlp", "fourier", "rff")}
        par = [get(m, b, s, ["best", "n_params"]) for s in seeds]
        print(f"  {m:9s} {np.mean(acc):.4f}±{np.std(acc, ddof=1):.4f} | "
              + " | ".join(f"{np.mean(v):6.1f}" for v in m90.values()) + f" | {np.mean(par):.0f}")
    cls = [runs.get(("classical", b, s)) for s in seeds]
    if all(cls):
        for k in [k for k, v in cls[0].items() if isinstance(v, dict) and "acc" in v]:
            print(f"  {k:22s} {np.mean([c[k]['acc'] for c in cls]):.4f}")

# ---------------------------------------------------------------- H2
print("\n=== H2: RED-QUEEN vs GA / random (paired over budget x seed)")
verdict = {}
for other in ("ga", "random"):
    for fam in ("mlp", "fourier", "rff"):
        x = [(get("redqueen", b, s, ["best", "audit", "m90", fam]),
              get(other, b, s, ["best", "audit", "m90", fam])) for b, s in pairs]
        x = [(a, c) for a, c in x if a is not None and c is not None]
        if len(x) < 5:
            continue
        a, c = map(np.array, zip(*x))
        diff = a - c
        p = wilcoxon(a, c, alternative="greater").pvalue if np.any(diff) else 1.0
        acc = [(get("redqueen", b, s, ["best", "acc"]), get(other, b, s, ["best", "acc"]))
               for b, s in pairs]
        acc = [(u, v) for u, v in acc if u is not None and v is not None]
        dacc = float(np.mean([u - v for u, v in acc]))
        tag = "PRIMARY" if fam == "mlp" else "in-family"
        print(f"  vs {other:6s} {fam:7s} ({tag:9s}) n={len(x):2d} median dM90={np.median(diff):+.1f} "
              f"wins={int((diff > 0).sum())}/{len(x)} p={p:.4f} | d_acc={dacc * 100:+.2f} pp")
        if fam == "mlp":
            verdict[other] = rule_holds(p, dacc)

print("\nDecision (DESIGN.md section 9):")
if verdict.get("ga"):
    print("  METHOD CLAIM HOLDS vs GA" + (" and vs random search" if verdict.get("random")
                                          else " (not vs random search: no claim about the operator)"))
else:
    print("  Method claim NOT supported -> write as adversarial audit "
          "(report the search's best M90 as an empirical upper bound)")

# ---------------------------------------------------------------- H1
print("\n=== H1: best quantum head vs best classical head (test acc, paired over seeds)")
ps, rows = [], []
for b in budgets:
    q, c = [], []
    for s in seeds:
        qa = [get(m, b, s, ["best", "acc"]) for m in QUANTUM]
        cr = runs.get(("classical", b, s))
        if None in qa or cr is None:
            continue
        # selection by validation accuracy, ties averaged
        q.append(val_pick([(get(m, b, s, ["best", "acc_val"]), get(m, b, s, ["best", "acc"]))
                           for m in QUANTUM]))
        c.append(val_pick([(v["acc_val"], v["acc"]) for k, v in cr.items()
                           if isinstance(v, dict) and "acc_val" in v and k != "linear_probe_full"]))
    if len(q) >= 3:
        d = np.array(q) - np.array(c)
        p = wilcoxon(q, c).pvalue if np.any(d) else 1.0
        ps.append(p)
        rows.append((b, d.mean(), len(q)))
for (b, dm, n), pa in zip(rows, holm(np.array(ps)) if ps else []):
    print(f"  budget {b:g}: quantum - classical = {dm * 100:+.2f} pp (n={n}), Holm p={pa:.4f}")
