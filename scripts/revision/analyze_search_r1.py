"""Revision R1: analysis of the new search runs (Reviewer 4, points 1 and 2).

Usage (repository root): python scripts/revision/analyze_search_r1.py

A  fixed parasites vs co-evolving parasites (RED-QUEEN), and vs GA / random
B  doubled search budget (296 hosts) at the middle label budget: decision rule
   among the three searches, and change from 156 to 296 hosts per method
C  imitation weight lambda in {0, 0.1, 0.2, 0.5, 1.0} at the middle budget
Primary outcome as in the paper: MLP M90 (censored at 384) with one-sided
Wilcoxon; accuracy as mean paired difference with 95 % bootstrap interval.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATASETS, ROOT, load_runs  # noqa: E402
from rules import rule_holds  # noqa: E402

BUDGETS = (3, 6, 12, 24, 48, 96, 192)
NAMES = {"eurosat": "EuroSAT", "so2sat": "So2Sat", "ucmerced": "UC Merced",
         "breastmnist": "BreastMNIST", "cifar10": "CIFAR-10"}
MID = {"eurosat": 0.05, "so2sat": 0.05, "ucmerced": 0.25, "breastmnist": 0.25, "cifar10": 0.05}

PAPER = {ds: load_runs(d) for ds, d in DATASETS.items()}


def rev(name, ds):
    return load_runs(f"results/revision/{name}_{ds}")


FIX = {ds: rev("fixedpar", ds) for ds in DATASETS}
G20 = {ds: rev("gen20", ds) for ds in DATASETS}
LAM = {lam: {ds: rev(f"lam{lam}", ds) for ds in DATASETS} for lam in ("0.1", "0.5", "1")}


def m90(r):
    a = r["best"]["audit"]
    return 384.0 if a["m90_censored"]["mlp"] else float(a["m90"]["mlp"])


def cens(r):
    return r["best"]["audit"]["m90_censored"]["mlp"]


def area(r):
    c = np.array([r["best"]["audit"]["curves"]["mlp"][str(m)] for m in BUDGETS])
    return 1 - float(np.trapezoid(c, dx=1.0) / (len(c) - 1))


def acc(r):
    return r["best"]["acc"]


def boot_ci(x, level=0.95, n=10000):
    x = np.asarray(x, float)
    rng = np.random.default_rng(2026)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return tuple(float(v) for v in np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2]))


def compare(P, label):
    """P: list of (run_a, run_b). Prints the decision rule for a over b and
    encoding-free checks. Returns a dict."""
    if not P:
        print(f"  {label:14s} no pairs")
        return None
    d = np.array([m90(a) - m90(b) for a, b in P])
    w, l = int((d > 0).sum()), int((d < 0).sum())
    p = float(wilcoxon(d, alternative="greater").pvalue) if np.any(d) else 1.0
    sg = np.array([0 if (cens(a) and cens(b)) else np.sign(m90(a) - m90(b)) for a, b in P])
    ws, ls = int((sg > 0).sum()), int((sg < 0).sum())
    ps = binomtest(ws, ws + ls, 0.5, alternative="greater").pvalue if ws + ls else 1.0
    da = 100 * np.array([area(a) - area(b) for a, b in P])
    pa = float(wilcoxon(da, alternative="greater").pvalue) if np.any(da) else 1.0
    dacc = 100 * np.array([acc(a) - acc(b) for a, b in P])
    lo, hi = boot_ci(dacc)
    lo_a, hi_a = boot_ci(da)
    rule = rule_holds(p, dacc.mean() / 100)
    print(f"  {label:14s} n={len(P):2d} M90 W/L {w:2d}/{l:<2d} p={p:.4f} | sign p={ps:.4f} | "
          f"d(1-A) {da.mean():+6.2f}pp [{lo_a:+.2f},{hi_a:+.2f}] p={pa:.4f} | "
          f"dacc {dacc.mean():+6.2f}pp [{lo:+.2f},{hi:+.2f}] | rule {'yes' if rule else 'no'}")
    return {"w": w, "l": l, "p": p, "dacc": dacc.mean(), "ci": (lo, hi), "rule": rule,
            "darea": da.mean(), "area_ci": (lo_a, hi_a), "p_area": pa, "n": len(P)}


def pairs(A, B, keys=None):
    keys = keys if keys is not None else sorted(k for k in A)
    out = []
    for k in keys:
        ka = k
        kb = (None, k[1], k[2])
        a = A.get(ka)
        b = next((v for kk, v in B.items() if kk[1:] == k[1:]), None)
        if a is not None and b is not None:
            out.append((a, b))
    return out


def by_method(runs, method):
    return {k: v for k, v in runs.items() if k[0] == method}


groups = [(NAMES[ds], [ds]) for ds in DATASETS] + [("Pooled", list(DATASETS))]

# ---------------------------------------------------------------- A
print("=" * 78)
print("A. Fixed parasites (RED-QUEEN with the initial parasites frozen)")
for other, label in (("redqueen", "co-evolving RQ"), ("ga", "GA"), ("random", "random")):
    print(f"\n-- fixed parasites vs {label}  (positive: fixed harder to imitate / more accurate)")
    for name, dl in groups:
        P = []
        for ds in dl:
            P += pairs(by_method(FIX[ds], "redqueen_fixedpar"), by_method(PAPER[ds], other))
        compare(P, name)
print("\n-- co-evolving RQ vs fixed parasites, two-sided view of the M90 difference")
for name, dl in groups:
    P = []
    for ds in dl:
        P += pairs(by_method(PAPER[ds], "redqueen"), by_method(FIX[ds], "redqueen_fixedpar"))
    if P:
        d = np.array([m90(a) - m90(b) for a, b in P])
        p2 = float(wilcoxon(d).pvalue) if np.any(d) else 1.0
        print(f"  {name:12s} n={len(P)} W/L {(d > 0).sum()}/{(d < 0).sum()} two-sided p={p2:.4f}")
# sanity: the first generation of a fixed-parasite run equals RQ's first generation
same0 = []
for ds in DATASETS:
    for k, r in FIX[ds].items():
        q = PAPER[ds].get(("redqueen", k[1], k[2]))
        if q:
            same0.append(abs(r["history"][0]["best_fitness"] - q["history"][0]["best_fitness"]) < 1e-12)
print(f"\nsanity: generation-0 best fitness identical to the paired RQ run in {sum(same0)}/{len(same0)} runs")

# ---------------------------------------------------------------- B
print("\n" + "=" * 78)
print("B. Doubled search budget (16 + 20 x 14 = 296 hosts), middle label budget")
for a, b in (("redqueen", "ga"), ("redqueen", "random")):
    print(f"\n-- {a} vs {b} at 296 hosts")
    for name, dl in groups:
        P = []
        for ds in dl:
            P += pairs(by_method(G20[ds], a), by_method(G20[ds], b))
        compare(P, name)
    print(f"-- same comparison at 156 hosts, same label budget (paper runs)")
    for name, dl in groups:
        P = []
        for ds in dl:
            A = {k: v for k, v in by_method(PAPER[ds], a).items() if k[1] == MID[ds]}
            P += pairs(A, by_method(PAPER[ds], b))
        compare(P, name)
print("\n-- 296 vs 156 hosts, per method (positive: more inimitable / more accurate at 296)")
for m in ("redqueen", "ga", "random"):
    for name, dl in groups:
        P = []
        for ds in dl:
            P += pairs(by_method(G20[ds], m), by_method(PAPER[ds], m))
        compare(P, f"{m} {name}")
same10 = []
for ds in DATASETS:
    for k, r in G20[ds].items():
        q = PAPER[ds].get(k)
        if q:
            h = [x["best_fitness"] for x in r["history"][:11]]
            same10.append(h == [x["best_fitness"] for x in q["history"]])
print(f"sanity: first 10 generations identical to the paper run in {sum(same10)}/{len(same10)} runs")

# ---------------------------------------------------------------- C
print("\n" + "=" * 78)
print("C. Imitation weight lambda (middle label budget, 25 runs per value)")
print(f"{'lambda':>6s} | mean acc | median M90 | censored | mean 1-A | vs GA: rule test")


def lam_runs(lam):
    out = {}
    for ds in DATASETS:
        if lam == "0":
            src = {k: v for k, v in by_method(PAPER[ds], "ga").items() if k[1] == MID[ds]}
        elif lam == "0.2":
            src = {k: v for k, v in by_method(PAPER[ds], "redqueen").items() if k[1] == MID[ds]}
        else:
            src = by_method(LAM[lam][ds], "redqueen")
        for k, v in src.items():
            out[(ds,) + k[1:]] = v
    return out


ga = lam_runs("0")
rows = {}
for lam in ("0", "0.1", "0.2", "0.5", "1"):
    rs = lam_runs(lam)
    if not rs:
        print(f"{lam:>6s} | (no runs yet)")
        continue
    v = list(rs.values())
    P = [(rs[k], ga[k]) for k in rs if k in ga]
    print(f"{lam:>6s} | {100 * np.mean([acc(r) for r in v]):6.2f} | {np.median([m90(r) for r in v]):6.0f} | "
          f"{sum(cens(r) for r in v):3d}/{len(v):<3d} | {100 * np.mean([area(r) for r in v]):6.2f} |", end="")
    rows[lam] = rs
    if lam != "0":
        compare(P, "")
    else:
        print()
print("\nper dataset: mean acc / median M90 by lambda")
for ds in DATASETS:
    cells = []
    for lam, rs in rows.items():
        v = [r for k, r in rs.items() if k[0] == ds]
        if v:
            cells.append(f"{lam}: {100 * np.mean([acc(r) for r in v]):.1f}/{np.median([m90(r) for r in v]):.0f}")
    print(f"  {NAMES[ds]:12s} " + "  ".join(cells))
# monotone trend across lambda (Spearman over runs, exploratory)
lam_x, a_y, m_y = [], [], []
for lam, rs in rows.items():
    for r in rs.values():
        lam_x.append(float(lam)), a_y.append(acc(r)), m_y.append(area(r))
if len(set(lam_x)) > 2:
    print(f"Spearman lambda vs accuracy rho={spearmanr(lam_x, a_y).statistic:+.2f} "
          f"p={spearmanr(lam_x, a_y).pvalue:.4f}; vs 1-A rho={spearmanr(lam_x, m_y).statistic:+.2f} "
          f"p={spearmanr(lam_x, m_y).pvalue:.4f} (pooled over datasets; exploratory)")
