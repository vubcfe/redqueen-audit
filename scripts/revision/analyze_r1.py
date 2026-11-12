"""Revision R1 (CMC 96001): analyses on the stored run records that need no
new runs. Writes every number quoted in the revision to stdout.

Usage (from the repository root): python scripts/revision/analyze_r1.py

Sections
  A  censoring of MLP M90 (Reviewer 4, point 4)
  B  H2 without the censored-value encoding: sign test, Wilcoxon under
     several encodings, agreement at fixed budgets, area under the
     imitation curve (Reviewer 4, point 4)
  C  paired accuracy differences with 95 % bootstrap and 90 % intervals
     (Reviewer 4, point 5)
  D  trade-off evidence by source (Reviewer 4, point 5)
  E  effective two-qubit gate counts (Reviewer 3, point 2)
  F  label-subset nesting across budgets and spectrum sizes (Reviewer 4, point 6)
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, QUANTUM, ROOT, load_runs  # noqa: E402

from redqueen.genome import HostGenome  # noqa: E402

BUDGETS = (3, 6, 12, 24, 48, 96, 192)
NAMES = {"eurosat": "EuroSAT", "so2sat": "So2Sat", "ucmerced": "UC Merced",
         "breastmnist": "BreastMNIST", "cifar10": "CIFAR-10"}
RNG = np.random.default_rng(2026)
ALL = {ds: load_runs(d) for ds, d in DATASETS.items()}


def best(r):
    return r["best"]


def curve(r):
    return np.array([best(r)["audit"]["curves"]["mlp"][str(m)] for m in BUDGETS])


def m90(r, censor=384.0):
    a = best(r)["audit"]
    return censor if a["m90_censored"]["mlp"] else float(a["m90"]["mlp"])


def auc(r):
    """Mean agreement over the log-spaced budget grid (trapezoid on log2 M,
    normalised to [0, 1]); higher = easier to imitate. Uses observed values
    only, so no censoring."""
    c = curve(r)
    return float(np.trapezoid(c, dx=1.0) / (len(c) - 1))


def pairs(ds, a, b):
    runs = ALL[ds]
    keys = sorted({(k[1], k[2]) for k in runs if k[0] == a})
    return [(runs[(a, bu, s)], runs[(b, bu, s)]) for bu, s in keys
            if (a, bu, s) in runs and (b, bu, s) in runs]


def pooled(a, b):
    return [p for ds in DATASETS for p in pairs(ds, a, b)]


def boot_ci(x, level=0.95, n=10000):
    """Percentile bootstrap over paired runs; a fixed generator per call, so a
    reported interval does not depend on which analyses ran before it."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(2026)
    m = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    lo, hi = np.quantile(m, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def wil(x, alternative):
    x = np.asarray(x, float)
    return float(wilcoxon(x, alternative=alternative).pvalue) if np.any(x != 0) else 1.0


def sign_test(x):
    """One-sided exact sign test that x > 0, ties dropped. Valid under
    censoring: a censored-vs-observed pair has a known sign."""
    w, l = int((x > 0).sum()), int((x < 0).sum())
    p = binomtest(w, w + l, 0.5, alternative="greater").pvalue if w + l else 1.0
    return w, l, float(p)


def sign_of_m90(ra, rb):
    """Sign of M90(a) - M90(b) using only what is observed: two censored
    values are a tie (unknown order)."""
    ca, cb = best(ra)["audit"]["m90_censored"]["mlp"], best(rb)["audit"]["m90_censored"]["mlp"]
    if ca and cb:
        return 0
    return int(np.sign(m90(ra) - m90(rb)))


groups = [(NAMES[ds], ds) for ds in DATASETS] + [("Pooled", None)]


def get_pairs(ds, a, b):
    return pooled(a, b) if ds is None else pairs(ds, a, b)


# ------------------------------------------------------------------ A
print("=" * 78)
print("A. Censoring of MLP M90 (agreement 0.9 not reached at M = 192)")
print(f"{'dataset':12s}" + "".join(f"{m:>10s}" for m in QUANTUM))
tot = {m: [0, 0] for m in QUANTUM}
for name, ds in groups[:-1]:
    row = []
    for m in QUANTUM:
        rs = [r for k, r in ALL[ds].items() if k[0] == m]
        c = sum(best(r)["audit"]["m90_censored"]["mlp"] for r in rs)
        tot[m][0] += c
        tot[m][1] += len(rs)
        row.append(f"{c:>4d}/{len(rs):<5d}")
    print(f"{name:12s}" + "".join(f"{v:>10s}" for v in row))
print(f"{'all':12s}" + "".join(f"{f'{c}/{n}':>10s}" for c, n in tot.values()))
n_c = sum(c for c, _ in tot.values())
n_all = sum(n for _, n in tot.values())
print(f"all quantum runs: {n_c}/{n_all} censored ({100 * n_c / n_all:.1f}%)")
for a, b in (("redqueen", "ga"), ("redqueen", "random")):
    for name, ds in groups:
        P = get_pairs(ds, a, b)
        ca = sum(best(x)["audit"]["m90_censored"]["mlp"] for x, _ in P)
        cb = sum(best(y)["audit"]["m90_censored"]["mlp"] for _, y in P)
        both = sum(best(x)["audit"]["m90_censored"]["mlp"] and best(y)["audit"]["m90_censored"]["mlp"]
                   for x, y in P)
        one = sum(best(x)["audit"]["m90_censored"]["mlp"] != best(y)["audit"]["m90_censored"]["mlp"]
                  for x, y in P)
        print(f"  {a} vs {b:7s} {name:12s}: censored {a} {ca}, {b} {cb}, both {both}, "
              f"pairs with >=1 censored {one + both}/{len(P)}")

# ------------------------------------------------------------------ B
print("\n" + "=" * 78)
print("B. H2 without the M90 = 384 encoding (one-sided: RED-QUEEN harder to imitate)")
for a, b in (("redqueen", "ga"), ("redqueen", "random")):
    print(f"\n--- {a} vs {b}")
    print(f"{'':12s} {'sign W/L':>9s} {'p_sign':>7s} | Wilcoxon p at censor value"
          f" {'193':>7s} {'384':>7s} {'768':>7s} | median dM90 at 193/384 |"
          f" dAgree@12 p | @48 p | @192 p | dAUC (95% CI) p")
    for name, ds in groups:
        P = get_pairs(ds, a, b)
        sg = np.array([sign_of_m90(x, y) for x, y in P])
        w, l, ps = sign_test(sg)
        pw = {c: wil([m90(x, c) - m90(y, c) for x, y in P], "greater") for c in (193, 384, 768)}
        med = {c: float(np.median([m90(x, c) - m90(y, c) for x, y in P])) for c in (193, 384)}
        fx = {}
        for M in (12, 48, 192):
            i = BUDGETS.index(M)
            d = np.array([curve(y)[i] - curve(x)[i] for x, y in P])   # >0: a harder
            fx[M] = (float(d.mean()), wil(d, "greater"))
        dA = np.array([auc(y) - auc(x) for x, y in P])               # >0: a harder
        lo, hi = boot_ci(dA)
        print(f"{name:12s} {w:>4d}/{l:<4d} {ps:7.4f} |{'':27s}{pw[193]:7.4f} {pw[384]:7.4f} "
              f"{pw[768]:7.4f} | {med[193]:+6.0f}/{med[384]:+6.0f} | "
              + " | ".join(f"{100 * fx[M][0]:+5.2f}pp {fx[M][1]:.4f}" for M in (12, 48, 192))
              + f" | {100 * dA.mean():+5.2f}pp [{100 * lo:+5.2f},{100 * hi:+5.2f}] "
              f"{wil(dA, 'greater'):.4f}")

# ------------------------------------------------------------------ C
print("\n" + "=" * 78)
print("C. Paired test-accuracy differences (pp), RED-QUEEN minus comparator")
print("   mean, 95% bootstrap CI (10000 resamples of paired runs), 90% CI "
      "(lower end > -1 pp would be evidence of non-inferiority at 1 pp, alpha 0.05), "
      "two-sided Wilcoxon p")
for a, b in (("redqueen", "ga"), ("redqueen", "random")):
    print(f"\n--- {a} vs {b}")
    for name, ds in groups:
        P = get_pairs(ds, a, b)
        d = 100 * np.array([best(x)["acc"] - best(y)["acc"] for x, y in P])
        lo, hi = boot_ci(d)
        lo9, hi9 = boot_ci(d, 0.90)
        ni = "yes" if lo9 > -1 else "no"
        print(f"{name:12s} n={len(d):2d} mean {d.mean():+6.2f}  95% [{lo:+6.2f},{hi:+6.2f}]"
              f"  90% [{lo9:+6.2f},{hi9:+6.2f}]  non-inferior@1pp: {ni:3s}"
              f"  Wilcoxon p={wil(d, 'two-sided'):.4f}")

# ------------------------------------------------------------------ D
print("\n" + "=" * 78)
print("D. Trade-off evidence by source")
print("D1 search-time gap vs test accuracy inside each run's final population is in")
print("   the paper (scripts/analyze_pooled.py); not recomputed here.")
print("D2 held-out MLP auditor: Spearman rho between test accuracy and inimitability")
print("   across the 75 selected heads per dataset, with and without HEA L=2;")
print("   inimitability = M90 (censored at 384) and, uncensored, 1 - AUC.")
for name, ds in groups[:-1]:
    rs = [r for k, r in ALL[ds].items() if k[0] in QUANTUM]
    out = []
    for excl in (False, True):
        sub = [r for r in rs if not (excl and r["method"] == "hea_L2")]
        acc = [best(r)["acc"] for r in sub]
        r1 = spearmanr(acc, [m90(r) for r in sub])
        r2 = spearmanr(acc, [1 - auc(r) for r in sub])
        out.append(f"n={len(sub):2d} rho(M90)={r1.statistic:+.2f} p={r1.pvalue:.3f} "
                   f"rho(1-AUC)={r2.statistic:+.2f} p={r2.pvalue:.3f}")
    print(f"{name:12s} all: {out[0]}\n{'':12s} w/o HEA L=2: {out[1]}")

# ------------------------------------------------------------------ E
print("\n" + "=" * 78)
print("E. Two-qubit gates of the selected heads: as built L*|E| vs effective (L-1)*|E|")
for m in QUANTUM:
    gs = [HostGenome.from_dict(best(r)["genome"]) for ds in DATASETS
          for k, r in ALL[ds].items() if k[0] == m]
    built = np.mean([g.two_qubit_gates for g in gs])
    eff = np.mean([g.two_qubit_gates_effective for g in gs])
    L1 = sum(g.layers == 1 for g in gs)
    cz = sum(g.ent_gate == "CZ" for g in gs)
    print(f"{m:9s} n={len(gs):3d} mean built {built:5.1f} effective {eff:5.1f} | "
          f"L=1 (product circuit) {L1:3d} | mean L {np.mean([g.layers for g in gs]):.2f} | CZ {cz}")
# Figure 3 example (EuroSAT, 5 % labels, seed 0)
for m in ("ga", "redqueen"):
    g = HostGenome.from_dict(best(ALL["eurosat"][(m, 0.05, 0)])["genome"])
    print(f"Figure 3 {m}: L={g.layers} |E|={len(g.edges)} gate={g.ent_gate} "
          f"built={g.two_qubit_gates} effective={g.two_qubit_gates_effective}")

# ------------------------------------------------------------------ F
print("\n" + "=" * 78)
print("F1. Spectrum sizes: order <= 3, one representative per +/- pair, cap 2000")
from redqueen.surrogate import spectrum  # noqa: E402

for L in (1, 2, 3):
    full = (2 * L + 1) ** 8
    n3 = len(spectrum(8, L, 3))
    print(f"L={L}: order<=3 non-zero frequencies (up to sign) {n3}, used {min(n3, 2000)}; "
          f"full spectrum has ({2 * L + 1}^8 - 1)/2 = {(full - 1) // 2} up to sign")
print("F2. Label subsets across budgets (same seed): see scripts/revision/check_nesting.py")
print(f"\n(root: {ROOT})")

# ------------------------------------------------------------------ G
print("\n" + "=" * 78)
print("G. LaTeX rows for the robustness table (tab:h2robust)")
for a, b in (("redqueen", "ga"), ("redqueen", "random")):
    print(f"% {a} vs {b}")
    for name, ds in groups:
        P = get_pairs(ds, a, b)
        cens = sum(best(x)["audit"]["m90_censored"]["mlp"] or best(y)["audit"]["m90_censored"]["mlp"]
                   for x, y in P)
        sg = np.array([sign_of_m90(x, y) for x, y in P])
        w, l, ps = sign_test(sg)
        dA = 100 * np.array([auc(y) - auc(x) for x, y in P])
        lo, hi = boot_ci(dA)
        pa = wil(dA, "greater")
        d = 100 * np.array([best(x)["acc"] - best(y)["acc"] for x, y in P])
        clo, chi = boot_ci(d)
        lo9, _ = boot_ci(d, 0.90)
        fp = lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}"  # noqa: E731
        print(f"{name} & {cens}/{len(P)} & {w}/{l} & {fp(ps)} & ${dA.mean():+.2f}$ "
              f"$[{lo:+.2f},{hi:+.2f}]$ & {fp(pa)} & ${d.mean():+.2f}$ $[{clo:+.2f},{chi:+.2f}]$ "
              f"& ${lo9:+.2f}$ \\\\")
