"""Revision R1 (Reviewer 4, point 3): analysis of the MLP re-audit
(scripts/revision/reaudit_mlp.py).

Usage (repository root): python scripts/revision/analyze_reaudit.py

1  reproduction of the stored heads and of the original auditor
2  convergence: fit loss and test agreement from 2000 to 10000 steps
3  initialisation: spread over five seeds; H2 re-run per seed
4  tuned auditor (lr x seed x steps chosen on the selection split): M90 and H2
5  fit-set size (EuroSAT, So2Sat, CIFAR-10): M90 and H2 at 124/336/1000/3200
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, ROOT  # noqa: E402

BUDGETS = (3, 6, 12, 24, 48, 96, 192)
NAMES = {"eurosat": "EuroSAT", "so2sat": "So2Sat", "ucmerced": "UC Merced",
         "breastmnist": "BreastMNIST", "cifar10": "CIFAR-10"}
CENSOR = 384


def load():
    out = {}
    for ds in DATASETS:
        for f in (ROOT / "results/revision/reaudit" / ds).glob("*.json"):
            r = json.loads(f.read_text())
            out[(ds, r["method"], r["budget"], r["seed"])] = r
    return out


R = load()
print(f"re-audited runs: {len(R)}")


def m90(curve):
    hit = [m for m, a in zip(BUDGETS, curve) if a >= 0.9]
    return hit[0] if hit else CENSOR


def grid(r, key="agree_test"):
    """(budgets, lr, seed, ckpt) array."""
    return np.array([r["budgets"][str(m)]["grid"][key] for m in BUDGETS])


def idx(r):
    return r["lrs"].index(3e-3), r["ckpts"].index(2000), r["ckpts"].index(10000)


def curve_seed(r, s, ckpt=2000, lr=3e-3):
    g = grid(r)
    return g[:, r["lrs"].index(lr), s, r["ckpts"].index(ckpt)]


def curve_tuned(r):
    """Per budget: the (lr, seed, ckpt) with the highest selection agreement
    (first in grid order on ties), scored on test."""
    gs, gt = grid(r, "agree_sel"), grid(r, "agree_test")
    out = []
    for b in range(len(BUDGETS)):
        k = int(np.argmax(gs[b].reshape(-1)))
        out.append(gt[b].reshape(-1)[k])
    return np.array(out)


def curve_oracle(r):
    return grid(r).reshape(len(BUDGETS), -1).max(1)


def h2(score, ds_list, a="redqueen", b="ga"):
    """One-sided Wilcoxon that score(a) > score(b) over paired runs."""
    x = []
    for ds in ds_list:
        for k, r in R.items():
            if k[0] == ds and k[1] == a and (ds, b, k[2], k[3]) in R:
                x.append(score(r) - score(R[(ds, b, k[2], k[3])]))
    x = np.array(x, float)
    if len(x) == 0:
        return None
    w, l = int((x > 0).sum()), int((x < 0).sum())
    p = wilcoxon(x, alternative="greater").pvalue if np.any(x) else 1.0
    return w, l, float(p), len(x)


def fmt(t):
    return "n/a" if t is None else f"{t[0]:2d}/{t[1]:<2d} p={t[2]:.4f} (n={t[3]})"


groups = [(NAMES[ds], [ds]) for ds in DATASETS] + [("Pooled", list(DATASETS))]

# ---------------------------------------------------------------- 1
print("\n1. Reproduction")
acc_diff = [abs(r["acc_stored"] - r["acc_rebuilt"]) for r in R.values()]
ag_diff = [abs(r["agree12_stored"] - r["agree12_rebuilt"]) for r in R.values()]
print(f"max |acc stored - rebuilt| = {max(acc_diff):.2e}; "
      f"max |MLP agreement@12 stored - rebuilt| = {max(ag_diff):.2e}")

# ---------------------------------------------------------------- 2
print("\n2. Convergence of the original setting (lr 3e-3), mean over runs and seeds")
print("budget | fit R2 @2000 | fit R2 @10000 | MSE ratio 10000/2000 | test agree @2000 -> @10000 "
      "(mean change, 5th-95th pct of change)")
for bi, M in enumerate(BUDGETS):
    r2a, r2b, ratio, d = [], [], [], []
    for r in R.values():
        li, c2, c10 = idx(r)
        f = np.array(r["budgets"][str(M)]["grid"]["fit_r2"])[li]
        mse = np.array(r["budgets"][str(M)]["grid"]["fit_mse"])[li]
        t = np.array(r["budgets"][str(M)]["grid"]["agree_test"])[li]
        r2a += list(f[:, c2]); r2b += list(f[:, c10])
        ratio += list(mse[:, c10] / mse[:, c2]); d += list(t[:, c10] - t[:, c2])
    d = np.array(d)
    print(f"M={M:3d} | {np.mean(r2a):.3f} | {np.mean(r2b):.3f} | {np.median(ratio):.2f} | "
          f"{100 * d.mean():+.2f}pp [{100 * np.quantile(d, .05):+.2f},{100 * np.quantile(d, .95):+.2f}]")
# M90 at 10000 vs 2000 steps
same = sum(m90(curve_seed(r, 0, 10000)) == m90(curve_seed(r, 0, 2000)) for r in R.values())
lower = sum(m90(curve_seed(r, 0, 10000)) < m90(curve_seed(r, 0, 2000)) for r in R.values())
print(f"M90 (seed 0) unchanged from 2000 to 10000 steps in {same}/{len(R)} heads, lower in {lower}")
for name, dl in groups:
    print(f"  H2 vs GA at 10000 steps (seed 0) {name:12s}: "
          f"{fmt(h2(lambda r: m90(curve_seed(r, 0, 10000)), dl))}")

# ---------------------------------------------------------------- 3
print("\n3. Initialisation (lr 3e-3, 2000 steps, seeds 0-4)")
sd = np.array([[np.std([curve_seed(r, s)[bi] for s in range(5)], ddof=1) for bi in range(len(BUDGETS))]
               for r in R.values()])
print("mean sd of test agreement over seeds per budget (pp): "
      + " ".join(f"M={M}:{100 * v:.2f}" for M, v in zip(BUDGETS, sd.mean(0))))
spread = [len({m90(curve_seed(r, s)) for s in range(5)}) for r in R.values()]
print(f"heads whose M90 is the same for all five seeds: {sum(x == 1 for x in spread)}/{len(R)}")
for other in ("ga", "random"):
    for name, dl in groups:
        res = [h2(lambda r, s=s: m90(curve_seed(r, s)), dl, b=other) for s in range(5)]
        if res[0] is None:
            continue
        ps = [x[2] for x in res]
        print(f"  H2 vs {other:6s} {name:12s}: per-seed W/L "
              + ", ".join(f"{x[0]}/{x[1]}" for x in res)
              + f" | p range [{min(ps):.4f}, {max(ps):.4f}]")
    for name, dl in groups:
        print(f"  H2 vs {other:6s} {name:12s} seed-mean curve: "
              f"{fmt(h2(lambda r: m90(np.mean([curve_seed(r, s) for s in range(5)], 0)), dl, b=other))}")

# ---------------------------------------------------------------- 4
print("\n4. Tuned auditor (lr x seed x steps selected on the selection split)")
chg = np.array([m90(curve_tuned(r)) - m90(curve_seed(r, 0)) for r in R.values()])
print(f"M90 tuned vs original-setting seed 0: lower {int((chg < 0).sum())}, same {int((chg == 0).sum())}, "
      f"higher {int((chg > 0).sum())} of {len(chg)}")
gain = np.array([curve_tuned(r) - curve_seed(r, 0) for r in R.values()])
print("mean test-agreement gain of tuning per budget (pp): "
      + " ".join(f"M={M}:{100 * v:+.2f}" for M, v in zip(BUDGETS, gain.mean(0))))
cens_o = sum(m90(curve_seed(r, 0)) == CENSOR for r in R.values())
cens_t = sum(m90(curve_tuned(r)) == CENSOR for r in R.values())
cens_or = sum(m90(curve_oracle(r)) == CENSOR for r in R.values())
print(f"censored: original setting {cens_o}, tuned {cens_t}, oracle {cens_or} (of {len(R)})")
for other in ("ga", "random"):
    for name, dl in groups:
        print(f"  H2 vs {other:6s} {name:12s} tuned: {fmt(h2(lambda r: m90(curve_tuned(r)), dl, b=other))}"
              f" | oracle: {fmt(h2(lambda r: m90(curve_oracle(r)), dl, b=other))}")
print("median M90 per method (original seed 0 / tuned)")
for ds in DATASETS:
    row = []
    for m in ("redqueen", "ga", "random", "hea_L1", "hea_L2"):
        rs = [r for k, r in R.items() if k[0] == ds and k[1] == m]
        if rs:
            row.append(f"{m}:{np.median([m90(curve_seed(r, 0)) for r in rs]):.0f}/"
                       f"{np.median([m90(curve_tuned(r)) for r in rs]):.0f}")
    print(f"  {NAMES[ds]:12s} " + "  ".join(row))

# ---------------------------------------------------------------- 5
print("\n5. Fit-set size (lr 3e-3, 2000 steps; three draws per size, mean agreement)")
big = [ds for ds in ("eurosat", "so2sat", "cifar10")]


def curve_size(r, n):
    if n == "full":
        return np.mean([curve_seed(r, s) for s in range(3)], 0)
    j = r["fit_sizes"].index(n)
    c2 = r["ckpts"].index(2000)
    return np.array([np.mean(np.array(r["budgets"][str(M)]["fit_size"]["agree_test"])[j, :, c2])
                     for M in BUDGETS])


for n in (124, 336, 1000, "full"):
    cens = {ds: sum(m90(curve_size(r, n)) == CENSOR for k, r in R.items() if k[0] == ds)
            for ds in big}
    med = {ds: {m: np.median([m90(curve_size(r, n)) for k, r in R.items() if k[0] == ds and k[1] == m])
                for m in ("redqueen", "ga", "random")} for ds in big}
    print(f"n={n}: censored " + ", ".join(f"{NAMES[ds]} {c}" for ds, c in cens.items())
          + " | median M90 RQ/GA/random: "
          + "; ".join(f"{NAMES[ds]} {v['redqueen']:.0f}/{v['ga']:.0f}/{v['random']:.0f}"
                      for ds, v in med.items()))
    for other in ("ga", "random"):
        for name, dl in [(NAMES[ds], [ds]) for ds in big] + [("3 datasets", big)]:
            print(f"   H2 vs {other:6s} {name:12s}: {fmt(h2(lambda r: m90(curve_size(r, n)), dl, b=other))}")
# mean agreement drop with smaller fit sets
for n in (124, 336, 1000):
    d = np.array([curve_size(r, n) - curve_size(r, "full") for k, r in R.items() if k[0] in big])
    print(f"agreement change at n={n} vs full, per budget (pp): "
          + " ".join(f"M={M}:{100 * v:+.1f}" for M, v in zip(BUDGETS, d.mean(0))))
