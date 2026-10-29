"""Figures and summary numbers for the manuscript, from results/ only.

Usage (from the project root): python paper/make_figures.py
Writes paper/fig_*.pdf and paper/numbers.txt.
"""

import json
from collections import defaultdict
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr, wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, "scripts")
from rules import holm, val_pick  # noqa: E402

OUT = Path("paper")
DATASETS = {"EuroSAT": "results/paper", "So2Sat": "results/paper_so2sat",
            "UC Merced": "results/paper_ucmerced", "BreastMNIST": "results/paper_breastmnist",
            "CIFAR-10": "results/paper_cifar10"}
QUANTUM = ("redqueen", "ga", "random", "hea_L1", "hea_L2")
LABEL = {"redqueen": "RED-QUEEN", "ga": "GA ($\\lambda=0$)", "random": "Random search",
         "hea_L1": "HEA L=1", "hea_L2": "HEA L=2"}
# reference categorical slots 1-3 for the three searched heads, neutral greys for the rest
COLOR = {"redqueen": "#eb6834", "ga": "#2a78d6", "random": "#1baf7a",
         "hea_L1": "#8a8984", "hea_L2": "#c3c2b7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

runs = {}
for ds, d in DATASETS.items():
    for f in Path(d).glob("*.json"):
        r = json.loads(f.read_text())
        runs[(ds, r["method"], r["budget"], r["seed"])] = r
keys = sorted({(k[0], k[2], k[3]) for k in runs})


# parameter-matched classical heads: the MLP bracket around the host sizes
MATCHED = lambda k: k.startswith("mlp_p")


def classical_pick(cr, matched_only=False):
    """Validation-selected classical head (ties averaged)."""
    return val_pick([(v["acc_val"], v["acc"]) for k, v in cr.items()
                     if isinstance(v, dict) and "acc_val" in v and k != "linear_probe_full"
                     and (MATCHED(k) or not matched_only)])


lines = []


def note(s):
    print(s)
    lines.append(s)


# ------------------------------------------------ Fig. 2: accuracy vs MLP M90
fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.9), sharey=False)
jit = np.random.default_rng(0)
for ax, ds in zip(axes, DATASETS):
    for m in ("hea_L2", "hea_L1", "random", "ga", "redqueen"):
        xs, ys = [], []
        for d, b, s in keys:
            if d != ds or (ds, m, b, s) not in runs:
                continue
            r = runs[(ds, m, b, s)]["best"]
            xs.append(r["audit"]["m90"]["mlp"])
            ys.append(100 * (r["acc"] - classical_pick(runs[(ds, "classical", b, s)])))
        jitter = np.exp(jit.uniform(-0.12, 0.12, len(xs)))
        ax.scatter(np.array(xs) * jitter, ys, s=11, color=COLOR[m], label=LABEL[m],
                   edgecolor="white", linewidth=0.4, zorder=3 if m in ("redqueen", "ga") else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([3, 12, 48, 384])
    ax.set_xticklabels(["3", "12", "48", "cens."])
    ax.axhline(0, color=INK2, linewidth=0.7)
    ax.set_title(ds, color=INK)
    ax.set_xlabel("MLP $M_{90}$")
axes[0].set_ylabel("acc. $-$ best classical (pp)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.13))
fig.savefig(OUT / "fig_tradeoff.pdf")
plt.close(fig)

# ------------------------------------------------ Fig. 3: MLP imitation curves
AUD = ["3", "6", "12", "24", "48", "96", "192"]
fig, axes = plt.subplots(1, 5, figsize=(7.1, 1.8), sharey=True)
for ax, ds in zip(axes, DATASETS):
    for m in ("hea_L2", "hea_L1", "random", "ga", "redqueen"):
        cur = np.array([[runs[(ds, m, b, s)]["best"]["audit"]["curves"]["mlp"][a] for a in AUD]
                        for d, b, s in keys if d == ds and (ds, m, b, s) in runs])
        ax.plot([int(a) for a in AUD], cur.mean(0), color=COLOR[m], linewidth=1.6 if m in ("redqueen", "ga") else 1.2,
                marker="o", markersize=2.5, label=LABEL[m])
    ax.axhline(0.9, color=INK2, linewidth=0.7, linestyle=(0, (3, 2)))
    ax.set_xscale("log", base=2)
    ax.set_xticks([3, 12, 48, 192])
    ax.set_xticklabels(["3", "12", "48", "192"])
    ax.set_title(ds, color=INK)
    ax.set_xlabel("auditor budget $M$")
axes[0].set_ylabel("agreement with host")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.14))
fig.savefig(OUT / "fig_curves.pdf")
plt.close(fig)

# ------------------------------------------------ H1 comparisons
def quantum_pick(d, b, s, methods=QUANTUM):
    return val_pick([(runs[(d, m, b, s)]["best"]["acc_val"], runs[(d, m, b, s)]["best"]["acc"])
                     for m in methods])


def h1_diffs(ds, quantum, matched_only):
    out = []
    for d, b, s in keys:
        if ds != "Pooled" and d != ds:
            continue
        q = quantum(d, b, s)
        out.append(100 * (q - classical_pick(runs[(d, "classical", b, s)], matched_only)))
    return np.array(out)


def ci(x, rng):
    boot = [rng.choice(x, len(x)).mean() for _ in range(5000)]
    return np.percentile(boot, [2.5, 97.5])


def wil(x):
    return wilcoxon(x).pvalue if np.any(x) else 1.0


rows = list(DATASETS) + ["Pooled"]
H1 = {}
for matched in (False, True):
    rng = np.random.default_rng(0)
    tag = "matched" if matched else "all"
    for ds in rows:
        x = h1_diffs(ds, quantum_pick, matched)
        lo, hi = ci(x, rng)
        H1[(tag, ds)] = (x.mean(), lo, hi)
        note(f"H1[{tag}] {ds}: best quantum - best {tag} classical = {x.mean():+.2f} pp, "
             f"95% CI [{lo:+.2f}, {hi:+.2f}], quantum better {int((x > 0).sum())}/{len(x)}, "
             f"two-sided Wilcoxon p={wil(x):.4f}")
# no selection on the quantum side: every method against the classical pick
for matched in (False, True):
    tag = "matched" if matched else "all"
    rng = np.random.default_rng(1)
    for m in ("redqueen", "ga", "random", "hea_L1", "hea_L2"):
        cells = []
        for ds in rows:
            x = h1_diffs(ds, lambda d, b, s: runs[(d, m, b, s)]["best"]["acc"], matched)
            lo, hi = ci(x, rng)
            cells.append(f"{ds} {x.mean():+.2f} [{lo:+.2f},{hi:+.2f}] (p={wil(x):.3f})")
        note(f"H1[{tag}] per method {m}: " + "; ".join(cells))
# pre-specified per-budget tests (Holm within dataset)
for ds in DATASETS:
    bud = sorted({b for d, b, s in keys if d == ds})
    ps, ms = [], []
    for b in bud:
        x = np.array([100 * (quantum_pick(d, bb, s) - classical_pick(runs[(d, "classical", bb, s)], False))
                      for d, bb, s in keys if d == ds and bb == b])
        ps.append(wil(x))
        ms.append(x.mean())
    adj = holm(ps)
    note(f"H1 per budget {ds}: " + "; ".join(f"{b:g}: {m:+.2f} pp Holm p={a:.3f}" for b, m, a in zip(bud, ms, adj)))

fig, ax = plt.subplots(figsize=(3.4, 2.1))
for j, (tag, col, lab) in enumerate((("all", "#2a78d6", "any classical head"),
                                     ("matched", "#eb6834", "parameter-matched MLPs"))):
    for i, ds in enumerate(rows):
        mean, lo, hi = H1[(tag, ds)]
        y = len(rows) - 1 - i + (0.14 if j == 0 else -0.14)
        ax.plot([lo, hi], [y, y], color=col, linewidth=1.2)
        ax.plot(mean, y, "o", color=col, markersize=4.5, label=lab if i == 0 else None)
ax.axvline(0, color=INK2, linewidth=0.7)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(rows[::-1])
ax.set_xlabel("best quantum $-$ best classical test acc. (pp)")
ax.grid(axis="y", visible=False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False)
fig.savefig(OUT / "fig_h1.pdf")
plt.close(fig)

# ------------------------------------------------ numbers quoted in the text
# H3 within runs: final-population hosts were all scored against the same
# final parasite panel, so their gaps are comparable; one rho per run
for ds in DATASETS:
    rhos, per = [], defaultdict(list)
    for (d, m, b, s), r in runs.items():
        if d != ds or m not in ("redqueen", "ga", "random"):
            continue
        fp = [(h["gap"], h["test_acc"]) for h in r["evaluated"] if h.get("in_final_pop")]
        g, a = map(np.array, zip(*fp))
        if np.ptp(g) > 0 and np.ptp(a) > 0:
            rhos.append(spearmanr(g, a)[0])
            per[m].append(rhos[-1])
    rhos = np.array(rhos)
    note(f"H3 {ds} per method: " + "; ".join(
        f"{m} median {np.median(v):+.2f}, negative {sum(x < 0 for x in v)}/{len(v)}" for m, v in sorted(per.items())))
    note(f"H3 {ds}: within-run Spearman(gap, test acc) over final population, "
         f"median {np.median(rhos):+.2f}, negative in {int((rhos < 0).sum())}/{len(rhos)} runs, "
         f"Wilcoxon p={wilcoxon(rhos).pvalue:.1e}")

# accuracy vs held-out MLP M90 across the selected heads of every quantum method
for ds in DATASETS:
    for excl in (False, True):
        pts = [(r["best"]["acc"], r["best"]["audit"]["m90"]["mlp"]) for (d, m, b, s), r in runs.items()
               if d == ds and m in QUANTUM and not (excl and m == "hea_L2")]
        a, m90 = map(np.array, zip(*pts))
        rho, p = spearmanr(a, m90)
        note(f"selected heads {ds}{' without HEA L=2' if excl else ''}: Spearman(acc, MLP M90) = "
             f"{rho:+.2f} (p={p:.2g}, n={len(a)})")

for ds in DATASETS:
    u = np.array([(r["best"]["audit"]["unbounded"]["mlp"], r["best"]["audit"]["unbounded"]["rff"])
                  for (d, m, b, s), r in runs.items() if d == ds and m in ("redqueen", "ga", "random")])
    note(f"large budget {ds}: median agreement MLP {np.median(u[:, 0]):.3f}, RFF {np.median(u[:, 1]):.3f}")

for ds in DATASETS:
    cl = [r for (d, m, b, s), r in runs.items() if d == ds and m == "classical"]
    lp = 100 * np.mean([r["linear_probe_full"]["acc"] for r in cl])
    bq = max(100 * np.mean([r["best"]["acc"] for (d, m, b, s), r in runs.items() if d == ds and m == q])
             for q in QUANTUM)
    note(f"linear probe {ds}: {lp:.1f}%, best quantum method mean {bq:.1f}%, gap {lp - bq:.1f} pp")

for ds in DATASETS:
    r = next(runs[k] for k in runs if k[0] == ds and k[1] == "redqueen")
    note(f"{ds}: audit pool {r['sizes']['pool_audit']} -> auditor fit {int(0.8 * r['sizes']['pool_audit'])}")

win, vsz = defaultdict(int), defaultdict(set)
for (d, m, b, s), r in runs.items():
    if m == "classical":
        ks = [k for k, v in r.items() if isinstance(v, dict) and "acc_val" in v and k != "linear_probe_full"]
        top = max(r[k]["acc_val"] for k in ks)
        for k in ks:
            win[k] += r[k]["acc_val"] == top
    else:
        vsz[d].add(r["sizes"]["search_val"])
note("validation-best classical heads (ties counted): " + ", ".join(
    f"{k} {v}" for k, v in sorted(win.items(), key=lambda kv: -kv[1])) + f" | total {sum(win.values())}")
note("search-validation sizes: " + "; ".join(f"{d} {sorted(v)}" for d, v in vsz.items()))
cl = next(r for k, r in runs.items() if k[0] == "EuroSAT" and k[1] == "classical")
note("classical params (EuroSAT): " + ", ".join(f"{k} {v['n_params']}" for k, v in cl.items()
                                               if isinstance(v, dict) and v.get("n_params")))

# Table 3 and other numbers quoted in the text
for ds in DATASETS:
    row = []
    for m in QUANTUM:
        rs = [r for (d, mm, b, s), r in runs.items() if d == ds and mm == m]
        row.append(f"{m} acc {100 * np.mean([r['best']['acc'] for r in rs]):.1f}"
                   f"±{100 * np.std([r['best']['acc'] for r in rs], ddof=1):.1f} "
                   f"F1 {100 * np.mean([r['best']['f1_macro'] for r in rs]):.1f} "
                   f"M90 {np.median([r['best']['audit']['m90']['mlp'] for r in rs]):g} "
                   f"params {min(r['best']['n_params'] for r in rs)}-{max(r['best']['n_params'] for r in rs)}")
    cl = [r for (d, mm, b, s), r in runs.items() if d == ds and mm == "classical"]
    row.append(f"best classical {100 * np.mean([classical_pick(r) for r in cl]):.1f}")
    row.append(f"best matched MLP {100 * np.mean([classical_pick(r, True) for r in cl]):.1f}")
    row.append(f"logreg {100 * np.mean([r['logreg']['acc'] for r in cl]):.1f}")
    f1 = [val_pick([(v["acc_val"], v["f1_macro"]) for k, v in r.items() if MATCHED(k)]) for r in cl]
    row.append(f"matched MLP F1 {100 * np.mean(f1):.1f}")
    row.append("RFF params " + "/".join(str(cl[0][f'rff_logreg_m{m}']['n_params']) for m in (8, 16, 32, 64)))
    note(f"Table 3 {ds}: " + " | ".join(row))
    sole = defaultdict(int)
    for d, b, s in keys:
        if d != ds:
            continue
        top = max(runs[(d, m, b, s)]["best"]["acc_val"] for m in QUANTUM)
        tops = [m for m in QUANTUM if runs[(d, m, b, s)]["best"]["acc_val"] == top]
        if len(tops) == 1:
            sole[tops[0]] += 1
    dm = [runs[(d, "redqueen", b, s)]["best"]["audit"]["m90"]["mlp"] - runs[(d, "ga", b, s)]["best"]["audit"]["m90"]["mlp"]
          for d, b, s in keys if d == ds]
    note(f"   {ds}: sole validation-best quantum head {dict(sole)}; median MLP M90 RED-QUEEN - GA {np.median(dm):+g}")

# qml-benchmarks, H2 on the 18 (task, seed) pairs
qml = {}
for f in Path("results/qmlbench_paper").glob("*.json"):
    r = json.loads(f.read_text())
    qml[(r["method"], r["stem"], r["seed"])] = r
qpairs = sorted({(k[1], k[2]) for k in qml})
for other in ("ga", "random"):
    for fam in ("mlp", "fourier", "rff"):
        a = np.array([qml[("redqueen", t, s)]["audit"]["m90"][fam] for t, s in qpairs], dtype=float)
        c = np.array([qml[(other, t, s)]["audit"]["m90"][fam] for t, s in qpairs], dtype=float)
        p = wilcoxon(a, c, alternative="greater").pvalue if np.any(a - c) else 1.0
        note(f"qml H2 vs {other} {fam}: n={len(a)} wins {int((a > c).sum())} losses {int((a < c).sum())} p={p:.3f}")

fails = [(k, runs[k]["best"]["audit"]["unbounded"]) for k in runs
         if k[1] in QUANTUM and min(runs[k]["best"]["audit"]["unbounded"]["rff"],
                                    runs[k]["best"]["audit"]["unbounded"]["mlp"]) < 0.95]
note(f"large-budget check: {len(fails)} of {sum(k[1] in QUANTUM for k in runs)} quantum runs below 0.95")
flag = defaultdict(int)
for k, _ in fails:
    flag[(k[0], k[1])] += 1
for ds in DATASETS:
    note(f"   flagged {ds}: " + ", ".join(f"{m} {flag[(ds, m)]}/15" for m in QUANTUM))
for k, u in fails:
    note(f"   {k} rff={u['rff']:.3f} mlp={u['mlp']:.3f}")

cens = defaultdict(lambda: [0, 0])
for k, r in runs.items():
    if k[1] in QUANTUM:
        cens[k[1]][0] += r["best"]["audit"]["m90_censored"]["mlp"]
        cens[k[1]][1] += 1
note("MLP M90 censored: " + ", ".join(f"{m} {c}/{n}" for m, (c, n) in cens.items()))

stats = defaultdict(list)
for (d, m, b, s), r in runs.items():
    if m in ("redqueen", "ga", "random"):
        stats[m].append((r["best"]["n_params"], r["best"]["two_qubit_gates"], r["best"]["genome"]["layers"],
                         r["best"]["genome"]["bandwidth"], r["search_seconds"], r.get("rejected_by_gate", 0),
                         len(r["evaluated"])))
for m, v in stats.items():
    v = np.array(v, dtype=float)
    note(f"{m}: params {v[:, 0].mean():.0f}, 2q gates {v[:, 1].mean():.1f}, layers {v[:, 2].mean():.2f}, "
         f"bandwidth median {np.median(v[:, 3]):g}, search {v[:, 4].mean() / 60:.1f} min, "
         f"gate rejections {v[:, 5].mean():.1f}, evaluated {v[:, 6].mean():.0f}")
for m in ("redqueen", "ga", "random"):
    bw = defaultdict(int)
    for (d, mm, b, s), r in runs.items():
        if mm == m:
            bw[r["best"]["genome"]["bandwidth"]] += 1
    note(f"{m} bandwidth counts: {dict(sorted(bw.items()))}")

for ds in DATASETS:
    r = next(runs[k] for k in runs if k[0] == ds and k[1] == "redqueen")
    note(f"{ds}: sizes {r['sizes']} PCA explained {r['pca_explained']:.2f}")

# spectrum enumeration confound: inputs covered by the first 8 frequencies
sys.path.insert(0, "src")
from redqueen.surrogate import spectrum  # noqa: E402
for L in (2, 3):
    sp = spectrum(8, L, 3)
    note(f"enumeration order, L={L}: first 8 of {len(sp)} frequencies cover "
         f"{int((np.abs(sp[:8]).sum(0) > 0).sum())} of 8 inputs")
for m in ("ga", "redqueen"):
    a = json.loads(Path(f"results/pilot_v2/{m}_b0.05_s0.json").read_text())["best"]["audit"]
    note(f"pilot v2 {m}: white-box agreement M=8 {a['fourier_m8']['agree']:.3f}, M=32 {a['fourier_m32']['agree']:.3f}")

(OUT / "numbers.txt").write_text("\n".join(lines) + "\n")
