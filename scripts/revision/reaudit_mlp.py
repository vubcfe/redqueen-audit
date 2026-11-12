"""Revision R1 (Reviewer 4, point 3): adequacy of the MLP auditor and its
sensitivity to the size of the auditor's fit set.

For every stored quantum run the selected head is retrained from its genome
and seed (bit-identical to the stored head in the paper's environment; the
script checks this on the test accuracy and on the original auditor at
M = 12). Then, at every audit budget M, a batch of one-hidden-layer tanh MLPs
of the budget's width is distilled on the head's logits:

* adequacy grid: learning rate {1e-3, 3e-3, 1e-2} x initialisation seed
  {0..4}, full-batch Adam on MSE, evaluated after {1000, 2000, 5000, 10000}
  steps. The original auditor is lr 3e-3, 2000 steps, one initialisation.
  Recorded per model and checkpoint: fit-set MSE and R^2, agreement on the
  audit pool's selection split (never used for fitting) and on test inputs.
* fit-set size (datasets with a 3200-image fit set): random subsets of 124,
  336 and 1000 images (the fit-set sizes of BreastMNIST and UC Merced, and an
  intermediate size), three draws each, original settings (lr 3e-3), same
  checkpoints.

All models of one budget are trained together as a batch with a per-model
learning rate (Adam is element-wise, so each model follows exactly its own
Adam trajectory; the update below is torch.optim.Adam's with default betas,
eps and no weight decay).

Usage (repository root, paper environment):
  python scripts/revision/reaudit_mlp.py DATASET [--shard i/n]
Output: results/revision/reaudit/DATASET/<method>_b<budget>_s<seed>.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATASETS, QUANTUM, ROOT, data_for, load_runs, rebuild  # noqa: E402

from redqueen import classical  # noqa: E402

BUDGETS = (3, 6, 12, 24, 48, 96, 192)
LRS = (1e-3, 3e-3, 1e-2)
SEEDS = (0, 1, 2, 3, 4)
CKPTS = (1000, 2000, 5000, 10000)
FIT_SIZES = (124, 336, 1000)
FIT_DRAWS = 3
LR_ORIG = 3e-3


def init_params(K, d, h, C, seeds, device):
    """nn.Linear's default law, U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weights
    and biases, one generator per model seed."""
    W1, b1, W2, b2 = [], [], [], []
    for s in seeds:
        g = torch.Generator().manual_seed(int(s))
        u = lambda shape, fan: (torch.rand(shape, generator=g) * 2 - 1) / math.sqrt(fan)  # noqa: E731
        W1.append(u((d, h), d)), b1.append(u((1, h), d))
        W2.append(u((h, C), h)), b2.append(u((1, C), h))
    return [torch.stack(p).to(device).requires_grad_() for p in (W1, b1, W2, b2)]


def forward(P, X):
    W1, b1, W2, b2 = P
    return torch.tanh(torch.einsum("nd,kdh->knh", X, W1) + b1) @ W2 + b2


def train_batch(Xf, Yf, Xs, Ys, Xt, Yt, h, lrs, seeds, weights):
    """Train K MLPs (K = len(lrs)) on (Xf, Yf) with per-model sample weights
    ``weights`` (K, N) (1 = in the model's fit set). Returns dict of arrays
    (K, len(CKPTS)): fit_mse, fit_r2, agree_sel, agree_test."""
    K, (N, d), C = len(lrs), Xf.shape, Yf.shape[1]
    P = init_params(K, d, h, C, seeds, Xf.device)
    lr = torch.tensor(lrs, device=Xf.device, dtype=torch.float32)
    m = [torch.zeros_like(p) for p in P]
    v = [torch.zeros_like(p) for p in P]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    w = weights / weights.sum(1, keepdim=True)                     # (K, N)
    out = {k: np.zeros((K, len(CKPTS))) for k in ("fit_mse", "fit_r2", "agree_sel", "agree_test")}
    tgt_sel, tgt_test = Ys.argmax(1), Yt.argmax(1)
    for step in range(1, CKPTS[-1] + 1):
        pred = forward(P, Xf)                                       # (K, N, C)
        per = ((pred - Yf) ** 2).mean(2)                            # (K, N)
        loss = (per * w).sum()                                      # sum of per-model MSEs
        grads = torch.autograd.grad(loss, P)
        with torch.no_grad():
            bc1, bc2 = 1 - b1_ ** step, 1 - b2_ ** step
            for p, g, mm, vv in zip(P, grads, m, v):
                mm.mul_(b1_).add_(g, alpha=1 - b1_)
                vv.mul_(b2_).addcmul_(g, g, value=1 - b2_)
                denom = (vv.sqrt() / math.sqrt(bc2)).add_(eps)
                step_size = (lr / bc1).view(-1, *([1] * (p.dim() - 1)))
                p.sub_(step_size * mm / denom)
        if step in CKPTS:
            j = CKPTS.index(step)
            with torch.no_grad():
                pred = forward(P, Xf)
                per = ((pred - Yf) ** 2).mean(2)
                mse = (per * w).sum(1)
                ym = (Yf.unsqueeze(0) * w.unsqueeze(2)).sum(1, keepdim=True)
                sst = (((Yf - ym) ** 2).mean(2) * w).sum(1)
                out["fit_mse"][:, j] = mse.cpu().numpy()
                out["fit_r2"][:, j] = (1 - mse / sst).cpu().numpy()
                out["agree_sel"][:, j] = (forward(P, Xs).argmax(2) == tgt_sel).float().mean(1).cpu().numpy()
                out["agree_test"][:, j] = (forward(P, Xt).argmax(2) == tgt_test).float().mean(1).cpu().numpy()
    return out


def audit_run(rec, d):
    model, g, d, acc = rebuild(rec, d)
    C = d["n_classes"]
    Xf, Xs, Yf, Ys, Yt, Xt = classical.targets(model, d)
    Yf, Ys, Yt = Yf.float(), Ys.float(), Yt.float()
    # reproduction check: original auditor at M = 12
    h12 = classical.mlp_hidden((2 * 12 + 1) * C, g.n, C)
    a12 = classical.audit_mlp(model, d, h12, T=(Xf, Xs, Yf, Ys, Yt, Xt))["agree"]
    res = {"method": rec["method"], "budget": rec["budget"], "seed": rec["seed"],
           "acc_stored": rec["best"]["acc"], "acc_rebuilt": acc,
           "agree12_stored": rec["best"]["audit"]["curves"]["mlp"]["12"], "agree12_rebuilt": a12,
           "n_fit": len(Xf), "n_sel": len(Xs), "n_test": len(Xt),
           "lrs": LRS, "seeds": SEEDS, "ckpts": CKPTS, "budgets": {}}
    big = len(Xf) >= 3000
    sizes = FIT_SIZES if big else ()
    res["fit_sizes"], res["fit_draws"] = sizes, FIT_DRAWS
    # per-run subsets, nested across sizes within a draw
    perms = [np.random.default_rng([rec["seed"], int(rec["budget"] * 1000), dr]).permutation(len(Xf))
             for dr in range(FIT_DRAWS)]
    for M in BUDGETS:
        h = classical.mlp_hidden((2 * M + 1) * C, g.n, C)
        lrs = [lr for lr in LRS for _ in SEEDS]
        seeds = [s for _ in LRS for s in SEEDS]
        W = [torch.ones(len(Xf)) for _ in lrs]
        for n in sizes:
            for dr in range(FIT_DRAWS):
                w = torch.zeros(len(Xf))
                w[torch.as_tensor(perms[dr][:n])] = 1.0
                W.append(w)
                lrs.append(LR_ORIG)
                seeds.append(100 + dr)
        W = torch.stack(W).to(Xf.device)
        out = train_batch(Xf, Yf, Xs, Ys, Xt, Yt, h, lrs, seeds, W)
        k = len(LRS) * len(SEEDS)
        res["budgets"][str(M)] = {
            "hidden": h,
            "grid": {key: val[:k].reshape(len(LRS), len(SEEDS), len(CKPTS)).tolist()
                     for key, val in out.items()},
            "fit_size": {key: val[k:].reshape(len(sizes), FIT_DRAWS, len(CKPTS)).tolist()
                         for key, val in out.items()} if sizes else None}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=list(DATASETS))
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N runs")
    a = ap.parse_args()
    i, n = (int(v) for v in a.shard.split("/"))
    runs = load_runs(DATASETS[a.dataset])
    keys = sorted((k for k in runs if k[0] in QUANTUM), key=lambda k: (k[1], k[2], k[0]))[i::n]
    if a.limit:
        keys = keys[:a.limit]
    out_dir = ROOT / "results/revision/reaudit" / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = {}
    for key in keys:
        f = out_dir / f"{key[0]}_b{key[1]:g}_s{key[2]}.json"
        if f.exists():
            continue
        t0 = time.time()
        rec = runs[key]
        dk = (rec["budget"], rec["seed"])
        if dk not in cache:
            cache.clear()
            cache[dk] = data_for(rec)
        res = audit_run(rec, cache[dk])
        res["seconds"] = time.time() - t0
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(res))
        tmp.replace(f)
        print(f"{a.dataset} {key}: acc {res['acc_stored']:.4f}/{res['acc_rebuilt']:.4f} "
              f"agree12 {res['agree12_stored']:.4f}/{res['agree12_rebuilt']:.4f} "
              f"({res['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
