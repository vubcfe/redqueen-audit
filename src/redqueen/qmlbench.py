"""RED-QUEEN on the qml-benchmarks suite (Bowles, Ahmed & Schuld, 2024).

Follows the protocol of ``scripts/score_with_best_hyperparameters.py`` in
XanaduAI/qml-benchmarks v0.1:

- the train/test CSV files produced by ``paper/benchmarks/generate_*.py``
  (seed 42), labels in {-1, 1};
- ``MinMaxScaler(feature_range=(-pi/2, pi/2))`` fitted on train only;
- model selection on train only (they use 5-fold GridSearchCV; RED-QUEEN
  selects on an internal 80/20 split of train);
- the selected configuration is refitted on the full train set with
  ``random_state`` 0..4 and test accuracy is reported for each.

Parasites see only training inputs: the imitation pools are the internal
train/val inputs plus jittered copies. The auditors get their own pool of
jittered copies of the train inputs drawn with a separate generator, so no
auditor input is a search-pool input. No test input is used before scoring.

The audit is run on the refit model with ``random_state=0``, i.e. on one of
the five models whose test accuracy is reported. ``gap_search`` describes the
search-time model (trained on the internal 80 % split) and is reported as
such.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from .classical import audit
from .coevo import Search
from .genome import HostGenome
from .host import accuracy, predict_logits, train_host
from .runutil import search_setup, write_json


def read_csv(path) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",")
    return a[:, :-1], a[:, -1]


def _pool(X: np.ndarray, copies: int, jitter: float, rng, keep_original: bool = True) -> np.ndarray:
    noisy = [X + rng.normal(0, jitter, X.shape) for _ in range(copies)]
    return np.concatenate(([X] if keep_original else []) + noisy)


def _audit_pool(X: np.ndarray, copies: int, jitter: float, rng) -> tuple[np.ndarray, int]:
    """Jittered copies only: the fit part comes from 80 % of the train points
    and the selection part from the other points, so no point has copies on
    both sides. Returns the pool and the row index where the parts meet
    (passed on as ``audit_split`` to ``classical._split_pool``)."""
    idx = rng.permutation(len(X))
    k = int(round(0.8 * len(X)))
    fit = _pool(X[idx[:k]], copies, jitter, rng, keep_original=False)
    sel = _pool(X[idx[k:]], copies, jitter, rng, keep_original=False)
    return np.concatenate([fit, sel]), len(fit)


def prepare(train_csv, test_csv, seed: int, val_frac: float = 0.2, copies: int = 8,
            jitter: float = 0.05, device: str = "cuda") -> dict:
    Xtr, ytr = read_csv(train_csv)
    Xte, yte = read_csv(test_csv)
    scaler = MinMaxScaler(feature_range=(-np.pi / 2, np.pi / 2)).fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
    to01 = lambda y: (y > 0).astype(int)  # noqa: E731
    ytr, yte = to01(ytr), to01(yte)
    i_tr, i_val = train_test_split(np.arange(len(Xtr)), test_size=val_frac, stratify=ytr,
                                   random_state=seed)
    rng = np.random.default_rng(seed)
    span = np.pi * jitter
    t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
    ty = lambda a: torch.tensor(a, dtype=torch.long, device=device)  # noqa: E731
    audit_pool, audit_split = _audit_pool(Xtr, 2 * copies, span, np.random.default_rng([seed, 1]))
    return {"X_tr": t(Xtr[i_tr]), "y_tr": ty(ytr[i_tr]), "X_val": t(Xtr[i_val]),
            "y_val": ty(ytr[i_val]), "X_full": t(Xtr), "y_full": ty(ytr),
            "X_test": t(Xte), "y_test": ty(yte),
            "X_pool_fit": t(_pool(Xtr[i_tr], copies, span, rng)),
            "X_pool_eval": t(_pool(Xtr[i_val], copies, span, rng)),
            "X_pool_audit": t(audit_pool), "audit_split": audit_split,
            "n_classes": 2, "n_features": Xtr.shape[1]}


def refit_and_score(g: HostGenome, d: dict, cfg: dict, repeats: int = 5) -> tuple[dict, list]:
    """Their scoring loop: refit on the full train set with seeds 0..4.
    Returns the scores and the refit models."""
    full = {"X_tr": d["X_full"], "y_tr": d["y_full"], "X_val": d["X_val"], "y_val": d["y_val"]}
    train_acc, test_acc, models = [], [], []
    for i in range(repeats):
        model, rec = train_host(g, full, 2, cfg["host_steps"], cfg["host_lr"], seed=i)
        train_acc.append(rec["acc_train"])
        test_acc.append(accuracy(predict_logits(model, d["X_test"]), d["y_test"]))
        models.append(model)
    return {"train_acc": train_acc, "test_acc": test_acc}, models


def run(cfg: dict, data_dir: str, stem: str, method: str, seed: int, data_root, out_dir) -> dict:
    """``data_dir``/``stem`` name the files written by the generator scripts,
    e.g. ``two_curves_diff``/``two_curves-5degree-0.1offset-4d``."""
    lam, mode = search_setup(method, cfg["lam"])
    t0 = time.time()
    root = Path(data_root) / data_dir
    d = prepare(root / f"{stem}_train.csv", root / f"{stem}_test.csv", seed)
    cfg = {**cfg, "n_qubits": d["n_features"]}
    res = Search(d, cfg, lam, seed).run(mode)
    best = res["best"]
    scores, refits = refit_and_score(best["genome"], d, cfg)
    rec = {"data_dir": data_dir, "stem": stem, "n_features": d["n_features"],
           "method": method, "seed": seed,
           "genome": best["genome"].to_dict(), "key": best["key"], "acc_val": best["acc_val"],
           "gap_search": best["gap_final"], "n_params": best["n_params"],
           "two_qubit_gates": best["genome"].two_qubit_gates, **scores,
           "audit": audit(refits[0], best["genome"], d) if cfg.get("audit", True) else None,
           "history": res["history"],
           "evaluated": [{"key": r["key"], "acc_val": r["acc_val"], "gap": r["gap_final"],
                          "in_final_pop": r["in_final_pop"]} for r in res["all"]],
           "search_seconds": res["seconds"], "seconds": time.time() - t0, "config": cfg}
    rec["test_acc_mean"] = float(np.mean(rec["test_acc"]))
    write_json(rec, Path(out_dir) / f"{method}_{stem}_s{seed}.json")
    return rec


def published(results_root, results_dir: str, stem: str) -> pd.DataFrame:
    """Mean/std test accuracy of every model in the paper's results folder.

    Quantum models and the MLP sit in one sub-folder per model; SVC results
    sit directly in the benchmark folder."""
    suffix = f"_{stem}_GridSearchCV-best-hyperparams-results.csv"
    rows = []
    for f in sorted(Path(results_root, results_dir).glob(f"**/*{suffix}")):
        acc = pd.read_csv(f)["test_acc"]
        rows.append({"model": f.name[: -len(suffix)], "test_mean": acc.mean(),
                     "test_std": acc.std()})
    return pd.DataFrame(rows)
