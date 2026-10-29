"""One run = (method, label budget, seed) -> one JSON record."""

from __future__ import annotations

import time
from pathlib import Path

from . import classical
from .coevo import Search
from .data import prepare
from .genome import fixed_hea
from .host import accuracy, predict_logits, train_host
from .runutil import SEARCH_METHODS, check_methods, search_setup, write_json

METHODS = sorted(set(SEARCH_METHODS) | {"hea_L1", "hea_L2", "classical"})


def _host_summary(model, g, rec: dict, d: dict, do_audit: bool) -> dict:
    logits = predict_logits(model, d["X_test"])
    pred = logits.argmax(1).cpu().numpy()
    out = {"genome": g.to_dict(), "key": g.key(), "n_params": g.n_params(d["n_classes"]),
           "two_qubit_gates": g.two_qubit_gates, "acc_train": rec["acc_train"],
           "acc_val": rec["acc_val"],
           **classical.scores(pred, d["y_test"].cpu().numpy())}
    out["gen_gap"] = out["acc_train"] - out["acc"]
    if do_audit:
        out["audit"] = classical.audit(model, g, d)
    return out


def run(cfg: dict, method: str, budget: float, seed: int, out_dir: str | Path) -> dict:
    check_methods([method], METHODS)
    t0 = time.time()
    d = prepare(cfg["features"], budget, seed, cfg["n_qubits"], cfg.get("view", "resnet"),
                cfg.get("n_pool_fit", 3000), cfg.get("n_pool_eval", 2000),
                cfg.get("n_pool_audit", 4000), keep_raw=(method == "classical"))
    record = {"method": method, "budget": budget, "seed": seed, "sizes": d["sizes"],
              "pca_explained": d["pca_explained"], "config": cfg}

    if method in SEARCH_METHODS:
        lam, mode = search_setup(method, cfg["lam"])
        res = Search(d, cfg, lam, seed).run(mode)
        b = res["best"]
        record["best"] = _host_summary(b["model"], b["genome"], b, d, cfg.get("audit", True))
        record["best"]["gap_search"] = b["gap_final"]
        record["history"] = res["history"]
        # per-host test accuracy is for post-hoc analysis only; selection used acc_val
        record["evaluated"] = [{"key": r["key"], "acc_val": r["acc_val"], "acc_train": r["acc_train"],
                                "gap": r["gap_final"], "n_params": r["n_params"],
                                "descriptor": list(r["genome"].descriptor()), "order": r["order"],
                                "in_final_pop": r["in_final_pop"],
                                "test_acc": accuracy(predict_logits(r["model"], d["X_test"]),
                                                     d["y_test"])}
                               for r in res["all"]]
        record.update(parasites=res["parasites"], archive=res["archive"],
                      rejected_by_gate=res["rejected_by_gate"], search_seconds=res["seconds"])
    elif method in ("hea_L1", "hea_L2"):
        g = fixed_hea(cfg["n_qubits"], int(method[-1]))
        model, rec = train_host(g, d, d["n_classes"], cfg["host_steps"], cfg["host_lr"], seed)
        record["best"] = _host_summary(model, g, rec, d, cfg.get("audit", True))
    else:  # classical
        record["logreg"] = classical.logreg(d)
        record["svm_rbf"] = classical.svm_rbf(d)
        record["linear_probe_full"] = classical.linear_probe_full(d)
        # parameter-matched heads as a bracket over the host sizes the search
        # can produce; each quantum host is compared with the nearest size
        for p in cfg.get("mlp_params", [150, 300, 600]):
            record[f"mlp_p{p}"] = classical.mlp_matched(d, p, cfg["host_steps"], seed=seed)
        for m in cfg.get("rff_m", []):
            record[f"rff_logreg_m{m}"] = classical.rff_logreg(d, m, seed=seed)

    record["seconds"] = time.time() - t0
    write_json(record, Path(out_dir) / f"{method}_b{budget:g}_s{seed}.json")
    return record
