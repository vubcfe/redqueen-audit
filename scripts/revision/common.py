"""Shared helpers for the revision analyses (CMC 96001, round 1).

``rebuild`` recreates the data split of a stored run and retrains its
selected head from the stored genome and seed, exactly as the search did
(``train_host`` with the run's seed), so that auditors can be re-run on the
same head without repeating the search.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from redqueen.data import prepare  # noqa: E402
from redqueen.genome import HostGenome  # noqa: E402
from redqueen.host import accuracy, predict_logits, train_host  # noqa: E402

DATASETS = {"eurosat": "results/paper", "so2sat": "results/paper_so2sat",
            "ucmerced": "results/paper_ucmerced", "breastmnist": "results/paper_breastmnist",
            "cifar10": "results/paper_cifar10"}
QUANTUM = ("redqueen", "ga", "random", "hea_L1", "hea_L2")


def load_runs(run_dir) -> dict:
    """Runs of a results directory (relative to the repository root)."""
    runs = {}
    for f in (ROOT / run_dir).glob("*.json"):
        r = json.loads(f.read_text())
        runs[(r["method"], r["budget"], r["seed"])] = r
    return runs


def data_for(rec: dict, device: str = "cuda") -> dict:
    cfg = rec["config"]
    return prepare(cfg["features"], rec["budget"], rec["seed"], cfg["n_qubits"],
                   cfg.get("view", "resnet"), cfg.get("n_pool_fit", 3000),
                   cfg.get("n_pool_eval", 2000), cfg.get("n_pool_audit", 4000), device=device)


def rebuild(rec: dict, d: dict | None = None):
    """(model, genome, data, reproduced test accuracy) for a stored run."""
    d = d or data_for(rec)
    cfg, b = rec["config"], rec["best"]
    g = HostGenome.from_dict(b["genome"])
    model, _ = train_host(g, d, d["n_classes"], cfg["host_steps"], cfg["host_lr"], rec["seed"])
    acc = accuracy(predict_logits(model, d["X_test"]), d["y_test"])
    return model, g, d, acc
