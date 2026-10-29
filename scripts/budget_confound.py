"""Re-audit the selected hosts of two run folders under absolute surrogate budgets.

Shows whether a gap measured with host-relative budgets survives when every
host faces the same number of surrogate features.
Usage: python scripts/budget_confound.py results/pilot results/pilot_v2
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from redqueen import classical  # noqa: E402
from redqueen.data import prepare  # noqa: E402
from redqueen.genome import HostGenome  # noqa: E402
from redqueen.host import train_host  # noqa: E402
from redqueen.surrogate import max_features  # noqa: E402

BUDGETS = (2, 4, 8, 16, 32)
for run_dir in sys.argv[1:]:
    for method in ("ga", "redqueen"):
        r = json.loads((Path(run_dir) / f"{method}_b0.05_s0.json").read_text())
        cfg, g = r["config"], HostGenome.from_dict(r["best"]["genome"])
        # same features and pools as the run being re-audited
        d = prepare(cfg["features"], r["budget"], r["seed"], cfg["n_qubits"],
                    cfg.get("view", "resnet"), cfg.get("n_pool_fit", 3000),
                    cfg.get("n_pool_eval", 2000), cfg.get("n_pool_audit", 4000))
        C = d["n_classes"]
        model, _ = train_host(g, d, C, cfg["host_steps"], cfg["host_lr"], r["seed"])
        m_rel = max_features(g, C, 1.0)
        row = {m: classical.audit_fourier(model, g, d, m)["agree"] for m in BUDGETS}
        print(f"{run_dir:18s} {method:8s} P={g.n_params(C):3d} relative budget M={m_rel:2d} | "
              + "  ".join(f"M={m}: {a:.3f}" for m, a in row.items()) + f" | {g.key()}")
