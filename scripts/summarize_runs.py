"""Print one block per EuroSAT run record: test metrics, size, audit M90."""

import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "results/paper")
for f in sorted(run_dir.glob("*.json")):
    r = json.loads(f.read_text())
    if "best" not in r:
        accs = {k: round(v["acc"], 4) for k, v in r.items() if isinstance(v, dict) and "acc" in v}
        print(f"{f.stem}: {accs}")
        continue
    b = r["best"]
    print(f"{f.stem}: test={b['acc']:.4f} f1={b['f1_macro']:.4f} train={b['acc_train']:.3f} "
          f"gen_gap={b['gen_gap']:.3f} params={b['n_params']} 2q={b['two_qubit_gates']} "
          f"search_gap={b.get('gap_search', float('nan')):.3f}")
    print(f"   {b['key']}")
    a = b.get("audit")
    if a and "m90" in a:          # current format
        print(f"   M90={a['m90']}  unbounded={ {k: round(v, 3) for k, v in a['unbounded'].items()} }")
        for fam, curve in a["curves"].items():
            print(f"   {fam:8s}" + " ".join(f"{m}:{v:.2f}" for m, v in curve.items()))
    elif a:                       # pilot format (host-relative budgets)
        print("   audit agree:", {k: round(v["agree"], 3) for k, v in a.items()})
