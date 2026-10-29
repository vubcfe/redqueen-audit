"""Command line: ``python -m redqueen {extract,run,suite,bench}``.

``suite`` and ``bench`` take ``--shard i/n`` so that ``n`` processes (e.g. two
per GPU) split one job list without overlap.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import yaml

from .experiment import METHODS, run
from .runutil import SEARCH_METHODS, check_methods


def _shard(text: str) -> tuple[int, int]:
    i, n = (int(v) for v in text.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"shard {text!r}: need 0 <= i < n")
    return i, n


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="redqueen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="cache backbone features (EuroSAT or --dataset)")
    ex.add_argument("--root", required=True)
    ex.add_argument("--out", default=None,
                    help="default: features/eurosat_ms_resnet50.npz or features/<dataset>_<arch>.npz")
    ex.add_argument("--dataset", default="eurosat",
                    help="eurosat, so2sat, ucmerced, breastmnist or cifar10")

    r = sub.add_parser("run", help="one (method, budget, seed)")
    r.add_argument("--config", required=True)
    r.add_argument("--method", required=True, choices=METHODS)
    r.add_argument("--budget", type=float, required=True)
    r.add_argument("--seed", type=int, default=0)

    for name, help_ in (("suite", "EuroSAT: methods x budgets x seeds"),
                        ("bench", "qml-benchmarks: datasets x methods x seeds")):
        s = sub.add_parser(name, help=help_)
        s.add_argument("--config", required=True)
        s.add_argument("--skip-existing", action="store_true")
        s.add_argument("--shard", type=_shard, default=(0, 1), help="i/n")

    a = ap.parse_args(argv)
    if a.cmd == "extract":
        if a.dataset == "eurosat":
            from .data import extract
            extract(a.root, a.out or "features/eurosat_ms_resnet50.npz")
        else:
            from .datasets import extract
            extract(a.dataset, a.root, a.out)    # None -> features/<dataset>_<arch>.npz
        return
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.cmd == "bench":
        _bench(cfg, a.skip_existing, a.shard)
        return
    out_dir = cfg.get("out_dir", "results/runs")
    if a.cmd == "run":
        print(_line(run(cfg, a.method, a.budget, a.seed, out_dir)))
        return
    check_methods(cfg["methods"], METHODS)
    jobs = list(itertools.product(cfg["methods"], cfg["budgets"], cfg["seeds"]))
    i, n = a.shard
    for method, budget, seed in jobs[i::n]:
        if a.skip_existing and (Path(out_dir) / f"{method}_b{budget:g}_s{seed}.json").exists():
            continue
        print(f"=== {method} budget={budget} seed={seed}", flush=True)
        print(_line(run(cfg, method, budget, seed, out_dir)), flush=True)


def _bench(cfg: dict, skip_existing: bool, shard: tuple[int, int]) -> None:
    from . import qmlbench
    out_dir = Path(cfg.get("out_dir", "results/qmlbench"))
    check_methods(cfg["methods"], SEARCH_METHODS)
    jobs = [(bench, stem, method, seed)
            for bench, method, seed in itertools.product(cfg["benchmarks"], cfg["methods"],
                                                         cfg["seeds"])
            for stem in bench["stems"]]
    i, n = shard
    for bench, stem, method, seed in jobs[i::n]:
        if skip_existing and (out_dir / f"{method}_{stem}_s{seed}.json").exists():
            continue
        rec = qmlbench.run(cfg, bench["data_dir"], stem, method, seed, cfg["data_root"], out_dir)
        ref = qmlbench.published(cfg["published_root"], bench["results_dir"], stem)
        best = ref.loc[ref["test_mean"].idxmax()] if len(ref) else None
        print(f"{method} {stem} s{seed}: test={rec['test_acc_mean']:.3f} "
              f"val={rec['acc_val']:.3f} gap={rec['gap_search']:.3f} "
              f"params={rec['n_params']} | best published: "
              + (f"{best['model']} {best['test_mean']:.3f}" if best is not None else "n/a")
              + f" ({rec['seconds']:.0f}s)", flush=True)


def _line(rec: dict) -> str:
    if "best" in rec:
        b = rec["best"]
        return (f"{rec['method']}: test acc={b['acc']:.4f} f1={b['f1_macro']:.4f} "
                f"val={b['acc_val']:.4f} params={b['n_params']} ({rec['seconds']:.0f}s)")
    parts = [f"{k}={v['acc']:.4f}" for k, v in rec.items() if isinstance(v, dict) and "acc" in v]
    return f"{rec['method']}: " + " ".join(parts) + f" ({rec['seconds']:.0f}s)"


if __name__ == "__main__":
    main()
