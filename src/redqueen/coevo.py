"""Red Queen co-evolution of quantum hosts and classical parasites.

One generation:
1. Every host in the population is trained once (results are cached by
   genome key) and its logits on the unlabeled pools are stored.
2. Every parasite in the current population and in the hall of fame is
   fitted to every host at every budget ``m`` of the panel (the same panel
   for all hosts). For each ``m`` the best parasite counts; a host's gap is
   ``mean_m (1 - max_parasite agreement)``, an area over the imitation curve,
   so a host cannot win by dodging one budget. Its fitness is
   ``acc_val + lam * gap``. Gaps are recomputed every generation, because
   the parasites move.
3. Parasite fitness is the mean agreement over current hosts and budgets. The best half
   survives, the rest are mutants of survivors. The generation's best parasite
   joins the hall of fame (anti-cycling memory).
4. Hosts enter a MAP-Elites archive keyed by (depth, edge bin). Parents are
   drawn by tournament from the population and the archive elites. Offspring
   must pass the trainability gate.

``lam = 0`` gives a plain GA with identical machinery (the main ablation).
``mode = "random"`` replaces tournament + crossover + mutation by fresh gated
random genomes and keeps everything else (budget, co-evolving parasites,
selection, fitness), which isolates the variation operator.
"""

from __future__ import annotations

import time

import numpy as np

from .genome import HostGenome, crossover, mutate, random_genome
from .host import gradient_variance, predict_logits, train_host
from .surrogate import budgets, imitate, mutate_parasite, random_parasite


class Search:
    def __init__(self, data: dict, cfg: dict, lam: float, seed: int, log=print):
        self.data, self.cfg, self.lam, self.seed, self.log = data, cfg, lam, seed, log
        self.rng = np.random.default_rng(seed)
        self.C = int(data["n_classes"])
        self.cache: dict[str, dict] = {}   # genome key -> record (incl. pool logits)
        self._gate: dict[str, bool] = {}   # genome key -> passed the trainability gate
        self._agree: dict[tuple, float] = {}
        self.rejected = 0

    # ----------------------------------------------------------------- hosts
    def gated_random(self, max_tries: int = 1000) -> HostGenome:
        for _ in range(max_tries):
            g = random_genome(self.cfg["n_qubits"], self.rng)
            if self.passes_gate(g):
                return g
        raise RuntimeError(f"no random genome passed the trainability gate in {max_tries} "
                           f"tries (gate_threshold={self.cfg.get('gate_threshold')})")

    def passes_gate(self, g: HostGenome) -> bool:
        """Trainability gate, memoised per genome; ``rejected`` counts distinct
        rejected genomes."""
        thr = self.cfg.get("gate_threshold", 0.0)
        if thr <= 0:
            return True
        k = g.key()
        if k not in self._gate:
            v = gradient_variance(g, self.data["X_tr"][:64],
                                  draws=self.cfg.get("gate_draws", 8), seed=self.seed)
            self._gate[k] = v >= thr
            self.rejected += not self._gate[k]
        return self._gate[k]

    def evaluate(self, g: HostGenome) -> dict:
        k = g.key()
        if k not in self.cache:
            model, rec = train_host(g, self.data, self.C, self.cfg["host_steps"],
                                    self.cfg["host_lr"], seed=self.seed)
            rec.update(genome=g, key=k, n_params=g.n_params(self.C),
                       Y_fit=predict_logits(model, self.data["X_pool_fit"]),
                       Y_eval=predict_logits(model, self.data["X_pool_eval"]),
                       model=model, order=len(self.cache))
            self.cache[k] = rec
        return self.cache[k]

    # ------------------------------------------------------------- parasites
    def agreements(self, rec: dict, p) -> list[float]:
        """Agreement of parasite ``p`` with a host at each of its budgets
        (memoised: hosts and parasites are immutable once created)."""
        ms = budgets(p, self.cfg)
        todo = [m for m in ms if (rec["key"], p.key(), m) not in self._agree]
        if todo:
            got = imitate(p, rec["genome"], todo, self.data["X_pool_fit"], rec["Y_fit"],
                          self.data["X_pool_eval"], rec["Y_eval"])
            for m, (a, _) in zip(todo, got):
                self._agree[(rec["key"], p.key(), m)] = a
        return [self._agree[(rec["key"], p.key(), m)] for m in ms]

    def score(self, recs: list[dict], panel: list) -> np.ndarray:
        """Set ``gap`` and ``fitness`` on every record; return the agreement
        array (hosts x parasites x budgets)."""
        A = np.array([[self.agreements(r, p) for p in panel] for r in recs])
        for r, a in zip(recs, A):
            r["gap"] = float((1.0 - a.max(0)).mean())
            r["fitness"] = r["acc_val"] + self.lam * r["gap"]
        return A

    # ---------------------------------------------------------------- search
    def run(self, mode: str = "evolve") -> dict:
        """``evolve``: offspring by tournament, crossover and mutation.
        ``random``: offspring are fresh gated random genomes; everything else
        (generations, co-evolving parasites, hall of fame, selection, fitness)
        is identical, so the two modes differ only in the variation operator."""
        if mode not in ("evolve", "random"):
            raise ValueError(f"unknown mode {mode!r}")
        cfg, t0 = self.cfg, time.time()
        pop_size, gens = cfg["pop_size"], cfg["generations"]
        kinds = tuple(cfg.get("parasite_kinds", ("fourier", "rff")))
        # with a budget panel m_frac has no effect, so it must not be mutated
        mutate_m_frac = not cfg.get("parasite_m_panel")
        parasites: list = []
        while len(parasites) < cfg["parasite_pop"]:
            p = random_parasite(self.rng, kinds)
            if p.key() not in {q.key() for q in parasites}:
                parasites.append(p)
        hof: list = []
        archive: dict[tuple, dict] = {}
        history = []

        pop: list[dict] = []
        while len(pop) < pop_size:          # pop_size distinct hosts (equal budgets)
            g = self.gated_random()
            if g.key() not in self.cache:
                pop.append(self.evaluate(g))
        panel = list(parasites)
        A = self.score(pop, panel)

        for gen in range(gens + 1):
            for r in pop:
                cell = r["genome"].descriptor()
                if cell not in archive or r["fitness"] > archive[cell]["fitness"]:
                    archive[cell] = r
            best = max(pop, key=lambda r: r["fitness"])
            history.append({"gen": gen, "evaluated": len(self.cache),
                            "best_fitness": best["fitness"], "best_acc_val": best["acc_val"],
                            "best_gap": best["gap"],
                            "mean_acc_val": float(np.mean([r["acc_val"] for r in pop])),
                            "mean_gap": float(np.mean([r["gap"] for r in pop])),
                            "parasite_mean_agree": float(A[:, :len(parasites)].mean()),
                            "archive_cells": len(archive), "seconds": time.time() - t0})
            self.log(f"[gen {gen:2d}] eval={len(self.cache):3d} best acc={best['acc_val']:.3f} "
                     f"gap={best['gap']:.3f} | mean acc={history[-1]['mean_acc_val']:.3f} "
                     f"gap={history[-1]['mean_gap']:.3f} cells={len(archive)}")
            if gen == gens:
                break

            # parasites: best half (mean agreement over hosts and budgets) survives,
            # mutants refill; the best joins the hall of fame (anti-cycling memory)
            p_fit = A[:, :len(parasites)].mean((0, 2))
            order = np.argsort(-p_fit)
            survivors = [parasites[i] for i in order[: max(1, len(parasites) // 2)]]
            if cfg["hof_size"]:
                hof = [p for p in hof if p.key() != survivors[0].key()] + [survivors[0]]
                hof = hof[-cfg["hof_size"]:]
            # keys stay unique so that the first len(parasites) columns of the
            # agreement array are exactly the current parasites
            n_par, seen = len(parasites), {p.key() for p in survivors}
            parasites, i = list(survivors), 0
            while len(parasites) < n_par:
                child = mutate_parasite(survivors[i % len(survivors)], self.rng, kinds,
                                        mutate_m_frac=mutate_m_frac)
                i += 1
                if child.key() not in seen:
                    seen.add(child.key())
                    parasites.append(child)
            panel = parasites + [p for p in hof if p.key() not in seen]

            # re-score the population and the archive elites against the new
            # panel (memoised), so tournaments and archive cells compare
            # current fitness, not fitness against parasites that are gone
            elites = list(archive.values()) if cfg.get("use_archive", True) else []
            self.score(pop + elites, panel)

            # hosts: offspring, then (mu + lambda) selection against the new panel
            pool = pop + elites
            pool = list({r["key"]: r for r in pool}.values())  # pop elites appear once
            children, attempts = [], 0
            while len(children) < cfg["offspring"] and attempts < 50 * cfg["offspring"]:
                attempts += 1
                if mode == "random":
                    child = random_genome(cfg["n_qubits"], self.rng)
                else:
                    pa, pb = (self._tournament(pool) for _ in range(2))
                    child = crossover(pa["genome"], pb["genome"], self.rng) \
                        if self.rng.random() < cfg["crossover_rate"] else pa["genome"]
                    child = mutate(child, self.rng)
                if child.key() not in self.cache and self.passes_gate(child):
                    children.append(self.evaluate(child))
            # a converged population can keep regenerating cached genomes; fill
            # the shortfall with gated random immigrants so that every method
            # evaluates exactly the same number of hosts
            immigrants = 0
            while len(children) < cfg["offspring"]:
                g = self.gated_random()
                if g.key() not in self.cache:
                    children.append(self.evaluate(g))
                    immigrants += 1
            history[-1]["immigrants"] = immigrants
            candidates = pop + children
            A_all = self.score(candidates, panel)
            keep = np.argsort([-r["fitness"] for r in candidates], kind="stable")[:pop_size]
            pop = [candidates[i] for i in keep]
            A = A_all[keep]

        # The reported host comes from the final population: only those hosts
        # were attacked by parasites that co-evolved against them. Other hosts
        # keep the gap from the last generation they were scored in
        # (``in_final_pop`` False marks it); they are not re-scored against
        # parasites that never adapted to them.
        # ``pop`` was last scored against ``panel`` in the selection step
        # (or at initialisation), so its gaps are already the final ones
        final_panel = panel
        everyone = list(self.cache.values())
        for r in everyone:
            r["gap_final"], r["fitness_final"] = r["gap"], r["fitness"]
            r["in_final_pop"] = False
        for r in pop:
            r["in_final_pop"] = True
        best = max(pop, key=lambda r: r["fitness_final"])
        return {"best": best, "all": everyone, "history": history,
                "parasites": [p.to_dict() for p in final_panel],
                "archive": {str(k): v["key"] for k, v in archive.items()},
                "rejected_by_gate": self.rejected, "seconds": time.time() - t0}

    def _tournament(self, pool: list[dict], k: int = 3) -> dict:
        idx = self.rng.choice(len(pool), size=min(k, len(pool)), replace=False)
        return max((pool[i] for i in idx), key=lambda r: r["fitness"])
