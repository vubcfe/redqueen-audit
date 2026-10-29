import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from redqueen import sim
from redqueen.genome import HostGenome, crossover, mutate, random_genome
from redqueen.host import QuantumHead, gradient_variance
from redqueen.classical import m_at
from redqueen.coevo import Search
from redqueen.surrogate import (ParasiteGenome, budgets, features, imitate, mutate_parasite,
                                prioritised, random_parasite, ridge, spectrum)

# a Python interpreter with PennyLane installed, for the simulator cross-check
# (e.g. REDQUEEN_PL_PYTHON=$(which python) when PennyLane is in this env)
PL_PYTHON = Path(os.environ.get("REDQUEEN_PL_PYTHON", "/nonexistent"))


def _expect_z(psi, n):
    return sim.probabilities(psi) @ sim.z_signs(n, psi.device).T


def test_rotation_expectations():
    for theta in [0.0, 0.3, 1.7, math.pi]:
        t = torch.tensor(theta)
        for gate, expected in [("RX", math.cos(theta)), ("RY", math.cos(theta)), ("RZ", 1.0)]:
            psi = sim.apply_1q(sim.zero_state(1, 1, "cpu"), sim.ROT[gate](t), 0)
            assert _expect_z(psi, 1).item() == pytest.approx(expected, abs=1e-6)


def test_cnot_truth_table_and_bell_state():
    n = 3
    for basis in range(2 ** n):
        psi = torch.zeros(1, 2 ** n, dtype=sim.CDTYPE)
        psi[0, basis] = 1
        out = sim.apply_cnot(psi.view((1,) + (2,) * n), 0, 2).reshape(-1)
        b = [(basis >> (n - 1 - i)) & 1 for i in range(n)]
        b[2] ^= b[0]
        assert out.abs().argmax().item() == sum(v << (n - 1 - i) for i, v in enumerate(b))
    psi = sim.apply_1q(sim.zero_state(1, 2, "cpu"), sim.ry(torch.tensor(math.pi / 2)), 0)
    psi = sim.apply_cnot(psi, 0, 1)
    p = sim.probabilities(psi)[0]
    assert p[0].item() == pytest.approx(0.5, abs=1e-6) and p[3].item() == pytest.approx(0.5, abs=1e-6)


def test_batched_gate_matches_shared_gate():
    psi = sim.zero_state(4, 3, "cpu")
    t = torch.full((4,), 0.7)
    a = sim.apply_1q(psi, sim.rx(t), 1)
    b = sim.apply_1q(psi, sim.rx(torch.tensor(0.7)), 1)
    assert torch.allclose(a, b, atol=1e-6)


def test_genome_operators_keep_constraints():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = random_genome(6, rng)
        g = crossover(mutate(g, rng), random_genome(6, rng), rng)
        assert len(g.edges) <= 12 and all(a != b for a, b in g.edges)
        assert HostGenome.from_dict(g.to_dict()) == g


@pytest.mark.skipif(not PL_PYTHON.exists(), reason="set REDQUEEN_PL_PYTHON to a Python with PennyLane")
def test_simulator_matches_pennylane():
    rng = np.random.default_rng(1)
    for trial in range(6):
        g = random_genome(4, rng)
        model = QuantumHead(g, 3, seed=trial)
        x = torch.tensor(rng.normal(size=(5, 4)), dtype=torch.float32)
        ours = model.expectations(x).detach().numpy()
        job = {"genome": g.to_dict(), "x": x.tolist(), "theta": model.theta.detach().tolist()}
        out = subprocess.run([str(PL_PYTHON), str(Path(__file__).parent / "pl_reference.py")],
                             input=json.dumps(job), capture_output=True, text=True, check=True)
        ref = np.array(json.loads(out.stdout))
        assert np.allclose(ours, ref, atol=1e-5), (g.key(), ours, ref)


def test_fourier_spectrum_is_exact():
    """A full-spectrum Fourier ridge reproduces a host's logits exactly
    (Schuld et al. 2021): the white-box parasite family is complete."""
    g = HostGenome(n=2, enc_gate="RX", bandwidth=1.0, layers=2, rot="RZRY",
                   edges=((0, 1),), ent_gate="CNOT", readout="ZZ")
    model = QuantumHead(g, 3, seed=0)
    X = torch.rand(400, 2) * 2 * math.pi - math.pi
    with torch.no_grad():
        Y = model(X)
    W = torch.as_tensor(g.bandwidth * spectrum(2, 2, 2).astype(float))
    beta = ridge(features(X, W), Y.double(), 1e-10)
    resid = features(X, W) @ beta - Y.double()
    assert resid.abs().max().item() < 1e-4


def test_imitation_scores_in_range():
    g = random_genome(4, np.random.default_rng(3))
    model = QuantumHead(g, 10, seed=0)
    Xf, Xe = torch.randn(300, 4), torch.randn(200, 4)
    with torch.no_grad():
        Yf, Ye = model(Xf), model(Xe)
    p = ParasiteGenome("fourier", 1.0, 2, 1.0, 1e-3, seed=0)
    agree, r2 = imitate(p, g, 8, Xf, Yf, Xe, Ye)
    assert 0.0 <= agree <= 1.0 and r2 <= 1.0


def test_budgets_are_host_independent():
    p = ParasiteGenome("rff", 0.5, 2, 1.0, 1e-3, seed=0)
    assert budgets(p, {"parasite_m_panel": [8, 16]}) == [8, 16]
    assert budgets(p, {"parasite_m_abs": 16}) == [8]


def test_prioritised_prefix_covers_every_input_whatever_the_depth():
    """The first n frequencies are the n first harmonics, for any depth L
    (the raw enumeration covered fewer inputs for deeper circuits)."""
    rng = np.random.default_rng(0)
    for layers in (1, 2, 3):
        head = prioritised(spectrum(8, layers, 3), rng)[:8]
        assert (np.abs(head).sum(1) == 1).all()
        assert sorted(np.nonzero(head)[1].tolist()) == list(range(8))


def test_mutated_parasite_always_has_a_new_key():
    rng = np.random.default_rng(0)
    for _ in range(500):
        p = random_parasite(rng)
        assert mutate_parasite(p, rng).key() != p.key()


def test_m_at():
    assert m_at({2: 0.5, 8: 0.95, 4: 0.91}) == 4
    assert m_at({2: 0.5}) is None
    assert m_at({2: 0.5}, censored=384) == 384


def test_search_smoke_keeps_parasite_columns_aligned():
    rng = np.random.default_rng(0)
    n, C = 3, 2

    def t(a):
        return torch.tensor(a, dtype=torch.float32)

    X = rng.normal(size=(160, n))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    data = {"X_tr": t(X[:80]), "y_tr": torch.tensor(y[:80]), "X_val": t(X[80:120]),
            "y_val": torch.tensor(y[80:120]), "X_pool_fit": t(rng.normal(size=(100, n))),
            "X_pool_eval": t(rng.normal(size=(60, n))), "n_classes": C}
    cfg = {"n_qubits": n, "host_steps": 5, "host_lr": 0.05, "pop_size": 4, "generations": 2,
           "offspring": 3, "elite": 1, "crossover_rate": 0.5, "parasite_pop": 4, "hof_size": 3,
           "parasite_m_panel": [2, 4], "gate_threshold": 0.0, "lam": 0.2}
    res = Search(data, cfg, 0.2, seed=0, log=lambda *_: None).run("evolve")
    keys = [p["kind"] + str(p["seed"]) + str(p["m_frac"]) + str(p["order"]) + str(p["sigma"])
            + str(p["alpha"]) for p in res["parasites"]]
    assert len(keys) == len(set(keys))
    for r in res["all"]:
        assert 0.0 <= r["gap_final"] <= 1.0
        assert r["fitness_final"] == pytest.approx(r["acc_val"] + 0.2 * r["gap_final"])
    assert res["best"]["fitness_final"] == max(r["fitness_final"] for r in res["all"])
    assert len(res["history"]) == 3


def test_gradient_variance_positive():
    g = random_genome(4, np.random.default_rng(5))
    assert gradient_variance(g, torch.randn(16, 4), draws=4) >= 0.0


def test_random_genome_small_n():
    rng = np.random.default_rng(0)
    for n in (2, 3):
        for _ in range(100):
            g = random_genome(n, rng)
            assert len(g.edges) <= n * (n - 1)


def test_panel_mode_never_mutates_m_frac():
    rng = np.random.default_rng(1)
    for _ in range(300):
        p = random_parasite(rng)
        q = mutate_parasite(p, rng, mutate_m_frac=False)
        assert q.m_frac == p.m_frac and q.key() != p.key()


def test_random_mode_coevolves_parasites():
    rng = np.random.default_rng(0)
    n = 3

    def t(a):
        return torch.tensor(a, dtype=torch.float32)

    X = rng.normal(size=(120, n))
    y = (X[:, 0] > 0).astype(int)
    data = {"X_tr": t(X[:60]), "y_tr": torch.tensor(y[:60]), "X_val": t(X[60:90]),
            "y_val": torch.tensor(y[60:90]), "X_pool_fit": t(rng.normal(size=(80, n))),
            "X_pool_eval": t(rng.normal(size=(40, n))), "n_classes": 2}
    cfg = {"n_qubits": n, "host_steps": 3, "host_lr": 0.05, "pop_size": 3, "generations": 2,
           "offspring": 2, "crossover_rate": 0.5, "parasite_pop": 4, "hof_size": 2,
           "parasite_m_panel": [2, 4], "gate_threshold": 1e-6}
    s = Search(data, cfg, 0.2, seed=0, log=lambda *_: None)
    res = s.run("random")
    assert len(res["history"]) == 3            # same schedule as evolve
    assert len({k[1] for k in s._agree}) > 4   # mutant parasites were created and used
    with pytest.raises(ValueError):
        Search(data, cfg, 0.2, seed=0).run("typo")


def test_imitate_panel_matches_single_budget_fits():
    """Sub-block solves at several budgets equal separate fits per budget."""
    g = random_genome(4, np.random.default_rng(7))
    model = QuantumHead(g, 3, seed=0)
    Xf, Xe = torch.randn(300, 4), torch.randn(150, 4)
    with torch.no_grad():
        Yf, Ye = model(Xf), model(Xe)
    for kind in ("fourier", "rff"):
        p = ParasiteGenome(kind, 1.0, 2, 1.0, 1e-3, seed=3)
        panel = imitate(p, g, [2, 5, 9], Xf, Yf, Xe, Ye)
        for m, (a, r2) in zip([2, 5, 9], panel):
            a1, r21 = imitate(p, g, m, Xf, Yf, Xe, Ye)
            assert a == pytest.approx(a1) and r2 == pytest.approx(r21, abs=1e-8)


def test_final_best_comes_from_final_population():
    rng = np.random.default_rng(0)
    n = 3

    def t(a):
        return torch.tensor(a, dtype=torch.float32)

    X = rng.normal(size=(120, n))
    y = (X[:, 0] * X[:, 2] > 0).astype(int)
    data = {"X_tr": t(X[:60]), "y_tr": torch.tensor(y[:60]), "X_val": t(X[60:90]),
            "y_val": torch.tensor(y[60:90]), "X_pool_fit": t(rng.normal(size=(80, n))),
            "X_pool_eval": t(rng.normal(size=(40, n))), "n_classes": 2}
    cfg = {"n_qubits": n, "host_steps": 3, "host_lr": 0.05, "pop_size": 3, "generations": 2,
           "offspring": 3, "crossover_rate": 0.5, "parasite_pop": 4, "hof_size": 2,
           "parasite_m_panel": [2, 4], "gate_threshold": 0.0}
    res = Search(data, cfg, 0.2, seed=1, log=lambda *_: None).run("evolve")
    assert res["best"]["in_final_pop"]
    finals = [r for r in res["all"] if r["in_final_pop"]]
    assert len(finals) == 3
    assert res["best"]["fitness_final"] == max(r["fitness_final"] for r in finals)


def test_check_methods_rejects_typos():
    from redqueen.runutil import check_methods, search_setup
    with pytest.raises(ValueError):
        check_methods(["redqueen", "redqeen"], ("redqueen",))
    assert search_setup("ga", 0.2) == (0.0, "evolve")
    assert search_setup("random", 0.2) == (0.2, "random")


def test_host_init_independent_of_global_rng():
    g = random_genome(4, np.random.default_rng(2))
    torch.manual_seed(123)
    a = QuantumHead(g, 3, seed=5).state_dict()
    torch.rand(1000)  # disturb the global RNG
    b = QuantumHead(g, 3, seed=5).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_audit_curve_matches_single_budget_and_qml_pool_split():
    from redqueen import classical, qmlbench
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    pool, k = qmlbench._audit_pool(X, 4, 0.0, np.random.default_rng(1))  # jitter 0: copies == points
    # fit and selection parts are copies of disjoint train points
    fit_src, sel_src = {tuple(r) for r in pool[:k]}, {tuple(r) for r in pool[k:]}
    assert len(pool) == 200 and k == 160 and len(fit_src) == 40 and len(sel_src) == 10
    assert not fit_src & sel_src
    g = random_genome(3, rng)
    model = QuantumHead(g, 2, seed=0)
    d = {"X_pool_audit": torch.randn(300, 3), "X_test": torch.randn(80, 3), "n_classes": 2}
    T = classical.targets(model, d)
    curve = classical.fourier_curve(g, T, [3, 6])
    assert curve[1]["agree"] == pytest.approx(classical.audit_fourier(model, g, d, 6, T=T)["agree"])
    rc = classical.rff_curve(T, [3, 6])
    assert rc[0]["agree"] == pytest.approx(classical.audit_rff(model, d, 3, T=T)["agree"])
    out = classical.audit(model, g, d, budgets=(3, 6), unbounded_m=12)
    assert set(out["m90"]) == {"fourier", "rff", "mlp"} and out["m90_censor_value"] == 12
    assert all(m <= out["fourier_spectrum_size"] for m in out["curves"]["fourier"])
    X31 = rng.normal(size=(31, 3))   # N not a multiple of 5
    p31, k31 = qmlbench._audit_pool(X31, 3, 0.0, np.random.default_rng(2))
    assert not {tuple(r) for r in p31[:k31]} & {tuple(r) for r in p31[k31:]}


def test_prepare_shrinks_pools_for_small_datasets(tmp_path):
    from redqueen.data import prepare
    rng = np.random.default_rng(0)
    n = 300
    split = np.array(["train"] * 200 + ["val"] * 40 + ["test"] * 60)
    f = tmp_path / "tiny.npz"
    np.savez(f, emb=rng.normal(size=(n, 16)), spec=rng.normal(size=(n, 6)),
             y=np.tile([0, 1, 2], n // 3), split=split, files=np.arange(n).astype(str))
    d = prepare(str(f), 0.25, 0, 4, device="cpu")
    s = d["sizes"]
    assert s["pool_fit"] + s["pool_eval"] + s["pool_audit"] == 200 - s["labeled"] + 40
    assert s["pool_fit"] > 0 and s["pool_eval"] > 0 and s["pool_audit"] > 0


def test_check_labels_and_singleton_class_split(tmp_path):
    from redqueen.data import prepare
    from redqueen.datasets import check_labels
    assert list(check_labels(np.array([0, 2, 1, 0]), "x")) == [0, 2, 1, 0]
    for bad in (np.eye(3)[[0, 1, 2]], np.array([1, 2, 3]), np.array([0, 2, 2])):
        with pytest.raises(ValueError):
            check_labels(bad, "x")
    # a class that gets a single labeled sample must not crash prepare
    rng = np.random.default_rng(0)
    y = np.array([0] * 150 + [1] * 140 + [2] * 10)
    split = np.array(["train"] * 300)
    split[rng.choice(300, 60, replace=False)] = "test"
    f = tmp_path / "imb.npz"
    np.savez(f, emb=rng.normal(size=(300, 8)), spec=rng.normal(size=(300, 6)), y=y,
             split=split, files=np.arange(300).astype(str))
    d = prepare(str(f), 0.05, 0, 4, device="cpu")
    assert d["sizes"]["labeled"] == 12
