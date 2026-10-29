"""Classical task baselines and held-out imitation auditors.

Baselines are trained on the labels (search-train), selected on search-val and
scored on test. Auditors are trained on *host logits* over unlabeled inputs
and scored by test agreement. Auditor families and budgets differ from the
search parasites (MLP distillation is never used during search, and the
unbounded budgets are far larger), so the audit is not circular.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from .host import predict_logits, train_head
from .surrogate import features, imitation_scores, prioritised, spectrum


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    return {"acc": float((pred == y).mean()), "f1_macro": float(f1_score(y, pred, average="macro"))}


def _np(t):
    return t.detach().cpu().numpy()


def _select(fits, X_val, y_val):
    best = max(fits, key=lambda m: (m[1].predict(X_val) == y_val).mean())
    return best


def logreg(d: dict, X=None) -> dict:
    Xtr, Xva, Xte = X if X is not None else (_np(d["X_tr"]), _np(d["X_val"]), _np(d["X_test"]))
    ytr, yva, yte = _np(d["y_tr"]), _np(d["y_val"]), _np(d["y_test"])
    fits = [(C, LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr)) for C in (0.01, 0.1, 1, 10, 100)]
    C, m = _select(fits, Xva, yva)
    return {**scores(m.predict(Xte), yte), "acc_val": float((m.predict(Xva) == yva).mean()),
            "hparams": {"C": C}}


def svm_rbf(d: dict) -> dict:
    Xtr, Xva, Xte = _np(d["X_tr"]), _np(d["X_val"]), _np(d["X_test"])
    ytr, yva, yte = _np(d["y_tr"]), _np(d["y_val"]), _np(d["y_test"])
    fits = [((C, g), SVC(C=C, gamma=g).fit(Xtr, ytr))
            for C in (0.1, 1, 10, 100) for g in ("scale", 0.01, 0.1, 1.0)]
    hp, m = _select(fits, Xva, yva)
    return {**scores(m.predict(Xte), yte), "acc_val": float((m.predict(Xva) == yva).mean()),
            "hparams": {"C": hp[0], "gamma": hp[1]}}


def linear_probe_full(d: dict) -> dict:
    """Reference only: logistic regression on the full embedding."""
    sc = StandardScaler().fit(d["raw"]["tr"])
    return logreg(d, X=(sc.transform(d["raw"]["tr"]), sc.transform(d["raw"]["val"]),
                        sc.transform(d["raw"]["test"])))


def mlp_hidden(target_params: int, d_in: int, n_classes: int) -> int:
    return max(1, round((target_params - n_classes) / (d_in + n_classes + 1)))


def mlp(d_in: int, hidden: int, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), nn.Linear(hidden, n_classes))


def mlp_matched(d: dict, target_params: int, steps: int, lr: float = 0.01, seed: int = 0) -> dict:
    C, d_in = d["n_classes"], d["X_tr"].shape[1]
    h = mlp_hidden(target_params, d_in, C)
    best = None
    for wd in (0.0, 1e-4, 1e-3, 1e-2):
        torch.manual_seed(seed)
        m = train_head(mlp(d_in, h, C).to(d["X_tr"].device), d["X_tr"], d["y_tr"], steps, lr, wd)
        acc_val = (predict_logits(m, d["X_val"]).argmax(1) == d["y_val"]).float().mean().item()
        if best is None or acc_val > best[0]:
            best = (acc_val, wd, m)
    acc_val, wd, m = best
    pred = _np(predict_logits(m, d["X_test"]).argmax(1))
    acc_tr = (predict_logits(m, d["X_tr"]).argmax(1) == d["y_tr"]).float().mean().item()
    n_params = sum(p.numel() for p in m.parameters())
    return {**scores(pred, _np(d["y_test"])), "acc_val": acc_val, "acc_train": acc_tr,
            "n_params": n_params, "hparams": {"hidden": h, "weight_decay": wd}}


def rff_logreg(d: dict, m: int, seed: int = 0) -> dict:
    """Budget-matched classical head: logistic regression on ``m`` Gaussian
    random Fourier features (2m + 1 inputs), bandwidth and C chosen on
    search-val."""
    ytr, yva, yte = _np(d["y_tr"]), _np(d["y_val"]), _np(d["y_test"])
    best = None
    for i, sigma in enumerate(RFF_SIGMAS):  # same grid and generators as the RFF auditor
        rng = np.random.default_rng([seed, i])
        W = torch.as_tensor(rng.normal(0, sigma, (m, d["X_tr"].shape[1])), device=d["X_tr"].device)
        F = [_np(features(d[k], W))[:, 1:] for k in ("X_tr", "X_val", "X_test")]  # sklearn adds bias
        for C in (0.1, 1, 10):
            clf = LogisticRegression(C=C, max_iter=5000).fit(F[0], ytr)
            acc_val = float((clf.predict(F[1]) == yva).mean())
            if best is None or acc_val > best[0]:
                best = (acc_val, sigma, C, clf.predict(F[2]))
    acc_val, sigma, C, pred = best
    return {**scores(pred, yte), "acc_val": acc_val, "m": m,
            "n_params": (2 * m + 1) * d["n_classes"], "hparams": {"sigma": sigma, "C": C}}


# ------------------------------------------------------------------ auditors
def _split_pool(d: dict):
    """Auditor fit/selection inputs: the audit pool only, which is disjoint
    from the pools the search parasites were fitted and scored on."""
    X = d["X_pool_audit"]
    k = d.get("audit_split", int(0.8 * len(X)))  # a builder may fix the boundary
    return X[:k], X[k:]


def targets(host_model, d: dict) -> tuple:
    """(X_fit, X_sel, Y_fit, Y_sel, Y_test, X_test): auditor inputs and host
    logits, computed once per audited host."""
    Xf, Xs = _split_pool(d)
    Ys = tuple(predict_logits(host_model, X) for X in (Xf, Xs, d["X_test"]))
    return (Xf, Xs) + Ys + (d["X_test"],)


def _ridge_curve(W, T, ms, alphas) -> list[tuple[float, dict]]:
    """For each budget m (a prefix of the rows of ``W``): ridge on the fit
    pool for each alpha, pick alpha by agreement on the selection pool, score
    on test. Features and the Gram matrix are built once at the full width.
    Returns one (selection agreement, test dict) per budget."""
    Xf, Xs, Yf, Ys, Yt, Xt = T
    M = W.shape[0]
    Ff, Fs, Ft = features(Xf, W), features(Xs, W), features(Xt, W)
    G, b = Ff.T @ Ff, Ff.T @ Yf.double()
    out = []
    for m in ms:
        k = min(max(1, m), M)
        cols = torch.cat([torch.arange(0, 1 + k), torch.arange(1 + M, 1 + M + k)]).to(G.device)
        Gs, bs = G[cols][:, cols], b[cols]
        eye = torch.eye(len(cols), dtype=G.dtype, device=G.device)
        best = None
        for a in alphas:
            beta = torch.linalg.solve(Gs + a * len(Ff) * eye, bs)
            sel = imitation_scores(Fs[:, cols] @ beta, Ys.double())[0]
            if best is None or sel > best[0]:
                best = (sel, beta)
        agree, r2 = imitation_scores(Ft[:, cols] @ best[1], Yt.double())
        out.append((best[0], {"agree": agree, "r2": r2, "m": k}))
    return out


FOURIER_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2)
RFF_ALPHAS = (1e-6, 1e-4, 1e-2)
RFF_SIGMAS = (0.25, 0.5, 1.0, 2.0)


def fourier_curve(g, T, ms, seed: int = 0) -> list[dict]:
    """White-box auditor: the m lowest frequencies of the host's spectrum,
    interaction order at most 3 (higher orders are not enumerated, so even the
    largest budget is not a complete Fourier model of an entangled host)."""
    spec = prioritised(spectrum(g.n, g.layers, min(3, g.n)), np.random.default_rng(seed))
    W = torch.as_tensor(g.bandwidth * spec[:max(ms)].astype(float), device=T[0].device)
    return [r for _, r in _ridge_curve(W, T, ms, FOURIER_ALPHAS)]


def rff_curve(T, ms, seed: int = 0) -> list[dict]:
    """Black-box auditor: m Gaussian random frequencies; bandwidth and penalty
    chosen per budget on the selection pool. One generator per sigma, so the
    frequencies of a budget are a prefix of a larger budget's."""
    per_sigma = []
    for i, sigma in enumerate(RFF_SIGMAS):
        rng = np.random.default_rng([seed, i])
        W = torch.as_tensor(rng.normal(0, sigma, (max(ms), T[0].shape[1])), device=T[0].device)
        per_sigma.append(_ridge_curve(W, T, ms, RFF_ALPHAS))
    return [max((curve[j] for curve in per_sigma), key=lambda sr: sr[0])[1]
            for j in range(len(ms))]


def audit_fourier(host_model, g, d: dict, m: int, seed: int = 0, T=None) -> dict:
    return fourier_curve(g, T or targets(host_model, d), [m], seed)[0]


def audit_rff(host_model, d: dict, m: int, seed: int = 0, T=None) -> dict:
    return rff_curve(T or targets(host_model, d), [m], seed)[0]


def audit_mlp(host_model, d: dict, hidden: int, steps: int = 2000, seed: int = 0, T=None) -> dict:
    """Distillation auditor: a one-hidden-layer MLP regressed on the host logits."""
    Xf, _, Yf, _, Yt, _ = T or targets(host_model, d)
    torch.manual_seed(seed)
    m = mlp(Xf.shape[1], hidden, Yf.shape[1]).to(Xf.device)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        nn.functional.mse_loss(m(Xf), Yf).backward()
        opt.step()
    agree, r2 = imitation_scores(predict_logits(m, d["X_test"]).double(), Yt.double())
    return {"agree": agree, "r2": r2, "hidden": hidden}


# Off the search panels ({4,8,16,32} and {8,16,32,64}) so that no audit budget
# is one the search optimised against.
AUDIT_M = (3, 6, 12, 24, 48, 96, 192)


def m_at(curve: dict, level: float = 0.9, censored=None):
    """Smallest budget whose agreement reaches ``level``. If no budget does,
    return ``censored`` (the next grid value), so the hardest-to-imitate hosts
    keep the highest rank instead of dropping out of rank tests."""
    hit = [m for m, a in sorted(curve.items()) if a >= level]
    return hit[0] if hit else censored


def audit(host_model, g, d: dict, budgets=AUDIT_M, unbounded_m: int = 2000) -> dict:
    """Imitation curves of auditors on test inputs.

    Every host faces the same absolute budgets: ``m`` frequencies for the
    Fourier and RFF auditors, an MLP with as many weights as an ``m``-frequency
    ridge, ``(2m + 1) * C``. ``m90`` is the smallest budget reaching 90 %
    agreement; larger means harder to imitate.

    All auditors use the audit pool and budgets off the search panel. Only
    ``mlp`` is a family the search never optimised against (the primary
    outcome); ``fourier`` and ``rff`` are the search families and are
    reported as in-family checks."""
    C, T = d["n_classes"], targets(host_model, d)
    ms = list(budgets) + [unbounded_m]
    four, rff = fourier_curve(g, T, ms), rff_curve(T, ms)
    # the Fourier auditor is capped by the host's order-3 spectrum size: key its
    # curve by the effective number of frequencies and keep each size once
    fourier = {}
    for r in four[:-1]:
        fourier.setdefault(r["m"], r["agree"])
    curves = {"fourier": fourier,
              "rff": {m: r["agree"] for m, r in zip(budgets, rff)},
              "mlp": {m: audit_mlp(host_model, d, mlp_hidden((2 * m + 1) * C, g.n, C),
                                   T=T)["agree"] for m in budgets}}
    censor = 2 * max(budgets)
    return {"curves": curves,
            "m90": {k: m_at(c, censored=censor) for k, c in curves.items()},
            "m90_censored": {k: m_at(c) is None for k, c in curves.items()},
            "m90_censor_value": censor,
            "fourier_spectrum_size": four[-1]["m"],
            "unbounded": {"fourier_order3": four[-1]["agree"], "rff": rff[-1]["agree"],
                          "mlp": audit_mlp(host_model, d, 512, T=T)["agree"]}}
