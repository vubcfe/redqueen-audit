"""Parasites: budget-limited classical surrogates that imitate a host.

A parasite is a ridge regression from Fourier features of the input to the
host's logits. It is fitted on unlabeled inputs and scored by how often its
argmax agrees with the host's on a disjoint set of unlabeled inputs.

``fourier`` (white-box) uses frequencies from the host's own spectrum. With
``layers`` re-uploads of ``enc_gate(bandwidth * x_i)``, the output is a
trigonometric polynomial whose frequencies on dimension ``i`` lie in
``bandwidth * {-L, ..., L}`` (Schuld et al., 2021). ``rff`` (black-box) draws
Gaussian random frequencies and knows nothing about the host.

The number of frequencies ``m`` (the budget) is chosen by the caller. It must
be the same for every host being compared: a budget that grows with the host
size lets a search raise its "gap" just by shrinking the host (see
``scripts/budget_confound.py``).
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from .genome import HostGenome

KINDS = ("fourier", "rff")
M_FRACS = (0.25, 0.5, 1.0)
ORDERS = (1, 2, 3)
SIGMAS = (0.25, 0.5, 1.0, 2.0, 4.0)
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)


@dataclass(frozen=True)
class ParasiteGenome:
    kind: str
    m_frac: float
    order: int
    sigma: float
    alpha: float
    seed: int

    def key(self) -> str:
        if self.kind == "fourier":
            return f"fourier|m{self.m_frac}|o{self.order}|a{self.alpha}|s{self.seed}"
        return f"rff|m{self.m_frac}|sig{self.sigma}|a{self.alpha}|s{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


def random_parasite(rng: np.random.Generator, kinds=KINDS) -> ParasiteGenome:
    return ParasiteGenome(kind=str(rng.choice(kinds)), m_frac=float(rng.choice(M_FRACS)),
                          order=int(rng.choice(ORDERS)), sigma=float(rng.choice(SIGMAS)),
                          alpha=float(rng.choice(ALPHAS)), seed=int(rng.integers(1 << 30)))


def mutate_parasite(p: ParasiteGenome, rng: np.random.Generator, kinds=KINDS,
                    mutate_m_frac: bool = True) -> ParasiteGenome:
    """Change exactly one field that matters (``order`` is unused by ``rff``,
    ``sigma`` by ``fourier``, ``m_frac`` under a budget panel) to a different
    value, so the mutant behaves differently and has a different key."""
    options = {"kind": kinds, "alpha": ALPHAS,
               "order" if p.kind == "fourier" else "sigma":
               ORDERS if p.kind == "fourier" else SIGMAS}
    if mutate_m_frac:
        options["m_frac"] = M_FRACS
    fields = [f for f, o in options.items() if len(o) > 1] + ["seed"]
    field = fields[int(rng.integers(len(fields)))]
    if field == "seed":
        return replace(p, seed=(p.seed + 1 + int(rng.integers((1 << 30) - 1))) % (1 << 30))
    current = getattr(p, field)
    choices = [o for o in options[field] if o != current]
    return replace(p, **{field: type(current)(choices[int(rng.integers(len(choices)))])})


def budgets(p: ParasiteGenome, cfg: dict) -> list[int]:
    """Frequency budgets at which parasite ``p`` attacks every host.

    ``parasite_m_panel`` (preferred): the whole panel, ignoring ``m_frac``, so
    a host cannot specialise against one budget. ``parasite_m_abs``: a single
    absolute budget scaled by the parasite's ``m_frac``."""
    if cfg.get("parasite_m_panel"):
        return [int(m) for m in cfg["parasite_m_panel"]]
    return [max(1, int(p.m_frac * cfg["parasite_m_abs"]))]


def max_features(host: HostGenome, n_classes: int, kappa: float) -> int:
    """Largest M such that (2M + 1) * n_classes <= kappa * n_params(host)."""
    return max(1, int((kappa * host.n_params(n_classes) / n_classes - 1) // 2))


@lru_cache(maxsize=None)
def spectrum(n: int, layers: int, order: int) -> np.ndarray:
    """Integer frequency vectors with entries in {-L..L}, at most ``order``
    non-zeros, one representative per +/- pair (enumeration order, which is
    grouped by input dimension -- use :func:`prioritised` to take a prefix)."""
    out = []
    for k in range(1, min(order, n) + 1):
        for dims in itertools.combinations(range(n), k):
            for vals in itertools.product([v for v in range(-layers, layers + 1) if v], repeat=k):
                if vals[0] > 0:
                    w = np.zeros(n, dtype=np.int8)
                    w[list(dims)] = vals
                    out.append(w)
    return np.stack(out)


def prioritised(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order frequencies by (number of active inputs, largest |frequency|,
    total |frequency|), random within ties.

    Taking a prefix then never favours some input dimensions over others, and
    the first ``n`` entries are the ``n`` first harmonics whatever the depth
    ``L``; the raw enumeration order would cover fewer inputs for deeper
    circuits and bias the comparison between hosts."""
    a = np.abs(spec.astype(int))
    keys = (rng.random(len(spec)), a.sum(1), a.max(1), np.count_nonzero(spec, axis=1))
    return spec[np.lexsort(keys)]


def frequencies(p: ParasiteGenome, host: HostGenome, m: int) -> np.ndarray:
    rng = np.random.default_rng(p.seed)
    if p.kind == "rff":
        return rng.normal(0.0, p.sigma, size=(m, host.n))
    spec = prioritised(spectrum(host.n, host.layers, p.order), rng)
    return host.bandwidth * spec[:m].astype(float)


def features(X: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    z = X.double() @ W.T
    ones = torch.ones(len(X), 1, dtype=z.dtype, device=z.device)
    return torch.cat([ones, torch.cos(z), torch.sin(z)], 1)


def ridge(Phi: torch.Tensor, Y: torch.Tensor, alpha: float) -> torch.Tensor:
    return ridge_path(Phi, Y, (alpha,))[0]


def ridge_path(Phi: torch.Tensor, Y: torch.Tensor, alphas) -> list[torch.Tensor]:
    """Ridge solutions for several penalties, building the Gram matrix once."""
    G, b = Phi.T @ Phi, Phi.T @ Y
    eye = torch.eye(G.shape[0], dtype=G.dtype, device=G.device)
    return [torch.linalg.solve(G + a * len(Phi) * eye, b) for a in alphas]


def imitation_scores(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    agree = (pred.argmax(1) == target.argmax(1)).double().mean().item()
    sst = (target - target.mean(0)).pow(2).sum()
    r2 = 1.0 - ((pred - target).pow(2).sum() / sst).item()
    return agree, r2


def imitate(p: ParasiteGenome, host: HostGenome, m,
            X_fit: torch.Tensor, Y_fit: torch.Tensor,
            X_eval: torch.Tensor, Y_eval: torch.Tensor):
    """Fit parasite ``p`` to the host logits with ``m`` frequencies; return
    (agreement, R^2) on the eval inputs. ``m`` may be a list of budgets: the
    frequencies of a smaller budget are a prefix of a larger one's, so the
    features and Gram matrix are built once at the largest budget and each
    budget solves on its sub-block. Returns one pair per budget."""
    single = np.ndim(m) == 0
    ms = [int(m)] if single else [int(v) for v in m]
    W = torch.as_tensor(frequencies(p, host, max(1, max(ms))), device=X_fit.device)
    M = W.shape[0]                    # may be < max(ms) if the spectrum is small
    F_fit, F_eval = features(X_fit, W), features(X_eval, W)
    G, b = F_fit.T @ F_fit, F_fit.T @ Y_fit.double()
    out = []
    for mm in ms:
        k = min(max(1, mm), M)
        cols = torch.cat([torch.arange(0, 1 + k), torch.arange(1 + M, 1 + M + k)]).to(G.device)
        Gs = G[cols][:, cols]
        Gs = Gs + p.alpha * len(F_fit) * torch.eye(len(cols), dtype=G.dtype, device=G.device)
        beta = torch.linalg.solve(Gs, b[cols])
        out.append(imitation_scores(F_eval[:, cols] @ beta, Y_eval.double()))
    return out[0] if single else out
