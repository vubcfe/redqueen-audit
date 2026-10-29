"""Quantum classifier head built from a :class:`HostGenome`, its training and
the trainability gate."""

from __future__ import annotations

import math
import time

import torch
from torch import nn

from . import sim
from .genome import HostGenome


class QuantumHead(nn.Module):
    def __init__(self, g: HostGenome, n_classes: int, seed: int = 0):
        super().__init__()
        self.g = g
        gen = torch.Generator().manual_seed(seed)
        self.theta = nn.Parameter(
            2 * math.pi * torch.rand(g.layers, g.n, len(g.rot_gates), generator=gen))
        # the readout init depends on (genome, seed) only, through ``gen``; it
        # never touches the global CPU/CUDA RNG. Same law as nn.Linear's default.
        self.readout = nn.Linear(g.n_obs, n_classes)
        bound = 1.0 / math.sqrt(g.n_obs)
        with torch.no_grad():
            self.readout.weight.uniform_(-bound, bound, generator=gen)
            self.readout.bias.uniform_(-bound, bound, generator=gen)
        self._consts = None

    def _constants(self, device):
        if self._consts is None or self._consts[0].device != device:
            zs = sim.z_signs(self.g.n, device)
            obs = [zs]
            if self.g.zz_pairs:
                obs.append(torch.stack([zs[a] * zs[b] for a, b in self.g.zz_pairs]))
            signs = torch.cat(obs, 0)
            cz = [sim.cz_sign(self.g.n, a, b, device) for a, b in self.g.edges]
            self._consts = (signs, cz)
        return self._consts

    def expectations(self, x: torch.Tensor) -> torch.Tensor:
        """Pauli expectations ``(B, n_obs)`` for standardised features ``x``."""
        g = self.g
        signs, cz = self._constants(x.device)
        psi = sim.zero_state(x.shape[0], g.n, x.device)
        enc = sim.ROT[g.enc_gate]
        angles = g.bandwidth * x
        for layer in range(g.layers):
            for w in range(g.n):
                psi = sim.apply_1q(psi, enc(angles[:, w]), w)
            for w in range(g.n):
                for k, gate in enumerate(g.rot_gates):
                    psi = sim.apply_1q(psi, sim.ROT[gate](self.theta[layer, w, k]), w)
            for e, (a, b) in enumerate(g.edges):
                psi = psi * cz[e] if g.ent_gate == "CZ" else sim.apply_cnot(psi, a, b)
        return sim.probabilities(psi) @ signs.T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.expectations(x))


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, batch: int = 4096) -> torch.Tensor:
    model.eval()
    return torch.cat([model(X[i:i + batch]) for i in range(0, len(X), batch)])


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return (logits.argmax(1) == y).float().mean().item()


def train_head(model: nn.Module, X: torch.Tensor, y: torch.Tensor, steps: int,
               lr: float, weight_decay: float = 0.0) -> nn.Module:
    """Full-batch Adam on cross-entropy (labeled sets here are < 2k samples)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()
    return model


def train_host(g: HostGenome, data: dict, n_classes: int, steps: int, lr: float,
               seed: int) -> tuple[QuantumHead, dict]:
    t0 = time.time()
    model = QuantumHead(g, n_classes, seed=seed).to(data["X_tr"].device)
    torch.manual_seed(seed)
    train_head(model, data["X_tr"], data["y_tr"], steps, lr)
    rec = {
        "acc_train": accuracy(predict_logits(model, data["X_tr"]), data["y_tr"]),
        "acc_val": accuracy(predict_logits(model, data["X_val"]), data["y_val"]),
        "train_seconds": time.time() - t0,
    }
    return model, rec


def gradient_variance(g: HostGenome, X: torch.Tensor, draws: int = 16, seed: int = 0) -> float:
    """Variance over random initialisations of d<Z_0>/d(theta), averaged over
    parameters (the barren-plateau diagnostic of McClean et al., 2018)."""
    grads = []
    for d in range(draws):
        model = QuantumHead(g, 2, seed=seed * 1000 + d).to(X.device)
        cost = model.expectations(X)[:, 0].mean()
        (grad,) = torch.autograd.grad(cost, model.theta)
        grads.append(grad.flatten())
    return torch.stack(grads).var(0).mean().item()
