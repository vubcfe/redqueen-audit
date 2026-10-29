"""Batched, differentiable statevector simulation in PyTorch.

States have shape ``(B, 2, ..., 2)`` with one axis per wire; wire 0 is the
most significant bit (PennyLane convention). Gates are either shared by the
whole batch (trainable parameters, matrix ``(2, 2)``) or per sample
(data encoding, matrix ``(B, 2, 2)``).
"""

from __future__ import annotations

import itertools

import torch

CDTYPE = torch.complex64


def zero_state(batch: int, n: int, device) -> torch.Tensor:
    psi = torch.zeros((batch,) + (2,) * n, dtype=CDTYPE, device=device)
    psi.view(batch, -1)[:, 0] = 1.0
    return psi


def rx(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    re = torch.stack([torch.stack([c, zero], -1), torch.stack([zero, c], -1)], -2)
    im = torch.stack([torch.stack([zero, -s], -1), torch.stack([-s, zero], -1)], -2)
    return torch.complex(re, im)


def ry(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    re = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return torch.complex(re, torch.zeros_like(re))


def rz(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    zero = torch.zeros_like(c)
    re = torch.stack([torch.stack([c, zero], -1), torch.stack([zero, c], -1)], -2)
    im = torch.stack([torch.stack([-s, zero], -1), torch.stack([zero, s], -1)], -2)
    return torch.complex(re, im)


ROT = {"RX": rx, "RY": ry, "RZ": rz}


def apply_1q(psi: torch.Tensor, U: torch.Tensor, wire: int) -> torch.Tensor:
    """Apply a single-qubit gate; ``U`` is ``(2, 2)`` or ``(B, 2, 2)``."""
    p = psi.movedim(1 + wire, -1)
    if U.dim() == 2:
        p = p @ U.T
    else:
        shape = p.shape
        p = torch.einsum("bij,bkj->bki", U, p.reshape(shape[0], -1, 2)).reshape(shape)
    return p.movedim(-1, 1 + wire)


def apply_cnot(psi: torch.Tensor, control: int, target: int) -> torch.Tensor:
    p = psi.movedim((1 + control, 1 + target), (1, 2))
    p = torch.stack([p[:, 0], p[:, 1].flip(1)], dim=1)
    return p.movedim((1, 2), (1 + control, 1 + target))


def cz_sign(n: int, a: int, b: int, device) -> torch.Tensor:
    s = torch.ones((2,) * n, device=device)
    idx = [slice(None)] * n
    idx[a], idx[b] = 1, 1
    s[tuple(idx)] = -1.0
    return s


def z_signs(n: int, device) -> torch.Tensor:
    """``(n, 2**n)`` matrix with the eigenvalue of ``Z_i`` on each basis state."""
    bits = torch.tensor(list(itertools.product([0, 1], repeat=n)), device=device)
    return (1 - 2 * bits).T.float()


def probabilities(psi: torch.Tensor) -> torch.Tensor:
    return psi.reshape(psi.shape[0], -1).abs().pow(2)
