"""Host genome: the design space of a quantum classifier head.

A host is a data re-uploading circuit on ``n`` qubits. Each of its ``layers``
blocks encodes feature ``i`` on wire ``i`` with ``enc_gate(bandwidth * x_i)``,
then applies the trainable rotations ``rot`` on every wire, then the
entangling gates on ``edges``. The read-out measures ``Z_i`` on every wire,
plus ``Z_a Z_b`` on every edge when ``readout == "ZZ"``, and a linear layer
maps these expectations to class logits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

ENC_GATES = ("RX", "RY")
BANDWIDTHS = (0.25, 0.5, 1.0, 2.0)
LAYERS = (1, 2, 3)
ROTS = ("RY", "RZRY", "RZRYRZ")
ENT_GATES = ("CZ", "CNOT")
READOUTS = ("Z", "ZZ")


@dataclass(frozen=True)
class HostGenome:
    n: int
    enc_gate: str
    bandwidth: float
    layers: int
    rot: str
    edges: tuple[tuple[int, int], ...]
    ent_gate: str
    readout: str

    @property
    def rot_gates(self) -> list[str]:
        return [self.rot[i:i + 2] for i in range(0, len(self.rot), 2)]

    @property
    def zz_pairs(self) -> list[tuple[int, int]]:
        if self.readout != "ZZ":
            return []
        return sorted({tuple(sorted(e)) for e in self.edges})

    @property
    def n_obs(self) -> int:
        return self.n + len(self.zz_pairs)

    @property
    def n_quantum_params(self) -> int:
        return self.layers * self.n * len(self.rot_gates)

    def n_params(self, n_classes: int) -> int:
        return self.n_quantum_params + (self.n_obs + 1) * n_classes

    @property
    def two_qubit_gates(self) -> int:
        return self.layers * len(self.edges)

    def descriptor(self) -> tuple[int, int]:
        """MAP-Elites cell: (re-upload depth, entangling-edge bin)."""
        e = len(self.edges)
        edge_bin = 0 if e == 0 else 1 if e < self.n else 2 if e < 2 * self.n else 3
        return self.layers, edge_bin

    def key(self) -> str:
        edges = "-".join(f"{a}{b}" for a, b in self.edges)
        return (f"{self.enc_gate}|{self.bandwidth}|L{self.layers}|{self.rot}|"
                f"{self.ent_gate}[{edges}]|{self.readout}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["edges"] = [list(e) for e in self.edges]
        return d

    @staticmethod
    def from_dict(d: dict) -> "HostGenome":
        return HostGenome(**{**d, "edges": tuple(tuple(e) for e in d["edges"])})


def _all_pairs(n: int) -> list[tuple[int, int]]:
    return [(a, b) for a in range(n) for b in range(n) if a != b]


def _normalise_edges(edges, ent_gate: str, max_edges: int) -> tuple[tuple[int, int], ...]:
    """Drop duplicates (orientation is irrelevant for CZ) and cap the count."""
    seen, out = set(), []
    for a, b in edges:
        k = tuple(sorted((a, b))) if ent_gate == "CZ" else (a, b)
        if a != b and k not in seen:
            seen.add(k)
            out.append((a, b))
    return tuple(out[:max_edges])


def ring_edges(n: int) -> tuple[tuple[int, int], ...]:
    return tuple((i, (i + 1) % n) for i in range(n)) if n > 2 else ((0, 1),)


def random_genome(n: int, rng: np.random.Generator, max_edges: int | None = None) -> HostGenome:
    max_edges = max_edges or 2 * n
    ent_gate = str(rng.choice(ENT_GATES))
    pairs = _all_pairs(n)
    k = int(rng.integers(0, min(max_edges, len(pairs)) + 1))
    edges = [pairs[i] for i in rng.choice(len(pairs), size=k, replace=False)]
    return HostGenome(
        n=n,
        enc_gate=str(rng.choice(ENC_GATES)),
        bandwidth=float(rng.choice(BANDWIDTHS)),
        layers=int(rng.choice(LAYERS)),
        rot=str(rng.choice(ROTS)),
        edges=_normalise_edges(edges, ent_gate, max_edges),
        ent_gate=ent_gate,
        readout=str(rng.choice(READOUTS)),
    )


def fixed_hea(n: int, layers: int) -> HostGenome:
    """Ring-CZ hardware-efficient head used as the fixed quantum baseline."""
    return HostGenome(n=n, enc_gate="RY", bandwidth=1.0, layers=layers, rot="RZRYRZ",
                      edges=ring_edges(n), ent_gate="CZ", readout="Z")


def mutate(g: HostGenome, rng: np.random.Generator, max_edges: int | None = None) -> HostGenome:
    max_edges = max_edges or 2 * g.n
    field = rng.choice(["enc_gate", "bandwidth", "layers", "rot", "edges", "edges",
                        "ent_gate", "readout"])
    if field == "edges":
        edges = list(g.edges)
        op = rng.choice(["add", "remove", "rewire"]) if edges else "add"
        if op == "add" and len(edges) < max_edges:
            pairs = _all_pairs(g.n)
            edges.append(pairs[int(rng.integers(len(pairs)))])
        elif op == "remove" and edges:
            edges.pop(int(rng.integers(len(edges))))
        elif edges:
            i = int(rng.integers(len(edges)))
            a, b = edges[i]
            c = int(rng.integers(g.n))
            edges[i] = (a, c) if rng.random() < 0.5 else (c, b)
        return replace(g, edges=_normalise_edges(edges, g.ent_gate, max_edges))
    options = {"enc_gate": ENC_GATES, "bandwidth": BANDWIDTHS, "layers": LAYERS,
               "rot": ROTS, "ent_gate": ENT_GATES, "readout": READOUTS}[field]
    current = getattr(g, field)
    choices = [o for o in options if o != current]
    value = choices[int(rng.integers(len(choices)))]
    g = replace(g, **{field: value})
    if field == "ent_gate":
        g = replace(g, edges=_normalise_edges(g.edges, g.ent_gate, max_edges))
    return g


def crossover(a: HostGenome, b: HostGenome, rng: np.random.Generator,
              max_edges: int | None = None) -> HostGenome:
    max_edges = max_edges or 2 * a.n
    pick = lambda f: getattr(a if rng.random() < 0.5 else b, f)  # noqa: E731
    ent_gate = pick("ent_gate")
    pool = list(dict.fromkeys(a.edges + b.edges))
    keep = [e for e in pool if rng.random() < 0.5]
    return HostGenome(
        n=a.n, enc_gate=pick("enc_gate"), bandwidth=pick("bandwidth"), layers=pick("layers"),
        rot=pick("rot"), edges=_normalise_edges(keep, ent_gate, max_edges),
        ent_gate=ent_gate, readout=pick("readout"),
    )
