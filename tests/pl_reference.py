"""PennyLane reference for the torch simulator.

Run with an interpreter that has PennyLane (not torch); reads a JSON job from
stdin and prints the expectation values as JSON.
"""

import json
import sys

import numpy as np
import pennylane as qml

job = json.load(sys.stdin)
g, X, theta = job["genome"], np.array(job["x"]), np.array(job["theta"])
n = g["n"]
rots = [g["rot"][i:i + 2] for i in range(0, len(g["rot"]), 2)]
gate = {"RX": qml.RX, "RY": qml.RY, "RZ": qml.RZ}
zz = sorted({tuple(sorted(e)) for e in g["edges"]}) if g["readout"] == "ZZ" else []
dev = qml.device("default.qubit", wires=n)


@qml.qnode(dev)
def circuit(x):
    for layer in range(g["layers"]):
        for w in range(n):
            gate[g["enc_gate"]](g["bandwidth"] * x[w], wires=w)
        for w in range(n):
            for k, r in enumerate(rots):
                gate[r](theta[layer][w][k], wires=w)
        for a, b in g["edges"]:
            (qml.CZ if g["ent_gate"] == "CZ" else qml.CNOT)(wires=[a, b])
    obs = [qml.expval(qml.PauliZ(w)) for w in range(n)]
    obs += [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in zz]
    return obs


print(json.dumps([[float(v) for v in circuit(x)] for x in X]))
