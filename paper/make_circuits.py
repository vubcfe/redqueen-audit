"""Quantikz circuit diagrams for the manuscript, generated from host genomes so
that the drawn circuit is exactly the one that was trained (gate order as in
src/redqueen/host.py: encoding, trainable rotations, entanglers in edge order;
one block is drawn and repeated L times; read-out after the last block).

Usage (from the project root): python paper/make_circuits.py
Writes paper/circuit_*.tex, which main.tex \\input s.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from redqueen.genome import HostGenome, fixed_hea  # noqa: E402

OUT = Path("paper")
ROT = {"RX": "R_X", "RY": "R_Y", "RZ": "R_Z"}
BAND = {0.25: r"\tfrac14", 0.5: r"\tfrac12", 1.0: "", 2.0: "2"}


def entangler_columns(g: HostGenome) -> list[list[tuple[int, int]]]:
    """Pack the edges into columns in their original order. A gate goes into the
    earliest column after the last one used by any wire in its vertical span,
    so drawn lines never cross another gate and gates sharing a wire keep their order."""
    cols, last = [], [-1] * g.n
    for a, b in g.edges:
        lo, hi = min(a, b), max(a, b)
        c = max(last[lo:hi + 1]) + 1
        while len(cols) <= c:
            cols.append([])
        cols[c].append((a, b))
        for w in range(lo, hi + 1):
            last[w] = c
    return cols


def quantikz(g: HostGenome, label: str, group_labels: bool, readout_label: str,
             repeat: str | None = None, ent_cols: int = 0) -> str:
    """``ent_cols`` pads the entangler with empty wire columns, so that circuits
    drawn side by side for comparison have the same width (no gates added)."""
    n, rots, ent = g.n, g.rot_gates, entangler_columns(g)
    ent = ent + [[]] * max(0, ent_cols - len(ent))
    rows = [[rf"\lstick{{$x_{{{w + 1}}}\;\ket{{0}}$}}"] for w in range(n)]
    enc = ROT[g.enc_gate]
    bw = BAND[g.bandwidth]
    for w in range(n):
        rows[w].append(rf"\gate{{{enc}({bw}x_{{{w + 1}}})}}")
    for k, r in enumerate(rots):
        for w in range(n):
            rows[w].append(rf"\gate{{{ROT[r]}(\theta_{{{w + 1},{k + 1}}})}}")
    for col in ent:
        cells = [r"\qw"] * n
        if not col:   # padding column: an empty node as wide as a column of CZ dots
            cells = [r"|[linecont,minimum width=0.4em,inner sep=0pt]| {} \qw"] * n
        for a, b in col:
            if g.ent_gate == "CZ":
                cells[a], cells[b] = rf"\ctrl{{{b - a}}}", r"\control{}"
            else:
                cells[a], cells[b] = rf"\ctrl{{{b - a}}}", r"\targ{}"
        for w in range(n):
            rows[w].append(cells[w])
    for w in range(n):
        rows[w].append(r"\meter{}")
    rows[0].append(rf"\rstick[{n}]{{\footnotesize {readout_label}}}")
    for w in range(1, n):
        rows[w].append("")
    # a light background per part of the block (no borders), and one frame
    # around the whole block that is repeated L times
    n_rot = len(rots)
    first = rows[0]
    part = ("{{draw=none,rounded corners=2pt,fill={fill},inner xsep=0.5pt,inner ysep=2pt}},"
            "background,label style={{label position=above,anchor=south,yshift=0.12cm}}")
    first[1] += (rf"\gategroup[{n},steps=1,style={part.format(fill='orange!12')}]"
                 rf"{{{r'\textit{encoding}' if group_labels else ''}}}")
    first[2] += (rf"\gategroup[{n},steps={n_rot},style={part.format(fill='blue!8')}]"
                 rf"{{{r'\textit{trainable}' if group_labels else ''}}}")
    if ent:
        first[2 + n_rot] += (rf"\gategroup[{n},steps={len(ent)},style={part.format(fill='green!10')}]"
                             rf"{{{r'\textit{entangler}' if group_labels else ''}}}")
    first[1] += (rf"\gategroup[{n},steps={1 + n_rot + len(ent)},style={{draw=black!45,rounded corners=3pt,"
                 rf"inner xsep=2pt,inner ysep=5pt}},label style={{label position={'below' if group_labels else 'above'},"
                 rf"anchor={'north' if group_labels else 'south'},yshift={'-0.05cm' if group_labels else '0.05cm'}}}]"
                 rf"{{{repeat or f'block repeated $L={g.layers}$ times'}}}")
    body = " \\\\\n".join(" & ".join(r) for r in rows)
    return (f"% {label}: {g.key()}\n"
            r"\begin{quantikz}[row sep={0.68cm,between origins}, column sep=0.22cm, font=\footnotesize]" + "\n"
            + body + "\n\\end{quantikz}\n")


def best_genome(path: str) -> tuple[HostGenome, dict]:
    r = json.loads(Path(path).read_text())
    return HostGenome.from_dict(r["best"]["genome"]), r["best"]


# Fig. method: the fixed ring-CZ head with the evolvable parts marked
hea = fixed_hea(8, 1)
(OUT / "circuit_template.tex").write_text(
    quantikz(hea, "fixed HEA L=1", True, r"linear $\to$ logits",
             repeat=r"block repeated $L$ times"))

# Fig. results: GA and RED-QUEEN heads selected in the same run
pair = {m: best_genome(f"results/paper/{m}_b0.05_s0.json") for m in ("ga", "redqueen")}
width = max(len(entangler_columns(g)) for g, _ in pair.values())
for m, (g, b) in pair.items():
    ro = r"linear $\to$ logits"
    (OUT / f"circuit_{m}.tex").write_text(quantikz(g, m, False, ro, ent_cols=width))
    print(f"{m}: {g.key()} acc {b['acc']:.4f} MLP M90 {b['audit']['m90']['mlp']} params {b['n_params']} "
          f"2q gates {b['two_qubit_gates']} entangler columns {len(entangler_columns(g))}")
