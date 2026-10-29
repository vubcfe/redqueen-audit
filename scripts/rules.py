"""Statistics shared by analyze_paper.py, analyze_pooled.py and paper/make_figures.py,
so that the decision rule and the selection of the "best" head are defined once."""

import numpy as np


def rule_holds(p: float, dacc: float) -> bool:
    """DESIGN.md section 9: one-sided p < 0.05 and mean test accuracy within 1 pp.
    "Within" is read as inclusive; the rounding keeps a mean of exactly -1.00 pp
    (e.g. -63 of 6300 test predictions) from failing on float error."""
    return p < 0.05 and round(dacc * 100, 6) >= -1.0


def holm(ps) -> np.ndarray:
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def val_pick(pairs) -> float:
    """Test accuracy of the candidate with the best validation accuracy, given
    (acc_val, acc) pairs; tied candidates are averaged, so none wins by position."""
    top = max(v for v, _ in pairs)
    return float(np.mean([a for v, a in pairs if v == top]))
