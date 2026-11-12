"""Revision R1 (Reviewer 4, point 6b): are the labeled subsets nested across
label budgets for the same seed? Recomputes the split of redqueen.data.prepare
(stratified train_test_split with random_state = seed) from the labels only.

Usage: python scripts/revision/check_nesting.py FEATURES.npz BUDGET [BUDGET ...]
"""
import sys

import numpy as np
from sklearn.model_selection import train_test_split

z = np.load(sys.argv[1], allow_pickle=True)
budgets = [float(b) for b in sys.argv[2:]]
y, split = z["y"].astype(int), z["split"]
tr = np.where(split == "train")[0]
for seed in range(5):
    lab = {b: set(train_test_split(tr, train_size=b, stratify=y[tr], random_state=seed)[0])
           for b in budgets}
    out = []
    for a, b in zip(budgets, budgets[1:]):
        out.append(f"{a:g} in {b:g}: {len(lab[a] & lab[b])}/{len(lab[a])}")
    other = set(train_test_split(tr, train_size=budgets[0], stratify=y[tr],
                                 random_state=seed + 1)[0])
    print(f"seed {seed}: " + "; ".join(out)
          + f" | overlap with seed {seed + 1} at {budgets[0]:g}: {len(lab[budgets[0]] & other)}")
