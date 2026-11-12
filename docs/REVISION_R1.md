# Revision R1 (CMC manuscript 96001)

Analyses and runs added in response to the first round of review. Everything
here is additive: the code paths of the original runs are unchanged (checked by
re-running a stored search with the old and the new code in the same
environment: identical histories, selected heads and audits), and the stored
records in `results/paper*` are untouched.

## Environment

The real-dataset runs of the paper used `py311env` (PyTorch 2.11, NumPy 2.4)
with `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`. In that environment, retraining a
stored head from its genome and seed reproduces its test accuracy and its MLP
audit exactly (`scripts/revision/check_reproduce.py`). With other library
versions or thread counts the floating-point results differ slightly and the
150 training steps at learning rate 0.05 amplify the difference (test accuracy
of individual heads can move by a few points), so use the same setup to
reproduce the records bit for bit.

## Code changes

| Change | Where | Reviewer |
|---|---|---|
| `two_qubit_gates_effective = (L - 1) * |E|`: the last entangling layer is not needed (CZ commutes with the Z read-out; CNOT is a basis permutation) | `genome.py`, recorded by `experiment.py` | R3.2 |
| `QuantumHead.expectations(..., final_entangler=False)` (verification only) and two unit tests | `host.py`, `tests/test_core.py` | R3.2 |
| `parasite_evolve: false` and method `redqueen_fixedpar`: RED-QUEEN with the initial parasites frozen | `coevo.py`, `runutil.py`, `experiment.py` | R4.2 |

## New runs (`configs/revision/`, `results/revision/`)

| Study | Configs | Runs |
|---|---|---|
| fixed parasites | `fixedpar_<dataset>.yaml` | 5 datasets x 3 budgets x 5 seeds = 75 |
| doubled search budget (296 hosts), RED-QUEEN / GA / random, middle label budget | `gen20_<dataset>.yaml` | 5 x 3 x 5 = 75 |
| lambda in {0.1, 0.5, 1.0}, RED-QUEEN, middle label budget | `lam<value>_<dataset>.yaml` | 3 x 5 x 5 = 75 |

Launch: `scripts/revision/launch_r1.sh "0 0 1"` (one worker per listed GPU,
resumable). On the lab server `scripts/revision/watchdog.sh` runs from cron
every 10 minutes and at boot, and relaunches stopped or stalled workers.

## Analyses

| Script | Content | Reviewer |
|---|---|---|
| `scripts/revision/analyze_r1.py` | censoring counts; H2 with a sign test, fixed-budget agreement, inimitability area and alternative censor encodings; paired accuracy differences with 95 % and 90 % bootstrap intervals; trade-off correlations by source; effective gate counts; spectrum sizes | R4.4, R4.5, R4.6, R3.2 |
| `scripts/revision/check_nesting.py` | nesting of label subsets across budgets | R4.6b |
| `scripts/revision/reaudit_mlp.py`, `analyze_reaudit.py` | MLP auditor: 5 initialisations x 3 learning rates x 4 checkpoints (to 10^4 steps), selection on the audit pool's selection split; fit-set sizes 124 / 336 / 1000 on the large datasets | R4.3 |
| `scripts/revision/analyze_search_r1.py` | fixed parasites, doubled budget, lambda | R4.1, R4.2 |

Outputs: `results/revision/*.txt`.

Outputs of the final analyses: `results/revision/analyze_r1.txt`,
`analyze_reaudit.txt`, `analyze_search_r1.txt`, `check_nesting_*.txt`.

## Notes from the runs

- The lab server rebooted once (2026-11-11 19:10); the cron watchdog
  relaunched the workers and the runs resumed from the finished records.
- Five `gen20_cifar10` runs (all in shard 2/3) died silently at start while
  the CIFAR-10 re-audit was running on the same machine (most likely memory
  pressure); the watchdog relaunched them and they completed. Records are
  written atomically, so no partial record was used.
- The re-audit's batched trainer does not reproduce the stored audit's
  initial weights; its stored-setting curves behave like another
  initialisation (standardised difference mean -0.01).
