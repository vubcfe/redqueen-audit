# redqueen-audit

Code, configurations and run records for the paper

> **Can a Variational Quantum Classifier Be Made Hard to Imitate? An Adversarial
> Audit with Co-Evolving Classical Surrogates.** Cao Vu Bui, FPT University,
> Da Nang. Submitted to *Computers, Materials & Continua* (special issue
> *Quantum Machine Learning: Algorithms, Theory, and Applications*).

RED-QUEEN co-evolves quantum classifier heads (hosts: 8-qubit data
re-uploading circuits on frozen ImageNet features) with budget-limited
classical imitators (parasites: Fourier and random-feature ridge surrogates).
Every selected head is then audited by held-out auditors, including an MLP
distillation network the search never saw, on inputs and budgets disjoint
from the search; the result is summarised by $M_{90}$, the smallest auditor
budget that reaches 90 % agreement with the head.

The hypotheses, primary outcome and decision rule are in
[docs/DESIGN.md](docs/DESIGN.md).

## Layout

| Path | Contents |
|---|---|
| `src/redqueen/sim.py` | batched, differentiable PyTorch state-vector simulator |
| `src/redqueen/genome.py` | host design space, mutation, crossover |
| `src/redqueen/host.py` | quantum head, training, trainability gate |
| `src/redqueen/surrogate.py` | parasites: white-box Fourier and black-box RFF imitators |
| `src/redqueen/coevo.py` | co-evolution loop, hall of fame, MAP-Elites archive; GA (λ = 0) and random search |
| `src/redqueen/classical.py` | classical baselines and held-out auditors ($M_{90}$) |
| `src/redqueen/data.py`, `datasets.py` | feature extraction and label-budget splits |
| `src/redqueen/experiment.py` | one (method, label budget, seed) run → JSON record |
| `src/redqueen/qmlbench.py` | runs on the qml-benchmarks datasets |
| `configs/` | one YAML per study (`paper*.yaml`: real datasets; `qmlbench_paper.yaml`) |
| `scripts/` | analysis (`analyze_paper.py`, `analyze_pooled.py`, `rules.py`), pilot confound check, launchers |
| `paper/` | scripts that produce every figure and every number in the paper (`numbers.txt`) |
| `results/` | all run records used in the paper (see below) |
| `tests/` | simulator, Fourier spectrum, operators; optional cross-check against PennyLane |

## Results

`results/` holds one JSON record per run, as used in the paper:

| Folder | Runs |
|---|---|
| `paper/` | EuroSAT, 90 (6 methods × 3 label budgets × 5 seeds) |
| `paper_so2sat/`, `paper_ucmerced/`, `paper_breastmnist/`, `paper_cifar10/` | 90 each |
| `qmlbench_paper/` | 54 (6 tasks × 3 seeds × 3 searched methods) |
| `pilot/`, `pilot_v2/` | pilot runs behind the measurement confounds (`pilot/budget_confound.txt`) |
| `pooled_analysis.txt` | output of `scripts/analyze_pooled.py` |

A record contains the configuration, the pool sizes, the selected head (genome,
accuracy, macro-F1, audit curves and $M_{90}$ for each auditor), every host
evaluated during the search, the search history and the final parasites.

All statistics and figures in the paper can be reproduced from these records
alone, without a GPU:

```bash
pip install -r requirements.txt
export PYTHONPATH=src
python scripts/analyze_pooled.py              # decision rule (H2), per dataset and pooled
python scripts/analyze_paper.py results/paper # one dataset in detail (H1, H2, summary table)
python paper/make_figures.py                  # Figs. 4-6 and paper/numbers.txt
python paper/make_circuits.py                 # circuit diagrams (quantikz) of Figs. 2-3
```

## Reproducing the runs

Two Python 3.11 environments were used:

- `requirements.txt`: the five real datasets (PyTorch 2.11, CUDA).
- `requirements-qmlbench.txt`: the qml-benchmarks runs, pinned to the versions of
  [XanaduAI/qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks)
  (PennyLane 0.34, JAX 0.4.23), cloned into `external/qml-benchmarks`.

Runs used two NVIDIA RTX 4090 GPUs; one RED-QUEEN run takes about 5 minutes.

### 1. Features

Each dataset is passed once through a frozen ImageNet ResNet and cached in
`features/` (not in this repository: about 490 MB).

| Dataset | Backbone | `--root` must contain |
|---|---|---|
| EuroSAT (multispectral, RGB bands used) | ResNet50 | `train.csv`, `validation.csv`, `test.csv` and the 13-band GeoTIFF tiles they list |
| So2Sat LCZ42 (GEO-Bench release, RGB) | ResNet50 | `train.npz`, `valid.npz`, `test.npz` with `images` (uint8 N×H×W×3) and `labels` |
| UC Merced | ResNet50 | `ucmerced.parquet` with an image column (inline bytes) and a label column |
| BreastMNIST (MedMNIST v2, 224 px) | ResNet18 | `breastmnist_224.npz` (MedMNIST format) |
| CIFAR-10 | ResNet18 | `cifar-10-batches-py/` (python version) |

```bash
export PYTHONPATH=src
python -m redqueen extract --root /path/to/EuroSAT                      # -> features/eurosat_ms_resnet50.npz
python -m redqueen extract --dataset so2sat      --root /path/to/so2sat
python -m redqueen extract --dataset ucmerced    --root /path/to/ucmerced
python -m redqueen extract --dataset breastmnist --root /path/to/breastmnist
python -m redqueen extract --dataset cifar10     --root /path/to/cifar10
```

Splits are the official ones where they exist; CIFAR-10 takes a stratified
10 % of training as validation, and UC Merced is split 70/10/20 (stratified,
seed 0).

### 2. Runs

```bash
python -m pytest -q tests
python -m redqueen run   --config configs/paper.yaml --method redqueen --budget 0.05 --seed 0
python -m redqueen suite --config configs/paper.yaml --skip-existing        # all runs of a config
python -m redqueen suite --config configs/paper.yaml --skip-existing --shard 0/4   # one of 4 shards
python -m redqueen bench --config configs/qmlbench_paper.yaml --skip-existing
```

Methods: `redqueen`, `ga` (same loop, λ = 0), `random` (same objective, random
proposals), `hea_L1`, `hea_L2` (fixed ring-CZ heads), `classical` (logistic
regression, RBF-SVM, parameter-matched MLPs, RFF-logistic, full-embedding
linear probe). `scripts/launch_paper.sh` and `scripts/launch_extra.sh` run all
configurations on two GPUs (`PY=/path/to/python`).

## Citation

If you use this code, please cite the paper (reference to be added on
publication).

## License

MIT, see [LICENSE](LICENSE). The datasets keep their own licenses.
