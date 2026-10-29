# RED-QUEEN: design and protocol

Target venue: CMC special issue *Quantum Machine Learning: Algorithms, Theory,
and Applications* (deadline 10 March 2027).

Working title: **Red Queen Co-Evolution of Quantum Classifier Heads and
Classical Surrogates for Label-Scarce Multispectral Land-Cover Classification**

## 1. Motivation

- Our ECIS 2026 paper (Bui & Vo) found 8-qubit VQC heads beat a classical
  head by up to 4.3 pp on EuroSAT at 5 % labels, and attributed it to implicit
  regularisation.
- Schreiber et al. (PRL 2023) show that re-uploading models have classical
  surrogates. If a surrogate matches the model, any gain is classical. Only
  generalisation remains a candidate for a quantum-specific effect.
- 2026 audits (Gillani et al.; Hashmi et al.; Fulton et al.) find that most
  reported QML gains disappear against parameter-matched classical controls.
- Prior work uses surrogates *after* the model is built, as a test. No QAS
  method we found uses the surrogate *during* the search.

## 2. Idea (TRIZ #13 "the other way round", #22 "harm into benefit")

Two populations co-evolve (Red Queen / host-parasite dynamics):

- **Hosts**: quantum classifier heads (encoding gate, bandwidth, re-upload
  depth, rotation set, entangling graph, read-out).
- **Parasites**: budget-limited classical surrogates (white-box Fourier
  features on the host's own frequency spectrum; black-box Gaussian random
  Fourier features). They are fitted to *imitate* a host's logits on
  unlabeled inputs. No labels are used.

Every parasite attacks every host at every budget of a fixed panel
`M ∈ {8, 16, 32, 64}` frequencies. The panel is the same for all hosts.
Host fitness = `acc_val + λ · gap`, with
`gap = mean_M (1 − max_parasite agreement_M)`, the area above the imitation
curve. Parasites come from the current population and a hall of fame of
past parasites. Parasite fitness is the mean agreement over hosts and
budgets. The hall of fame and a MAP-Elites archive (descriptors: re-upload
depth × entangling-edge count) prevent cycling and keep diversity.
Offspring are scored against the new panel and survive by (μ + λ) selection.

A trainability gate rejects hosts whose gradient variance at random
initialisation is below a threshold before any training is spent on them.

## 3. Hypotheses

- **H1.** The low-label gain of VQC heads over a *standard* classical head
  shrinks or vanishes against parameter-matched classical heads on the same
  features.
- **H2.** At equal validation accuracy, RED-QUEEN hosts have a larger
  dequantisation gap than hosts found by the same GA with λ = 0 or by
  equal-budget random search. The gap is measured by *held-out auditors*,
  i.e. surrogate families and budgets that the search never used.
- **H3.** Accuracy and gap form a trade-off front. We report its shape,
  whichever way it goes.
- **H4 (exploratory).** Gap correlates with the generalisation gap
  (train − test) under label scarcity.

The paper stands whether H1 and H2 are confirmed or refuted.

## 4. Data protocol

- EuroSAT-MS (13 bands, 64×64), official train/validation/test CSV splits
  (19 317 / 5 519 / 2 759).
- Frozen ImageNet ResNet50 on RGB (B4, B3, B2), images upsampled to 224 →
  2048-d. Also 26-d spectral statistics (per-band mean and std of all 13
  bands), used as a second view.
- PCA to `d = n_qubits` is fitted on the *unlabeled* train pool (labels never
  touched), then standardised.
- Label budgets: 1 %, 5 %, 10 % of train, stratified, seeds 0–4. The labeled
  budget is split 80/20 into search-train and search-val. The official
  validation split is used only as unlabeled inputs. The test split is
  touched once, for final numbers.
- Surrogate imitation uses unlabeled train images, split into disjoint fit and
  eval pools.

## 5. Methods compared (same features, same budgets, same seeds)

| Group | Method |
|---|---|
| Quantum, fixed | ring-CZ hardware-efficient head, L = 1, 2 (ECIS-like) |
| Quantum, searched | random search (equal evaluation budget, same fitness as RED-QUEEN); GA (same operators, λ = 0); **RED-QUEEN** (λ > 0) |
| Classical | logistic regression; MLPs at 100–600 parameters (a bracket over the host sizes the search produces; each host is compared with the nearest size); RBF-SVM (tuned); RFF-logistic at M ∈ {8, 16, 32, 64} (budget-matched to the search panel) |
| Reference | linear probe on the full 2048-d embedding (upper bound, not a competitor) |

## 6. Metrics

- Test accuracy and macro-F1, mean ± std over seeds; Wilcoxon signed-rank
  with Holm correction.
- Held-out auditor imitation curves on test inputs: white-box Fourier,
  black-box RFF and MLP distillation at absolute budgets
  `M ∈ {3, 6, 12, 24, 48, 96, 192}`, off the search panels (the MLP gets as
  many weights as an M-frequency ridge), plus a large budget (M = 2000;
  MLP with 512 hidden units). **M90** is the smallest M reaching 90 %
  agreement; larger means harder to imitate. Not reached within the grid ⇒
  censored at 384 (flagged), which keeps such hosts at the top rank in the
  Wilcoxon test. The white-box Fourier auditor enumerates interactions of
  order ≤ 3 only, so even at M = 2000 it is not a complete model of an
  entangled host.
- Generalisation gap (search-train − test).
- Cost: evaluated hosts, wall-clock time, circuit depth, two-qubit gates.

## 7. Ablations

λ ∈ {0, 0.1, 0.2, 0.4}; with and without hall of fame; with and without the
MAP-Elites archive; with and without the trainability gate; parasite family
(white-box only, black-box only, both); n_qubits ∈ {4, 8}.

## 8. Threats to validity

- Every circuit used here (≤ 8 qubits) is exactly simulable. "Gap" means
  *budget-limited* classical imitability, not computational hardness. We
  state this in the abstract.
- A host can raise its gap by producing noisy, high-frequency outputs. This
  is controlled by the accuracy term and checked post hoc with held-out
  auditors and the Fourier spectrum of the host's outputs.
- Circularity: auditors use a pool disjoint from the search pools and
  budgets off the search panel. The primary auditor (MLP distillation) is a
  family the search never saw; Fourier and RFF auditors share the search
  families and are reported only as in-family checks.
- Selection: the reported host is the best of the *final population*, the
  hosts the last parasites co-evolved against. Earlier hosts are kept for
  analysis but cannot be selected, since their final gap was never contested.
- qml-benchmarks: the audit runs on the first refit model (one of the five
  whose test accuracy is reported), not on the search-time model.
- **Budget confound (found in the pilot, 28 Oct 2026).** With a surrogate
  budget proportional to host size (`M ∝ n_params`), the search shrank the
  host (P = 106 vs 208) to shrink its attacker. Re-audited at equal absolute
  M, its "gap" disappeared (`scripts/budget_confound.py`). All budgets are
  now absolute and identical for every host.
- **Frequency-order confound (found in code review of the pilot).** The
  white-box auditor took a prefix of the spectrum in enumeration order, which
  covers fewer input dimensions for deeper circuits (L = 3: 3 of 8 inputs at
  M = 8; L = 2: 4 of 8). Frequencies are now taken lowest-order and lowest
  harmonic first, random within ties (`surrogate.prioritised`).
- **Single-budget specialisation.** A host optimised against one budget can
  dodge it and still be imitable at the next one (pilot v2: gap at M ≤ 16,
  none at M = 32). Fitness therefore averages over a budget panel.

## 9. Decision rule (fixed before the main runs)

Primary outcome: **MLP-distillation M90** on test inputs. The MLP is the only
auditor family the search never optimised against; all auditors are fitted
on a pool disjoint from the search pools, at budgets
`M ∈ {3, 6, 12, …, 192}` that are off the search panels. Secondary,
labelled in-family: Fourier (white-box) and RFF M90. Also test accuracy.

- **Method claim (H2) holds** if RED-QUEEN's M90 exceeds GA's in a
  one-sided Wilcoxon signed-rank test over the 15 paired (budget, seed) runs
  with p < 0.05, *and* its mean test accuracy is within 1 pp of GA's.
  The same test against random search must also hold for a claim about
  the search operator.
- Otherwise the paper is written as an **adversarial audit**: the best that a
  search targeting inimitability finds is reported as an empirical upper
  bound on budget-limited quantum-specific behaviour on these tasks.
- H1 is reported as is: paired test-accuracy differences between the best
  quantum head and the best classical head with Holm correction.
- A single run whose large-budget RFF or MLP auditor stays below 0.95
  agreement (the order-3 Fourier auditor is excluded from this check) is reported
  and inspected individually; it does not change the rule.

## 10. Additional datasets (added 28 Oct 2026, before any of their runs)

One real dataset is not enough for an audit claim, so the identical protocol
(search, panel, auditors, decision rule) is repeated on four datasets chosen
to match the studies they are compared with:

| Dataset | Domain | Classes | Backbone | Compared with | Label budgets |
|---|---|---|---|---|---|
| So2Sat LCZ42 (RGB, GEO-Bench) | Sentinel-2 | 17 | ResNet50 | Fan et al. 2023/2024 | 1, 5, 10 % |
| UC Merced | aerial RGB | 21 | ResNet50 | Otgonbaatar et al. 2023 | 10, 25, 50 % |
| BreastMNIST (224 px) | medical | 2 | ResNet18 | Hashmi et al. 2026 | 10, 25, 50 % |
| CIFAR-10 | vision | 10 | ResNet18 | Fulton et al. 2026 | 1, 5, 10 % |

Splits: official where they exist; CIFAR-10 val = stratified 10 % of train;
UC Merced stratified 70/10/20 (seed 0). The two small datasets use larger
label fractions so every class keeps several labels. Their unlabeled pools
are shrunk in the 3 : 2 : 4 ratio (sizes recorded per run), so surrogate and
auditor fits at the largest budgets are under-determined and rely on the
ridge penalty selected on the selection pool; this is reported as a
limitation. The decision rule of section 9 is applied per dataset and to the
pooled pairs over all five real datasets.
