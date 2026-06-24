# REVELATIONS — What the Training Runs Actually Taught Us

> Created June 2026, after rewriting `train.py` (IMPROVEMENTS.md Part II) and running the
> first honest pipeline: train on `UNSW_NB15_training-set.csv`, evaluate on the official
> held-out `UNSW_NB15_testing-set.csv` (175,341 flows), with centralized MLP and
> centralized-GNN baselines for comparison.

This file records *findings* — the empirical truths the runs exposed. Bugs go in
`ERRORS.md`; the forward plan goes in `IMPROVEMENTS.md`.

---

## R1 — The old headline number was inflated by ~13 accuracy points and ~2× on macro-F1

| Metric | Old (10% split of *train* CSV) | Honest (official test set) |
|---|---|---|
| Accuracy | 82.8% | **70.0%** |
| Weighted F1 | 0.828 | 0.685 |
| Macro F1 | 0.45 (reported) | **0.386** |

The previous "82.8%" was measured on an in-distribution random slice of the training CSV,
where graphs were single-class windows cut in CSV order. Once we (a) fit preprocessing on
train only, (b) build non-leaking similarity graphs, and (c) test on the genuinely held-out
official set, the score fell to its real level. **Weighted accuracy on this dataset is a
vanity metric** — 53% of the test set is Normal and 23% Generic, so a model can score ~70%
while completely failing every rare attack. Macro-F1 is the honest headline.

## R2 — The graph and the federation are not earning their complexity (the decisive result)

Final comparison on the official test set:

| model | accuracy | **macro-F1** | weighted-F1 | AUC |
|---|---|---|---|---|
| `mlp_flow` (per-flow, **no graph, no FL**) | 0.719 | **0.414** | 0.699 | 0.943 |
| `centralized_gnn` (GraphSAGE, no FL) | 0.705 | 0.393 | 0.688 | 0.936 |
| `federated_gnn` (GraphSAGE + FedAvg) | 0.700 | 0.386 | 0.685 | 0.929 |

A plain MLP on the 42 features **beats** GraphSAGE, which in turn beats the federated
version. This is the single most important revelation: **the feature-similarity graph adds
no signal, and federation costs a little accuracy.** Message-passing over kNN edges just
averages already-similar flows — redundant with the raw features. Any future claim that
"the GNN helps" must beat the `mlp_flow` baseline on macro-F1; right now it doesn't.

## R3 — Short runtime (2m 3s) is a *symptom*, not a convenience

The whole run — 2 baselines + 10 rounds × 7 clients × up to 10 local epochs — finished in
123 seconds on CPU. That is suspicious, and the cause is diagnostic:

- **Only 339 training graphs exist** (≈86k oversampled flows ÷ 256 nodes/window). The
  DataLoader batches by *graph* (8 graphs/batch), so each client epoch is ≈4 gradient
  steps. The whole federated run is ≈2,800 tiny updates.
- **Nodes within a graph are kNN-correlated**, so a 256-node "batch" carries far less
  independent signal than 256 i.i.d. samples. The windowed-graph formulation quietly
  collapses ~86k flows into a few hundred weakly-informative training units.
- The model is tiny (2 SAGE layers, hidden 64) and the data fits in RAM, so CPU is plenty.

**Takeaway:** the bottleneck is the *representation and the effective sample count*, not the
compute budget. By contrast the MLP trains on flows in shuffled batches of 512 — vastly more
diverse gradient steps — which is part of why it wins (R2).

## R4 — Early rounds are under-fit; the federation re-climbs from scratch every round

Read from `train-logs.txt`:

- **Round 1, Client 1**: val-loss falls monotonically 1.52 → 0.90 across all 10 epochs and
  val-macroF1 is *still rising* (0.16 → 0.36) at epoch 10 — never plateaus.
- **Only 1 early-stop fired in the entire run** (patience=5). Clients almost never converged
  within their 10-epoch local budget in early/mid rounds.
- **Round 2 collapse**: after Round 1's FedAvg, every client *restarts* at val-macroF1 ≈0.14
  (worse than Round 1's epoch-10 value of 0.36), and Round-2 test macro-F1 drops to ≈0.21
  for all clients before recovering.
- **Round 10**: val-loss is flat at ≈0.55 and val-macroF1 oscillates 0.51–0.53 — fully
  plateaued.

Two root causes behind the collapse / slow climb:
1. **A fresh `Adam` optimizer is created every local round**, so momentum/second-moment
   estimates are thrown away each round — every round restarts optimization cold.
2. **FedAvg averaging + a short 10-epoch local budget** means each round only partially
   re-learns what averaging diluted.

## R5 — Verdict on "more epochs / more rounds"

| Lever | Effect | Why |
|---|---|---|
| More **local epochs** in early rounds | **Helps (real)** | Round 1 was under-fit — still descending at epoch 10. |
| More **rounds** | **Marginal**, saturates ≈round 8–10 | Local test macro-F1 climbed 0.35→0.49 then flattened; Round 10 already plateaued. |
| More of either, to beat the MLP | **No** | Federated (0.386) has already reached the centralized GNN ceiling (0.393); centralized itself is *below* the MLP (0.414). |

**Conclusion:** the model is now essentially *compute-converged near its representational
ceiling*. Spending more epochs/rounds will (a) smooth the Round-2 instability and nudge
federated toward ≈0.39–0.40, but (b) **cannot break past ≈0.40 macro-F1**, because the
ceiling is set by the features/graph, the train→test distribution shift, and the rare-class
sample counts — not by optimization budget. Accuracy gains require changing the
*representation and the data handling*, not the training length (see IMPROVEMENTS.md Part III).

## R6 — Brute oversampling fixes recall but wrecks precision on ultra-rare classes

Per-class recall on the official test set (federated model):

| Class | Recall | Precision | Note |
|---|---|---|---|
| Worms (44 train rows) | **0.82** | **0.02** | Oversampled 44→1500 by duplication+jitter → model over-triggers Worms everywhere. |
| Shellcode (378) | 0.21 | 0.18 | Improved from 0.00 but noisy. |
| Analysis (677) | 0.04 | 0.05 | Still effectively unlearned. |
| Backdoor (583) | 0.03 | 0.05 | Still effectively unlearned. |
| Generic / Normal | 0.98 / 0.95 | high | Majority classes are easy. |

Duplicating 44 Worms samples 34× teaches the model the *44 points*, not the *class* — it
recalls them but smears the decision boundary, tanking precision. Minority performance needs
a smarter approach than brute oversampling (two-stage detection, class merging, threshold
calibration, or real attack data from the raw partitions) — detailed in IMPROVEMENTS.md.

## R7 — There is a large optimism gap between client-local test and the official test

Round-10 *client-local* test macro-F1 averaged ≈0.49, but the *official* test macro-F1 was
0.386. The ~0.10 gap is the combined effect of (a) client-local test sets being drawn from
the **oversampled train distribution**, and (b) the **train→test distribution shift** baked
into the official UNSW-NB15 split (the test set is 2× larger with different class
proportions). **Never trust client-local metrics as the headline** — only the official
held-out set counts.

---

# Run 2 — After the P0-A / cheap-L2 fixes (June 2026)

> Changes since Run 1: FedAdam server optimizer + shared client init; adaptive local epochs
> (25, patience 5); signed-log1p feature transform; class-balanced focal weights; per-round
> official-test monitoring; **best-by-validation checkpointing** (val split flow-level before
> oversampling, so the official test set is never used for model selection).

## R8 — The Round-2 collapse is gone; convergence is now healthy

FedAdam + a shared initial model (all clients start round 1 from the same weights, driven by a
persistent server-side Adam) eliminated the catastrophic Round-2 dip from Run 1 (R4). The
official-test macro-F1 now *climbs* across rounds instead of collapsing, and **early-stopping
fired 44 times vs 1** — confirming the adaptive 25-epoch budget cured the early-round
under-fitting. The L1 (optimization) diagnosis and fix were correct.

## R9 — Federation reached parity; it now costs ≈0

Final comparison (official test, best-val checkpoint):

| model | accuracy | macro-F1 | AUC |
|---|---|---|---|
| `mlp_flow` | 0.752 | 0.391 | 0.946 |
| `centralized_gnn` | 0.729 | 0.386 | 0.940 |
| `federated_gnn` | 0.712 | **0.390** | 0.940 |

All three now sit within noise of each other on macro-F1 (≈0.39). In Run 1 the federated model
*trailed* centralized (0.386 vs 0.393); now it *matches* it. **The cost of federation has been
driven to ≈0 by FedAdam** — exactly the L1 win the plan promised.

## R10 — The representational ceiling is real and still unbroken

Optimization is no longer the bottleneck — and the ceiling didn't move. Every model clusters at
**macro-F1 ≈ 0.39, AUC ≈ 0.94**, and the GNN still does **not** beat the per-flow MLP. This
confirms IMPROVEMENTS Part III's central claim: L1 fixes bring the federated model *up to* the
ceiling but cannot break it. Breaking it requires changing the representation (the real
host/time graph, L2) and the rare-class strategy (two-stage detection, L3) — not more compute.

## R11 — Single-run comparisons are unreliable; the dataset is too small for one seed

The MLP's macro-F1 was 0.414 (Run 1), 0.428 (Run 2a, +log1p), and 0.391 (this run, after a 10%
val holdout shrank its training data). Swings of ~0.03–0.04 from data/seed alone are larger than
the gaps *between models*. With only a few hundred graphs (R3), **any model-vs-model verdict must
be averaged over multiple seeds** before it can be trusted. The headline "GNN ≈ MLP" is robust;
finer rankings are not.

## R12 — Validation macro-F1 is only a loose proxy for official-test macro-F1

Best-by-validation selected round 12 (val 0.470 → official 0.390), but the official-test optimum
was round 5 (0.413). Validation (drawn from the *train* distribution) systematically mispredicts
the official test under UNSW's deliberate train→test shift (R7). Selecting on validation is still
the only honest choice — but the val↔test gap is itself a quantified symptom of the shift, and a
reason a *re-stratified* benchmark (IMPROVEMENTS Part III #13) should be reported alongside.

---

# Run 3 — LANL lateral-movement detection (the pivot pays off, June 2026)

> First federated GNN run on LANL authentication graphs (days 0–15, hourly snapshots, edge-level
> red-team detection). Temporal split: train days 0–10, test days 11–15. Test = **217 malicious
> edges among 5,376,274** (base rate 4.0×10⁻⁵). Metric suite = ROC-AUC, PR-AUC, detection@FPR.

| model | ROC-AUC | PR-AUC | TPR@FPR=1% | TPR@FPR=0.1% | TPR@FPR=0.01% |
|---|---|---|---|---|---|
| no-graph MLP | 0.911 | 0.013 | 0.470 | 0.336 | 0.032 |
| centralized GNN | **0.990** | 0.060 | 0.972 | 0.618 | 0.290 |
| federated GNN | 0.981 | **0.117** | 0.945 | **0.788** | 0.295 |

## R13 — The graph finally earns its place (the core publishable finding)

Unlike UNSW-NB15 (where GNN ≈ MLP, R10), on LANL the GNN **massively beats** the no-graph
baseline: ROC-AUC 0.99 vs 0.91, PR-AUC 0.060 vs 0.013 (**4.6×**), TPR@0.1%FPR 0.62 vs 0.34. This
is the whole thesis made empirical: **lateral movement is a graph property** — a single logon looks
benign; the malicious signal is in the connectivity pattern, which only message passing can see.
A non-graph model is structurally blind to it. *This* is the contribution UNSW-NB15 could never
provide.

## R14 — Federation does not hurt — and here it helped detection

The federated GNN matches the centralized one on ROC-AUC (0.981 vs 0.990) and is actually
**better on the operational metrics** — PR-AUC 0.117 vs 0.060 and TPR@0.1%FPR **0.788 vs 0.618**.
Best-val checkpointing (round 9) likely acted as regularization the fixed-epoch centralized run
didn't get; treat the *direction* cautiously pending multi-seed (R11), but the safe, strong claim
holds: **privacy-preserving FedAdam federation reaches centralized-grade lateral-movement detection
without sharing raw auth logs.** That is a legitimate FL contribution.

## R15 — The operational number is the story, not PR-AUC's absolute value

At a 4×10⁻⁵ base rate, PR-AUC 0.117 is ~2,900× random — but the line that lands in a paper/SOC is
**"detects 79% of red-team lateral-movement authentications at a 0.1% false-positive rate"**
(94% at 1% FPR). That is a genuinely strong, deployable operating point and the right way to report
this task. Accuracy/F1 would have been meaningless here (predict-all-benign = 99.996% accurate).

## R16 — Healthy convergence, but val→test gap and remaining caveats

Val PR-AUC climbed smoothly to a round-9 peak (0.223 → test 0.117) with no collapse (FedAdam +
shared init working as on UNSW). Caveats for the paper, all addressable: (a) the val→test PR-AUC gap
(~0.22→0.12) means selection is still noisy with ~36 val positives; (b) the federation is currently
**IID over time-snapshots** (round-robin), not the non-IID-by-host/domain partition the FL story
needs — LANL is single-domain (DOM1), so cross-org must be *simulated* by host-community partition;
(c) single seed — multi-seed mean±std required before the federated>centralized claim is trustworthy;
(d) only days 0–15 of 58 used.

---

*Living document. Append a new R-entry after every substantive run.*
