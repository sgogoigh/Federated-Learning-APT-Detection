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

*Living document. Append a new R-entry after every substantive run.*
