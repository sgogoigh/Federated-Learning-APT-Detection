# Federated GNN for Lateral-Movement (APT) Detection

Privacy-preserving detection of Advanced Persistent Threat (APT) **lateral movement** with a
federated Graph Neural Network, trained and evaluated on the **LANL "Comprehensive, Multi-Source
Cyber-Security Events"** dataset using its ground-truth red-team labels. A preliminary study on
UNSW-NB15 is retained to motivate the move to a graph-native dataset.

> **Headline result (non-IID, host-partitioned federation, 3-seed mean ± std):** naive federated
> aggregation collapses under realistic label skew (PR-AUC **0.0014**), but a **positive-aware /
> privacy-preserving aggregation** recovers detection to **parity with the centralized model**
> (PR-AUC **0.089**, **76 % of red-team logons detected at a 0.1 % false-positive rate**) — all
> while each organization shares only model updates, never raw authentication logs.

---

## 1. The research arc (and why two datasets)

This repo documents an honest, evidence-driven progression — see `REVELATIONS.md` (numbered
findings R1–R23) for the full story:

1. **UNSW-NB15 (preliminary, `train.py`).** A federated GraphSAGE flow classifier. Finding: on
   per-flow NIDS data the GNN never beats a plain MLP (macro-F1 ≈ 0.39 for both) — the data has no
   lateral-movement structure, so a graph adds nothing, and "APT" is not a defensible claim.
2. **LANL pivot (`lanl_prep.py` + `train_lanl.py`).** Authentication events form a real
   computer→computer graph with genuine red-team APT labels. Here the **graph matters** (GNN ≫
   no-graph baseline), and federation becomes a meaningful, novel setting.
3. **The contribution.** Realistic **non-IID host-community partitioning** (each client = an
   organization that only sees its own hosts) breaks vanilla FedAvg/FedAdam; a **positive-aware /
   local-validation-signal aggregation** fixes it. Validated with multi-seed error bars, an
   ablation, and a dose-response sweep.

## 2. Repository layout

| Path | Role |
|---|---|
| `lanl_prep.py` | Stream `auth.txt.gz`, build time-windowed authentication graphs, label malicious edges from `redteam.txt`. Has `--selftest`. |
| `train_lanl.py` | Federated edge classifier (GraphSAGE → edge MLP), FedAdam, host-community / IID partitioning, aggregation weighting (`edges`/`uniform`/`positives`/`val_signal`), rare-event metrics + baselines. |
| `lanl_experiments.py` | Orchestrates the multi-seed + `pos_smooth` sweep matrix and aggregates to mean ± std. |
| `train.py` | Preliminary UNSW-NB15 federated GraphSAGE pipeline (FedAdam, focal loss, baselines). |
| `LANL_SETUP.md` | How to obtain the LANL data (data-fence) and build the graphs. |
| `DATASET.md` | UNSW-NB15 dataset reference (preliminary study). |
| `REVELATIONS.md` | Empirical findings (R1–R23) — the paper's evidence log. |
| `IMPROVEMENTS.md` | Design history, implemented fixes, and the forward plan (Parts I–IV). |
| `ERRORS.md` | Catalogue of bugs encountered and their fixes. |
| `results/` | Result summaries: `lanl_experiment_summary.{md,json}`, per-run JSONs in `results/exp/`, and the UNSW preliminary reports. (Model binaries, encoders, logs and the datasets are git-ignored.) |

## 3. Setup

```bash
python -m venv .venv && .venv\Scripts\Activate.ps1      # Windows (or: source .venv/bin/activate)
pip install -r requirements.txt
```

## 4. Run the LANL pipeline (main contribution)

**a. Get the data** (see `LANL_SETUP.md`): download `auth.txt.gz` + `redteam.txt.gz` from
<https://csr.lanl.gov/data/cyber1/> into `../datasets/` (or set `DATASET_PATH` in `.env`).

**b. Build the authentication graphs** (days 0–15, hourly snapshots):
```bash
python lanl_prep.py --data_dir ../datasets --t_end 1382400 --bucket 3600 --out_dir ../datasets/lanl_graphs
python lanl_prep.py --selftest          # no data needed: verifies parse/bucket/label/PyG
```

**c. Train one federated configuration** (non-IID host partition, privacy-preserving weighting):
```bash
python train_lanl.py --partition host --agg_weight val_signal --rounds 15 --num_clients 7
# --agg_weight: positives | val_signal (fix) | edges | uniform (broken baselines)
```

**d. Reproduce the full results table** (multi-seed + pos_smooth sweep, resumable):
```bash
python lanl_experiments.py            # writes results/lanl_experiment_summary.{md,json}
```

## 5. Method summary

- **Graph.** Per time-window snapshot: nodes = computers, directed edges = aggregated
  `src→dst` logons with features (count, fail-ratio, #users, #auth-types, off-hours, novelty).
  Edge label = matches a `redteam` event in that window.
- **Model.** Two-layer GraphSAGE node embeddings → edge MLP on `[h_src ‖ h_dst ‖ edge_features]`,
  trained with negative sampling under extreme imbalance (base rate ≈ 4×10⁻⁵).
- **Federation.** Clients = host communities (Louvain on the aggregated auth graph) →
  realistic non-IID. Aggregation = FedAdam with **positive-aware** or **`val_signal`**
  (privacy-preserving: each client reports one local-validation scalar, no labels/counts shared).
- **Evaluation.** Temporal split (train days 0–10, test days 11–15); rare-event metrics —
  ROC-AUC, **PR-AUC**, and **detection rate at fixed low false-positive budgets** (not accuracy/F1,
  which are meaningless at this base rate).

## 6. Key results (LANL official temporal test split, 3-seed mean ± std)

| model (non-IID host) | ROC-AUC | PR-AUC | TPR@0.1 % FPR |
|---|---|---|---|
| no-graph MLP | 0.901 ± 0.009 | 0.0070 ± 0.0044 | 0.221 ± 0.150 |
| centralized GNN | 0.992 ± 0.001 | 0.074 ± 0.014 | 0.705 ± 0.070 |
| federated — edges (naive) | 0.905 ± 0.012 | **0.0014 ± 0.0009** | 0.086 ± 0.060 |
| federated — positives | 0.985 ± 0.003 | 0.081 ± 0.036 | 0.724 ± 0.043 |
| **federated — val_signal (private)** | 0.978 ± 0.010 | **0.089 ± 0.038** | **0.757 ± 0.059** |

Full numbers, the `pos_smooth` dose-response sweep, and per-run JSONs: `results/`.
