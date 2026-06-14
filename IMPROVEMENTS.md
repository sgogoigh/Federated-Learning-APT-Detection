# APT Detection — Federated GraphSAGE: Improvement Plan

> **Last updated**: June 2026  
> **Status**: Training complete (3 rounds / 5 clients). Re-training pending with fixes applied.

---

## Table of Contents

1. [Current State of the Project](#1-current-state-of-the-project)
2. [What Is Missing or Inadequate](#2-what-is-missing-or-inadequate)
3. [All Required Changes](#3-all-required-changes)
4. [Complete Updated Project Flow](#4-complete-updated-project-flow)
5. [Expected Outcome After Changes](#5-expected-outcome-after-changes)

---

## 1. Current State of the Project

### 1.1 Repository Structure

```
Federated-Learning-APT-Detection/
├── apt-fl-model.ipynb          # Main training notebook (Federated GraphSAGE)
├── test_model_final.py         # Full inference script (CSV + model)
├── test_model_final_cpu.py     # Lightweight CPU inference (no CSV needed)
├── test_multiple.py            # Batch synthetic + real sample inference
├── lanl_conversion.ipynb       # ABANDONED — LANL dataset exploration
├── unzip.py                    # Utility to unzip results.zip
├── requirements.txt            # Python dependencies
├── IMPROVEMENTS.md             # This file
├── results/
│   ├── global_model.pt         # Trained model weights (46 KB)
│   ├── metrics.json            # Per-client, per-round training metrics
│   ├── global_model_metrics.json # Final aggregated evaluation metrics
│   ├── confmat_round1.png      # Confusion matrix — Round 1
│   ├── confmat_round2.png      # Confusion matrix — Round 2
│   └── confmat_round3.png      # Confusion matrix — Round 3
└── data/
    └── UNSW_NB15_reduced_features.csv  # ~300MB, NOT committed to git
```

### 1.2 Model Architecture

The model is a **GraphSAGE classifier** implemented in PyTorch Geometric:

```python
class GraphSAGEClassifier(nn.Module):
    def __init__(self, in_dim=14, hidden_dim=64, num_classes=10, dropout=0.3):
        self.conv1 = SAGEConv(in_dim, hidden_dim)     # Neighbourhood aggregation layer 1
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)  # Neighbourhood aggregation layer 2
        self.fc    = nn.Linear(hidden_dim, num_classes) # Classification head
        self.dropout = nn.Dropout(dropout)
        self.relu    = nn.ReLU()

    def forward(self, data):
        x = relu(conv1(x, edge_index))
        x = dropout(x)
        x = relu(conv2(x, edge_index))
        x = global_mean_pool(x, batch)   # Graph-level pooling
        return fc(dropout(x))
```

**Input features (14 total)**:

| Feature | Description |
|---|---|
| `sport` | Source port |
| `dsport` | Destination port |
| `dur` | Flow duration |
| `sbytes` | Source bytes sent |
| `dbytes` | Destination bytes sent |
| `Spkts` | Source packets |
| `Dpkts` | Destination packets |
| `Sintpkt` | Source inter-packet time |
| `ct_srv_src` | Connection count to same src service |
| `ct_srv_dst` | Connection count to same dst service |
| `ct_dst_ltm` | Connection count to same dst in last T seconds |
| `hour_of_day` | Hour extracted from timestamp |
| `proto_enc` | Protocol (label-encoded) |
| `service_enc` | Service type (label-encoded) |

**Output**: 10-class classification over UNSW-NB15 attack categories:  
`Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Normal, Reconnaissance, Shellcode, Worms`

### 1.3 Graph Construction

Each graph is built by grouping network flows by `(hour_of_day, service)`, then chunked into subgraphs of up to 120 nodes each. Edges are drawn between nodes that share the same protocol encoding OR the same service encoding OR have similar source/destination port values (difference < 1e-3 after StandardScaler normalisation). The majority attack class in each subgraph becomes its graph-level label.

### 1.4 Federated Learning Setup

- **Algorithm**: FedAvg (plain averaging of client model weights)
- **Clients**: 5 (split via KMeans on graph feature centroids)
- **Rounds**: 3
- **Local epochs**: ~5 per round (with early stopping, patience=4)
- **Local LR**: 1e-3, weight decay 1e-4
- **Batch size**: 8 graphs per batch
- **Class weighting**: Inverse-frequency weights in CrossEntropyLoss (already present)

### 1.5 Training Results (Current — 3 Rounds, 5 Clients, nrows=5000)

#### Per-Client Results — Round 1

| Client | Samples | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|---|
| Client1 | 236 | 99.56% | 99.12% | 99.56% | 99.34% | **0.0 (bug)** |
| Client2 | 22 | 95.45% | 91.12% | 95.45% | 93.23% | **0.0 (bug)** |
| Client3 | 3 | 33.33% | 11.11% | 33.33% | 16.67% | **0.0 (bug)** |
| Client4 | 73 | 94.52% | 89.34% | 94.52% | 91.86% | **0.0 (bug)** |
| Client5 | 57 | 96.49% | 93.11% | 96.49% | 94.77% | **0.0 (bug)** |

#### Final Global Model (Round 3)

| Metric | Value |
|---|---|
| Accuracy | 84.47% |
| Precision | 78.08% |
| Recall | 84.47% |
| F1 Score | 80.11% |
| AUC | **0.0 (bug — never computed)** |

#### Confusion Matrix Diagnosis

All three confusion matrices tell the same story: **the model predicts almost everything as class "Normal"**. In Rounds 2 and 3, 372–373 samples are correctly predicted as Normal, while Fuzzers, Exploits, and Reconnaissance each get at most 1 correct prediction. This is called **class collapse** — the model learned to predict the majority class for all inputs.

---

## 2. What Is Missing or Inadequate

### 2.1 Critical Bugs

#### BUG-01 — `torch.cuda.set_device(0)` called unconditionally
**File**: `apt-fl-model.ipynb` — Cell 1  
**Severity**: CRITICAL — crashes immediately on any machine without a GPU  
**Code**:
```python
# Current (broken)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)   # <-- crashes if no GPU
```
Despite correctly detecting the device one line above, the notebook then unconditionally calls `torch.cuda.set_device(0)`. On a CPU-only machine this throws `AssertionError: Invalid device id`. The notebook was developed on Google Colab (GPU) and never tested locally.

---

#### BUG-02 — AUC is always 0.0 (never actually computed)
**File**: `apt-fl-model.ipynb` — `evaluate()` function  
**Severity**: HIGH — a core metric is silently broken  
**Code**:
```python
# Current (broken)
try:
    auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr")
except Exception:
    auc = 0.0    # <-- silently swallowed every single time
```
`roc_auc_score` with `multi_class="ovr"` throws a `ValueError` whenever the test set does not contain samples from every class. Because the dataset is split into tiny per-client test sets (e.g. Client3 has only 3 total samples), the condition is never met, and `0.0` is returned every time. The fix is to pass `labels=list(range(num_classes))` so sklearn knows all valid labels, and to zero-pad missing class probabilities.

---

#### BUG-03 — Final classification layer (`fc.`) not loaded in test scripts
**Files**: `test_model_final.py`, `test_multiple.py`  
**Severity**: CRITICAL — every test-time prediction is garbage  
**Code**:
```python
# Current (broken) — in both test scripts
model_dict = net.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
model_dict.update(filtered_dict)
net.load_state_dict(model_dict, strict=False)
```
This intentionally skips loading the `fc.` (output head) layer. As a result, the classification head retains random initialisation weights on every run — the model is making entirely random predictions regardless of what it learned during training. `test_model_final_cpu.py` does NOT have this bug (it uses `strict=False` without filtering).

---

### 2.2 Configuration Problems

#### CONFIG-01 — Dataset capped at 5,000 rows
**File**: `apt-fl-model.ipynb` — `main()` function  
**Impact**: HIGH — root cause of class collapse  
```python
df = pd.read_csv(args.csv, low_memory=False)   # <-- no cap in main()
```
But the CSV path passed to `main()` was loaded with `nrows=5000` in earlier notebook cells. With only 5,000 rows of a heavily imbalanced dataset (Normal class >> attack classes), the 7-client KMeans split gives some clients near-zero minority class samples. Client3 received only 3 total samples — it cannot learn anything meaningful from this.

The full UNSW-NB15 dataset has ~257k rows across 10 attack categories. Loading the full file will provide each client with many more minority-class examples.

#### CONFIG-02 — Only 3 federated rounds
**File**: `apt-fl-model.ipynb` — `main()` call  
**Impact**: MEDIUM — model has not had time to converge  
With only 3 rounds of federation and early stopping at patience=4, the global model has seen very limited cross-client knowledge exchange. 10 rounds allows the global model to benefit from the accumulated gradients of all clients across many iterations.

#### CONFIG-03 — Only 5 clients
**File**: `apt-fl-model.ipynb` — `main()` call  
**Impact**: MEDIUM — fewer clients = less federated diversity  
7 clients creates a more representative federated scenario and better stress-tests the FedAvg aggregation. The KMeans clustering will produce 7 data partitions based on feature centroids, creating more natural non-IID distributions.

---

### 2.3 Evaluation Gaps

#### EVAL-01 — No per-round learning curve
There are confusion matrices per round, but no plot showing how accuracy/F1/AUC evolves across rounds at the global level. This is essential for understanding federation convergence.

#### EVAL-02 — No global test set evaluation
Each round only evaluates on each client's own local test split. There is no held-out global test set that measures the global model's performance across the entire dataset.

#### EVAL-03 — No per-class breakdown in metrics.json
The metrics only store aggregate accuracy/precision/recall/F1/AUC. There is no per-class precision/recall breakdown, which would immediately expose the class collapse problem in a quantitative way.

#### EVAL-04 — `global_model_metrics.json` is averaged client metrics, not a real global eval
The current `global_model_metrics.json` averages client test results together. This is not a true global evaluation — it is a (client-count) weighted average of heterogeneous local test sets.

---

### 2.4 Code Quality Issues

#### CODE-01 — Duplicate `preprocess_df` function definition
The preprocessing function `preprocess_df` is defined **twice** in the notebook (cells 2 and 4) with identical code. Cell 4 is the one actually used. Cell 2 is dead code.

#### CODE-02 — `build_graphs` is defined but never executed in the captured output
Cell 3 (`build_graphs`) has `execution_count: null` — it was never run during the saved session. The function is defined in later code that was re-run. This is confusing notebook hygiene.

#### CODE-03 — Training ran on Google Colab (GPU), not locally
The saved notebook outputs reference `cuda:0` and CUDA 12.8 libraries. The notebook is not portable to CPU-only machines without the BUG-01 fix.

#### CODE-04 — No saved `LabelEncoder` (attack_enc)
The `LabelEncoder` that maps integer predictions back to attack category names is re-fitted from the CSV at test time. This means:
1. The test scripts MUST have access to the CSV to decode predictions
2. If the dataset encoding order changes (different `nrows`, different class distribution), the mapping could change and produce wrong labels
The encoder's `classes_` array should be saved alongside `global_model.pt`.

#### CODE-05 — `test_model_final_cpu.py` hardcodes class order
```python
attack_enc.classes_ = np.array([
    "Normal", "Exploits", "Reconnaissance", "Fuzzers",
    "DoS", "Generic", "Analysis", "Backdoor", "Shellcode", "Worms"
])
```
This hardcoded order may not match the alphabetical `LabelEncoder` order that training actually used. LabelEncoder sorts classes alphabetically, so the real order is: `Analysis(0), Backdoor(1), DoS(2), Exploits(3), Fuzzers(4), Generic(5), Normal(6), Reconnaissance(7), Shellcode(8), Worms(9)`. The hardcoded order in `test_model_final_cpu.py` puts Normal at index 0, which means predictions above index 0 will be mapped to the wrong attack name.

---

### 2.5 Missing Features

#### FEAT-01 — No data/ directory in the repository
The dataset `UNSW_NB15_reduced_features.csv` (~300 MB) is not committed and there is no script to download or prepare it. New contributors cannot reproduce results without external knowledge.

#### FEAT-02 — No download/setup script
There is no `setup.py`, `Makefile`, or shell script that automates the full reproducibility chain (install deps → fetch data → run training → test).

#### FEAT-03 — No training curves / round-over-round plots
Beyond the per-round confusion matrices, there are no time-series plots showing loss, accuracy, or F1 evolving across epochs within each round, or across rounds globally.

#### FEAT-04 — AUC ROC curves not plotted
Even after fixing the AUC computation bug, the numeric AUC score will be saved but there are no actual ROC curve plots (one per attack class, using OvR strategy) to visually inspect model discrimination ability.

---

## 3. All Required Changes

### Priority 1 — Bug Fixes (must be done before re-training)

#### Fix BUG-01: Guard `cuda.set_device`

**File**: `apt-fl-model.ipynb` — Cell 1

```python
# BEFORE
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

# AFTER
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
```

---

#### Fix BUG-02: Compute AUC correctly

**File**: `apt-fl-model.ipynb` — `evaluate()` function

Change the function signature to accept `num_classes`:

```python
# BEFORE
def evaluate(model, loader):
    ...
    try:
        auc = roc_auc_score(y_true, np.array(y_prob), multi_class="ovr")
    except Exception:
        auc = 0.0
    return acc, prec, rec, f1, auc, (y_true, y_pred)

# AFTER
def evaluate(model, loader, num_classes):
    ...
    try:
        if len(set(y_true)) < 2:
            auc = 0.0  # Cannot compute AUC with only one class present
        else:
            auc = roc_auc_score(
                y_true,
                np.array(y_prob),
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes))
            )
    except Exception as e:
        print(f"  [AUC warning] {e}")
        auc = 0.0
    return acc, prec, rec, f1, auc, (y_true, y_pred)
```

Update all callers of `evaluate()` to pass `num_classes`:
```python
# In train_local()
val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader, num_classes)
acc, prec, rec, f1, auc, (yt, yp) = evaluate(model, test_loader, num_classes)
```

---

#### Fix BUG-03: Load full state_dict in test scripts

**File**: `test_model_final.py`

```python
# BEFORE — skips fc. layer, randomises output head
model_dict = net.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
model_dict.update(filtered_dict)
net.load_state_dict(model_dict, strict=False)

# AFTER — load the complete saved model
net.load_state_dict(state_dict, strict=False)
```

**File**: `test_multiple.py` — same fix as above.

---

### Priority 2 — Configuration Changes

#### Change CONFIG-01: Remove `nrows` cap

**File**: `apt-fl-model.ipynb` — `main()` function and the notebook cell that calls it

```python
# BEFORE
df = pd.read_csv(args.csv, low_memory=False, nrows=5000)

# AFTER — load the full ~257k row dataset
df = pd.read_csv(args.csv, low_memory=False)
```

Also update any test scripts that reference `nrows`:
```python
# test_model_final.py and test_multiple.py
# BEFORE
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=True, nrows=5000)

# AFTER
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=True)
```

---

#### Change CONFIG-02 & CONFIG-03: 10 Rounds, 7 Clients

**File**: `apt-fl-model.ipynb` — `main()` call at end of notebook

```python
# BEFORE
args = argparse.Namespace(
    csv="data/UNSW_NB15_reduced_features.csv",
    num_clients=5,
    rounds=3,
    epochs_local=10,
    hidden_dim=64
)

# AFTER
args = argparse.Namespace(
    csv="data/UNSW_NB15_reduced_features.csv",
    num_clients=7,
    rounds=10,
    epochs_local=10,
    hidden_dim=64
)
```

---

### Priority 3 — Code Quality Fixes

#### Fix CODE-04: Save `LabelEncoder` alongside the model

**File**: `apt-fl-model.ipynb` — inside `main()`, after training completes

```python
import pickle

# After torch.save(global_state, "results/global_model.pt")
with open("results/attack_encoder.pkl", "wb") as f:
    pickle.dump(attack_enc, f)
print("Label encoder saved to results/attack_encoder.pkl")
```

Update test scripts to load the saved encoder instead of re-fitting from CSV:
```python
# BEFORE (in test scripts)
le_attack = LabelEncoder()
df["attack_label"] = le_attack.fit_transform(df["attack_cat"].astype(str))
attack_enc = le_attack

# AFTER
import pickle
with open("results/attack_encoder.pkl", "rb") as f:
    attack_enc = pickle.load(f)
num_classes = len(attack_enc.classes_)
```

---

#### Fix CODE-05: Correct hardcoded class order in `test_model_final_cpu.py`

```python
# BEFORE (incorrect order — Normal is not index 0 after alphabetical LabelEncoding)
attack_enc.classes_ = np.array([
    "Normal", "Exploits", "Reconnaissance", "Fuzzers",
    "DoS", "Generic", "Analysis", "Backdoor", "Shellcode", "Worms"
])

# AFTER (alphabetical — matches LabelEncoder output)
attack_enc.classes_ = np.array([
    "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
    "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
])
```

Or better — load from the saved pickle (see CODE-04 fix above).

---

#### Fix CODE-01: Remove duplicate `preprocess_df` definition

Delete Cell 2 from the notebook (the first, never-executed definition of `preprocess_df`). Cell 4 is the active one.

---

### Priority 4 — New Evaluation Features

#### Add EVAL-01: Per-round learning curve plot

After all rounds complete, add to `main()`:

```python
import matplotlib.pyplot as plt

rounds = list(range(1, args.rounds + 1))
avg_accs, avg_f1s, avg_aucs = [], [], []

for r in range(1, args.rounds + 1):
    round_metrics = allres[f"Round{r}"]
    clients = list(round_metrics.values())
    avg_accs.append(np.mean([c["acc"] for c in clients]))
    avg_f1s.append(np.mean([c["f1"] for c in clients]))
    avg_aucs.append(np.mean([c["auc"] for c in clients]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, values, label, color in zip(
    axes,
    [avg_accs, avg_f1s, avg_aucs],
    ["Accuracy", "F1 Score", "AUC"],
    ["steelblue", "darkorange", "green"]
):
    ax.plot(rounds, values, marker="o", color=color)
    ax.set_title(f"Global {label} per Round")
    ax.set_xlabel("Federated Round")
    ax.set_ylabel(label)
    ax.set_xticks(rounds)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/learning_curves.png", dpi=150)
plt.show()
print("Learning curves saved to results/learning_curves.png")
```

#### Add EVAL-02: Global test set evaluation

Maintain a held-out 10% global test split before creating clients:

```python
def main(args):
    df = pd.read_csv(args.csv, low_memory=False)
    df, attack_enc = preprocess_df(df)
    graphs = build_graphs(df)

    # NEW: hold out 10% as a global test set
    random.shuffle(graphs)
    split = int(0.9 * len(graphs))
    train_graphs, global_test_graphs = graphs[:split], graphs[split:]

    clients = create_clients(train_graphs, args.num_clients)
    global_test_loader = DataLoader(global_test_graphs, batch_size=8)

    # ... training loop ...

    # After all rounds:
    in_dim = train_graphs[0].x.shape[1]
    num_classes = len(attack_enc.classes_)
    global_net = GraphSAGEClassifier(in_dim, config["hidden_dim"], num_classes).to(DEVICE)
    global_net.load_state_dict(global_state)
    global_net.eval()

    g_acc, g_prec, g_rec, g_f1, g_auc, (g_yt, g_yp) = evaluate(
        global_net, global_test_loader, num_classes
    )
    print(f"\nGlobal Test Set: acc={g_acc:.4f} f1={g_f1:.4f} auc={g_auc:.4f}")
```

#### Add EVAL-03: Per-class metrics breakdown

After training, add classification report:

```python
from sklearn.metrics import classification_report

report = classification_report(
    g_yt, g_yp,
    target_names=attack_enc.classes_,
    zero_division=0
)
print("\nPer-class Classification Report:")
print(report)

with open("results/classification_report.txt", "w") as f:
    f.write(report)
```

---

## 4. Complete Updated Project Flow

The following describes the full project pipeline after all changes are applied.

### Step 0 — Environment Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1     # Windows
# source .venv/bin/activate    # Linux/macOS

# 2. Install standard dependencies
pip install -r requirements.txt

# 3. Install PyG extensions (pre-built wheels, CPU version)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.4.1+cpu.html
```

### Step 1 — Dataset Preparation

Place the dataset at:
```
data/UNSW_NB15_reduced_features.csv
```

The dataset must contain these columns:
`sport, dsport, proto, service, dur, sbytes, dbytes, Spkts, Dpkts, Sintpkt, ct_srv_src, ct_srv_dst, ct_dst_ltm, hour_of_day, attack_cat, Label`

### Step 2 — Preprocessing (inside `apt-fl-model.ipynb`)

`preprocess_df(df)` performs:
1. Strip whitespace from column names
2. Fill empty `attack_cat` with `"Normal"`
3. Numeric coerce + fillna for `Label`, `hour_of_day`, and all numeric columns
4. **StandardScaler** normalisation on the 12 numeric feature columns
5. **LabelEncoder** on `proto` → `proto_enc`, `service` → `service_enc`
6. **LabelEncoder** on `attack_cat` → `attack_label` (alphabetical, 10 classes)
7. Return the processed DataFrame and the fitted `le_attack` encoder

### Step 3 — Graph Construction

`build_graphs(df, max_nodes=120)`:
1. Group flows by `(hour_of_day, service)` — each group becomes a potential graph
2. Chunk large groups into subgraphs of at most 120 nodes
3. For each subgraph:
   - Each row becomes a node with 14-dimensional feature vector
   - Edges are drawn between nodes sharing protocol encoding OR service encoding OR similar port values
   - Graph label = majority attack class among nodes
   - Binary label = 1 if any node is malicious, else 0
4. Return a list of PyG `Data` objects

**After fix**: Full dataset yields ~50,000+ graphs vs. ~2,500 currently.

### Step 4 — Client Partitioning

`create_clients(graphs, num_clients=7)`:
1. For each graph, compute a 2D feature centroid: `[mean(proto_enc), mean(service_enc)]`
2. Run **KMeans** (k=7) on these centroids
3. Assign graphs to clients by cluster membership
4. Result: 7 clients with non-IID data distributions (different service/protocol profiles)

### Step 5 — Federated Training Loop (10 rounds)

For each round `r` in `[1 … 10]`:

1. **Distribute** global model state to all 7 clients
2. **Local training** on each client independently:
   - Split client data: 70% train / 15% val / 15% test
   - Initialize `GraphSAGEClassifier` from global state
   - Compute **inverse-frequency class weights** from local data to combat imbalance
   - Train with **Adam** optimizer (lr=1e-3, wd=1e-4), **CrossEntropyLoss** (weighted)
   - Apply **early stopping** (patience=4) on validation accuracy
   - Restore best weights; evaluate on local test split
   - Report: `acc, prec, rec, f1, auc` per client
3. **Aggregation**: FedAvg — simple mean of all client model state_dicts
4. **Save** `results/global_model.pt`
5. **Plot** confusion matrix: all client test predictions pooled → `results/confmat_roundN.png`

### Step 6 — Post-Training Evaluation

After all 10 rounds:

1. **Load** global model on held-out global test set (10% of all graphs)
2. **Compute** global accuracy, precision, recall, F1, AUC
3. **Generate** per-class classification report → `results/classification_report.txt`
4. **Plot** learning curves (accuracy, F1, AUC across rounds) → `results/learning_curves.png`
5. **Save** label encoder → `results/attack_encoder.pkl`
6. **Save** final global metrics → `results/global_model_metrics.json`
7. **Save** all per-round per-client metrics → `results/metrics.json`
8. **Run** `test_model()` demo with synthetic + real samples

### Step 7 — Inference / Testing

**Option A — Lightweight (no CSV needed)**:
```bash
python test_model_final_cpu.py
```
Loads `results/global_model.pt` + `results/attack_encoder.pkl`, runs inference on synthetic graph inputs, outputs predicted attack type + countermeasures.

**Option B — Full test with real data**:
```bash
python test_model_final.py
```
Loads CSV, re-fits encoder (use pkl after CODE-04 fix), runs synthetic + real sample predictions.

**Option C — Batch synthetic test**:
```bash
python test_multiple.py
```
Runs 7 different synthetic attack scenarios through the model.

### Step 8 — Output Artefacts

After a successful full run, the `results/` directory will contain:

| File | Description |
|---|---|
| `global_model.pt` | Final federated model weights |
| `attack_encoder.pkl` | Saved LabelEncoder (attack class mappings) |
| `metrics.json` | Per-client, per-round: acc, prec, rec, f1, auc, predictions |
| `global_model_metrics.json` | Final global test set evaluation |
| `classification_report.txt` | Per-class precision/recall/F1/support |
| `learning_curves.png` | Accuracy, F1, AUC across all 10 rounds |
| `confmat_round1.png` … `confmat_round10.png` | Per-round confusion matrices |

---

## 5. Expected Outcome After Changes

### Performance Projections

With the full dataset (~257k rows), 7 clients, and 10 rounds:

| Metric | Current (buggy, 5k rows) | Expected (fixed, full dataset) |
|---|---|---|
| AUC | 0.0 (broken) | 0.75 – 0.92 |
| F1 Score | 80.1% (inflated by class collapse) | 70 – 85% (real multi-class) |
| Attack recall | ~0% for all attack classes | >50% for major categories |
| Confusion matrix | Single column (all "Normal") | Diagonal pattern |
| Rounds to convergence | N/A (only 3 rounds run) | ~6–8 rounds |

### Training Time Estimate (CPU-only)

With the full 257k-row dataset and 10 rounds:
- Graph construction: ~5–15 min
- Per round (7 clients × local training): ~20–40 min
- **Total estimated time: 3–7 hours on CPU**

Consider running overnight or on a machine with a GPU.

---

*This document is maintained alongside the project and should be updated after each training run.*
