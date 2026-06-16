#!/usr/bin/env python3
"""
Federated GraphSAGE Training for APT Detection (UNSW-NB15)

This script implements the full federated learning pipeline:
1. Load and preprocess the UNSW-NB15 dataset
2. Build graph representations of network flows
3. Partition graphs across clients via KMeans clustering
4. Train GraphSAGE classifiers using FedAvg for N rounds
5. Evaluate on a global held-out test set
6. Save model, metrics, confusion matrices, learning curves, and classification report

Usage:
    python train.py
    python train.py --rounds 10 --num_clients 7
"""

import os
import sys
import argparse
import json

class Logger(object):
    def __init__(self, filename="train-logs.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger()
import random
import copy
import pickle
import time

import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter
from dotenv import load_dotenv

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()  # loads DATASET_PATH from .env

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# BUG-01 FIX: Guard cuda.set_device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available. Using CPU.")

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Feature columns — adapted for the official UNSW_NB15_training-set.csv
# The original code used sport/dsport/hour_of_day which don't exist in the
# official train/test CSVs.  We substitute connection-table equivalents and
# add rate, sttl, dttl to keep 14 features.
# ---------------------------------------------------------------------------
NUMERIC_FEATURE_COLS = [
    "dur",            # flow duration
    "sbytes",         # source bytes
    "dbytes",         # destination bytes
    "spkts",          # source packets
    "dpkts",          # destination packets
    "sinpkt",         # source inter-packet time
    "rate",           # total packets/sec
    "sttl",           # source TTL
    "dttl",           # destination TTL
    "ct_srv_src",     # conn count same srv+src
    "ct_srv_dst",     # conn count same srv+dst
    "ct_dst_ltm",     # conn count same dst recently
]

# These two will be label-encoded and appended as the last 2 features
CAT_PROTO = "proto"
CAT_SERVICE = "service"

# Total features = 12 numeric + 2 encoded = 14
IN_DIM = 14

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GraphSAGEClassifier(nn.Module):
    """Two-layer GraphSAGE with global mean pooling for graph classification."""
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(self.dropout(x))

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess_df(df):
    """Clean, encode, and scale the DataFrame. Returns (df, LabelEncoder)."""
    df.columns = [c.strip().lower() for c in df.columns]  # normalise to lowercase

    # Normalise label naming: Backdoors → Backdoor
    if "attack_cat" in df.columns:
        df["attack_cat"] = (
            df["attack_cat"]
            .astype(str)
            .str.strip()
            .replace("", np.nan)
            .replace("Backdoors", "Backdoor")
            .fillna("Normal")
        )
    else:
        raise ValueError("Missing column: attack_cat")

    # Ensure binary label exists
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    else:
        # derive from attack_cat if absent
        df["label"] = (df["attack_cat"] != "Normal").astype(int)

    # Encode protocol and service
    le_proto = LabelEncoder()
    df["proto_enc"] = le_proto.fit_transform(df[CAT_PROTO].astype(str))
    le_service = LabelEncoder()
    df["service_enc"] = le_service.fit_transform(df[CAT_SERVICE].astype(str))

    # Ensure numeric columns are numeric
    for c in NUMERIC_FEATURE_COLS:
        if c not in df.columns:
            print(f"  [warn] Column '{c}' not found — filling with 0")
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # StandardScaler on numeric columns
    scaler = StandardScaler()
    df[NUMERIC_FEATURE_COLS] = scaler.fit_transform(df[NUMERIC_FEATURE_COLS])

    # Attack label encoder (alphabetical order)
    le_attack = LabelEncoder()
    df["attack_label"] = le_attack.fit_transform(df["attack_cat"])

    return df, le_attack

# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graphs(df, max_nodes=120):
    """Build PyG Data graphs by grouping flows on (service_enc, proto_enc)."""
    graphs = []

    # Group by service (we dropped hour_of_day since it's unavailable in the
    # official CSV — group by service_enc alone to still create varied subgraphs)
    for service_val, grp in df.groupby("service_enc"):
        grp = grp.reset_index(drop=True)
        for i in range(0, len(grp), max_nodes):
            sub = grp.iloc[i : i + max_nodes]
            if len(sub) < 2:
                continue  # skip trivial graphs

            G = nx.Graph()
            for idx, row in sub.iterrows():
                feats = np.concatenate([
                    row[NUMERIC_FEATURE_COLS].to_numpy(),
                    [row["proto_enc"], row["service_enc"]],
                ]).astype(np.float32)
                G.add_node(
                    idx,
                    x=feats,
                    y=row["attack_label"],
                    label=row["label"],
                    attack=row["attack_cat"],
                )

            # Edges: connect nodes that share proto_enc OR service_enc OR
            # have very similar scaled feature values on the first two numeric cols
            nodes = list(G.nodes)
            for a in range(len(nodes)):
                for b in range(a + 1, len(nodes)):
                    ra = G.nodes[nodes[a]]
                    rb = G.nodes[nodes[b]]
                    if (
                        ra["x"][-2] == rb["x"][-2]  # same proto_enc
                        or ra["x"][-1] == rb["x"][-1]  # same service_enc
                        or abs(ra["x"][0] - rb["x"][0]) < 1e-3  # similar dur
                        or abs(ra["x"][1] - rb["x"][1]) < 1e-3  # similar sbytes
                    ):
                        G.add_edge(nodes[a], nodes[b])

            # Manual PyG conversion
            x = np.array([G.nodes[n]["x"] for n in G.nodes], dtype=np.float32)
            x = torch.from_numpy(x)

            edges = np.array(list(G.edges), dtype=np.int64)
            if edges.size == 0:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                max_idx = x.size(0)
                edges = edges[(edges[:, 0] < max_idx) & (edges[:, 1] < max_idx)]
                edge_index = torch.from_numpy(edges.T)

            # Graph-level label = majority attack class
            label = Counter([G.nodes[n]["y"] for n in G.nodes]).most_common(1)[0][0]
            y = torch.tensor([label], dtype=torch.long)
            binary = torch.tensor(
                [1 if any(G.nodes[n]["label"] for n in G.nodes) else 0],
                dtype=torch.long,
            )
            batch = torch.zeros(x.size(0), dtype=torch.long)
            data_obj = Data(x=x, edge_index=edge_index, y=y, binary=binary, batch=batch)

            if edge_index.numel() > 0 and edge_index.max() >= x.size(0):
                print(f"  [warn] Invalid edge index in service={service_val}")

            graphs.append(data_obj)

    return graphs

# ---------------------------------------------------------------------------
# Client partitioning
# ---------------------------------------------------------------------------
def create_clients(graphs, num_clients):
    """Partition graphs into clients via KMeans on feature centroids."""
    feats = np.array([
        [np.mean(g.x[:, -2].numpy()), np.mean(g.x[:, -1].numpy())]
        for g in graphs
    ])
    kmeans = KMeans(n_clusters=num_clients, random_state=SEED, n_init=10).fit(feats)
    clients = {i: [] for i in range(num_clients)}
    for i, lbl in enumerate(kmeans.labels_):
        clients[lbl].append(graphs[i])
    return clients

# ---------------------------------------------------------------------------
# Evaluation  (BUG-02 FIX)
# ---------------------------------------------------------------------------
def evaluate(model, loader, num_classes):
    """Evaluate a model on a DataLoader, returning acc/prec/rec/f1/auc and preds."""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            out = model(batch)
            prob = torch.softmax(out, dim=1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend(prob.argmax(dim=1).cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # BUG-02 FIX: pass all known labels so partial test sets still work
    try:
        if len(set(y_true)) < 2:
            auc = 0.0
        else:
            auc = roc_auc_score(
                y_true,
                np.array(y_prob),
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
    except Exception as e:
        print(f"  [AUC warning] {e}")
        auc = 0.0

    return acc, prec, rec, f1, auc, (y_true, y_pred)

# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------
def train_local(graphs, global_state, num_classes, config, name):
    """Train a local model on a single client's data partition."""
    if len(graphs) < 3:
        print(f"  [{name}] Skipping — only {len(graphs)} graphs (need ≥3)")
        return global_state, None

    shuffled = random.sample(graphs, len(graphs))
    split1 = int(0.7 * len(shuffled))
    split2 = int(0.85 * len(shuffled))
    train_data, val_data, test_data = shuffled[:split1], shuffled[split1:split2], shuffled[split2:]

    in_dim = graphs[0].x.shape[1]
    model = GraphSAGEClassifier(in_dim, config["hidden_dim"], num_classes).to(DEVICE)
    if global_state:
        model.load_state_dict(global_state, strict=False)

    # Inverse-frequency class weights
    labels = [g.y.item() for g in graphs]
    class_counts = Counter(labels)
    epsilon = 1e-6
    weights = torch.tensor(
        [1.0 / (class_counts.get(i, 0) + epsilon) for i in range(num_classes)],
        dtype=torch.float32,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    best = None
    wait = 0
    patience = 4

    for ep in range(config["epochs"]):
        model.train()
        losses = []
        for b in train_loader:
            b = b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(b), b.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader, num_classes)
        print(
            f"  [{name}] Epoch {ep+1} loss={np.mean(losses):.4f} "
            f"val_acc={val_acc:.3f} f1={val_f1:.3f}"
        )
        if val_acc > (best[0] if best else 0):
            best = (val_acc, copy.deepcopy(model.state_dict()))
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f"  [{name}] Early stop at epoch {ep+1}")
            break

    if best is None:
        # fallback: use current weights
        best = (0.0, copy.deepcopy(model.state_dict()))

    model.load_state_dict(best[1])
    test_loader = DataLoader(test_data, batch_size=8)
    acc, prec, rec, f1, auc, (yt, yp) = evaluate(model, test_loader, num_classes)
    print(f"  [{name}] Test: acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")

    return copy.deepcopy(model.state_dict()), {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc,
        "yt": yt, "yp": yp,
    }

# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------
def average_states(states):
    """Average a list of model state_dicts (FedAvg)."""
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        avg[k] = sum(s[k].float() for s in states) / len(states)
    return avg

# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------
def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main(args):
    t0 = time.time()

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    dataset_path = os.environ.get("DATASET_PATH", "data").strip('"').strip("'")
    csv_path = os.path.join(dataset_path, args.csv)
    if not os.path.isfile(csv_path):
        # Try as an absolute / relative path directly
        csv_path = args.csv

    # Temporary hardcoding
    csv_path = r"C:\Users\sgogo\OneDrive\Desktop\APT Detection Fed Learning\datasets\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv"

    print(f"\n{'='*60}")
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Preprocess
    # ------------------------------------------------------------------
    print("Preprocessing...")
    df, attack_enc = preprocess_df(df)
    num_classes = len(attack_enc.classes_)
    print(f"  Classes ({num_classes}): {list(attack_enc.classes_)}")
    print(f"  Class distribution:")
    for cls_name in attack_enc.classes_:
        count = (df["attack_cat"] == cls_name).sum()
        print(f"    {cls_name:20s}: {count:>7,}")

    # ------------------------------------------------------------------
    # 3. Build graphs
    # ------------------------------------------------------------------
    print("\nBuilding graphs...")
    graphs = build_graphs(df)
    print(f"  Built {len(graphs):,} graphs")

    # ------------------------------------------------------------------
    # 4. Hold out 10% as global test set  (EVAL-02)
    # ------------------------------------------------------------------
    random.shuffle(graphs)
    split_idx = int(0.9 * len(graphs))
    train_graphs = graphs[:split_idx]
    global_test_graphs = graphs[split_idx:]
    print(f"  Train graphs: {len(train_graphs):,}")
    print(f"  Global test graphs: {len(global_test_graphs):,}")

    # ------------------------------------------------------------------
    # 5. Create federated clients
    # ------------------------------------------------------------------
    clients = create_clients(train_graphs, args.num_clients)
    print(f"\nClient partition ({args.num_clients} clients):")
    for i in range(args.num_clients):
        print(f"  Client {i+1}: {len(clients[i]):,} graphs")

    # ------------------------------------------------------------------
    # 6. Federated training loop
    # ------------------------------------------------------------------
    config = {
        "epochs": args.epochs_local,
        "lr": 1e-3,
        "wd": 1e-4,
        "hidden_dim": args.hidden_dim,
    }

    global_state = None
    allres = {}

    for r in range(args.rounds):
        print(f"\n{'='*60}")
        print(f"Round {r+1}/{args.rounds}")
        print(f"{'='*60}")

        states = []
        roundm = {}

        for i in range(args.num_clients):
            name = f"Client{i+1}"
            print(f"\n  Training {name} ({len(clients[i])} graphs)...")
            state, met = train_local(
                clients[i], global_state, num_classes, config, name
            )
            if met is not None:
                states.append(state)
                roundm[name] = met

        if not states:
            print("  [ERROR] No client produced a model this round. Skipping.")
            continue

        global_state = average_states(states)
        torch.save(global_state, "results/global_model.pt")
        allres[f"Round{r+1}"] = roundm

        # ---- Per-round confusion matrix ----
        all_y, all_p = [], []
        for c in roundm.values():
            all_y.extend(c["yt"])
            all_p.extend(c["yp"])

        cm = confusion_matrix(all_y, all_p, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(cm, display_labels=attack_enc.classes_)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, xticks_rotation=90, cmap="Blues")
        ax.set_title(f"Confusion Matrix — Round {r+1}")
        plt.tight_layout()
        plt.savefig(f"results/confmat_round{r+1}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved results/confmat_round{r+1}.png")

    # ------------------------------------------------------------------
    # 7. Save per-round metrics
    # ------------------------------------------------------------------
    with open("results/metrics.json", "w") as f:
        json.dump(clean_json(allres), f, indent=2)
    print("\nSaved results/metrics.json")

    # ------------------------------------------------------------------
    # 8. Save label encoder  (CODE-04 FIX)
    # ------------------------------------------------------------------
    with open("results/attack_encoder.pkl", "wb") as f:
        pickle.dump(attack_enc, f)
    print("Saved results/attack_encoder.pkl")

    # ------------------------------------------------------------------
    # 9. Global test set evaluation  (EVAL-02)
    # ------------------------------------------------------------------
    if global_state and len(global_test_graphs) > 0:
        print(f"\n{'='*60}")
        print("Global Test Set Evaluation")
        print(f"{'='*60}")

        global_net = GraphSAGEClassifier(
            IN_DIM, config["hidden_dim"], num_classes
        ).to(DEVICE)
        global_net.load_state_dict(global_state, strict=False)
        global_net.eval()

        global_test_loader = DataLoader(global_test_graphs, batch_size=8)
        g_acc, g_prec, g_rec, g_f1, g_auc, (g_yt, g_yp) = evaluate(
            global_net, global_test_loader, num_classes
        )

        global_metrics = {
            "accuracy": g_acc,
            "precision": g_prec,
            "recall": g_rec,
            "f1_score": g_f1,
            "auc": g_auc,
            "num_samples": len(g_yt),
        }
        with open("results/global_model_metrics.json", "w") as f:
            json.dump(clean_json(global_metrics), f, indent=2)
        print(f"  Accuracy:  {g_acc:.4f}")
        print(f"  Precision: {g_prec:.4f}")
        print(f"  Recall:    {g_rec:.4f}")
        print(f"  F1 Score:  {g_f1:.4f}")
        print(f"  AUC:       {g_auc:.4f}")
        print("Saved results/global_model_metrics.json")

        # ---- Per-class classification report  (EVAL-03) ----
        report = classification_report(
            g_yt, g_yp,
            labels=list(range(num_classes)),
            target_names=attack_enc.classes_,
            zero_division=0,
        )
        print(f"\n{report}")
        with open("results/classification_report.txt", "w") as f:
            f.write(report)
        print("Saved results/classification_report.txt")

        # ---- Global confusion matrix ----
        cm = confusion_matrix(g_yt, g_yp, labels=list(range(num_classes)))
        disp = ConfusionMatrixDisplay(cm, display_labels=attack_enc.classes_)
        fig, ax = plt.subplots(figsize=(10, 8))
        disp.plot(ax=ax, xticks_rotation=90, cmap="Greens")
        ax.set_title("Confusion Matrix — Global Test Set")
        plt.tight_layout()
        plt.savefig("results/confmat_global.png", dpi=150)
        plt.close(fig)
        print("Saved results/confmat_global.png")

    # ------------------------------------------------------------------
    # 10. Learning curves  (EVAL-01)
    # ------------------------------------------------------------------
    if allres:
        rounds_list = sorted(allres.keys())
        avg_accs, avg_f1s, avg_aucs = [], [], []

        for rkey in rounds_list:
            rm = allres[rkey]
            clients_met = list(rm.values())
            avg_accs.append(np.mean([c["acc"] for c in clients_met]))
            avg_f1s.append(np.mean([c["f1"] for c in clients_met]))
            avg_aucs.append(np.mean([c["auc"] for c in clients_met]))

        round_nums = list(range(1, len(rounds_list) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, values, label, color in zip(
            axes,
            [avg_accs, avg_f1s, avg_aucs],
            ["Accuracy", "F1 Score", "AUC"],
            ["steelblue", "darkorange", "green"],
        ):
            ax.plot(round_nums, values, marker="o", color=color, linewidth=2)
            ax.set_title(f"Average {label} per Round", fontsize=12)
            ax.set_xlabel("Federated Round")
            ax.set_ylabel(label)
            ax.set_xticks(round_nums)
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Federated Learning Convergence", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results/learning_curves.png", dpi=150)
        plt.close(fig)
        print("Saved results/learning_curves.png")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    mins, secs = divmod(int(elapsed), 60)
    hrs, mins = divmod(mins, 60)
    print(f"\nTraining complete in {hrs}h {mins}m {secs}s.")
    print("Model saved to results/global_model.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated GraphSAGE APT Detection")
    parser.add_argument(
        "--csv",
        default="UNSW_NB15_training-set.csv",
        help="CSV filename inside DATASET_PATH (default: UNSW_NB15_training-set.csv)",
    )
    parser.add_argument("--num_clients", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--epochs_local", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    args = parser.parse_args()
    main(args)
