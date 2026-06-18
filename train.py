#!/usr/bin/env python3
"""
Federated GraphSAGE Training for APT Detection (UNSW-NB15) — robust pipeline.

Implements the fixes catalogued in IMPROVEMENTS.md, PART II ("Implemented Fixes,
June 2026"):

  F1  Train-only preprocessing (no leakage)
  F2  Honest evaluation on the official held-out testing-set.csv
  F3  Non-leaking feature-similarity graphs (no CSV-order / temporal artifact)
  F4  Minority oversampling (no imbalanced-learn dependency)
  F5  Focal loss + safe (neutral) class weights
  F6  Stratified IID client partitioning
  F7  Sample-weighted FedAvg + FedProx proximal term
  F8  Robust metrics (macro-F1 / per-class recall headline, safe per-class AUC)
  F9  Centralized GraphSAGE + per-flow MLP baselines for honest comparison
  F10 Reproducibility/hygiene (no hardcoded paths, timestamped logs, config-driven dims)

Usage:
    python train.py
    python train.py --rounds 10 --num_clients 7
    python train.py --skip_baselines --rounds 3        # quick smoke test
"""

import os
import sys
import argparse
import json
import time
import random
import copy
import pickle

# ---------------------------------------------------------------------------
# F10: timestamped, non-truncating log file (one file per run)
# ---------------------------------------------------------------------------
class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "w", encoding="utf-8")

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            enc = getattr(self.terminal, "encoding", None) or "ascii"
            self.terminal.write(message.encode(enc, "replace").decode(enc))
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


import numpy as np
import pandas as pd
from collections import Counter
from dotenv import load_dotenv

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------------------
# Configuration / reproducibility
# ---------------------------------------------------------------------------
load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)

# BUG-01 (kept): guard cuda.set_device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.set_device(0)

os.makedirs("results", exist_ok=True)

# 39 numeric features available in the official UNSW_NB15 train/test sets
NUMERIC_FEATURE_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports",
]
CONT_DIM = len(NUMERIC_FEATURE_COLS)          # 39
CAT_COLS = ["proto", "service", "state"]
UNKNOWN = "<unknown>"


# ===========================================================================
# Models
# ===========================================================================
class GraphSAGEClassifier(nn.Module):
    """Two-layer GraphSAGE with embeddings for proto/service/state + continuous features.

    Node feature layout in `data.x`: [continuous_dim | proto_idx | service_idx | state_idx]
    """
    def __init__(self, num_protos, num_services, num_states, continuous_dim,
                 hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.continuous_dim = continuous_dim          # F10: no magic 39 in forward()
        self.proto_emb = nn.Embedding(num_protos, 16)
        self.service_emb = nn.Embedding(num_services, 8)
        self.state_emb = nn.Embedding(num_states, 8)

        in_dim = continuous_dim + 16 + 8 + 8
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def _embed(self, x):
        d = self.continuous_dim
        cont = x[:, :d]
        p = self.proto_emb(x[:, d].long())
        s = self.service_emb(x[:, d + 1].long())
        st = self.state_emb(x[:, d + 2].long())
        return torch.cat([cont, p, s, st], dim=1)

    def forward(self, data):
        x = self._embed(data.x)
        x = self.relu(self.conv1(x, data.edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, data.edge_index))
        return self.fc(self.dropout(x))


class MLPClassifier(nn.Module):
    """Per-flow baseline (F9). Same embeddings as the GNN but NO graph / message passing."""
    def __init__(self, num_protos, num_services, num_states, continuous_dim,
                 hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.continuous_dim = continuous_dim
        self.proto_emb = nn.Embedding(num_protos, 16)
        self.service_emb = nn.Embedding(num_services, 8)
        self.state_emb = nn.Embedding(num_states, 8)
        in_dim = continuous_dim + 16 + 8 + 8
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        d = self.continuous_dim
        cont = x[:, :d]
        p = self.proto_emb(x[:, d].long())
        s = self.service_emb(x[:, d + 1].long())
        st = self.state_emb(x[:, d + 2].long())
        return self.net(torch.cat([cont, p, s, st], dim=1))


class FocalLoss(nn.Module):
    """Focal loss with per-class alpha weights (F5)."""
    def __init__(self, alpha, gamma=1.5):
        super().__init__()
        self.alpha = alpha          # tensor [C]
        self.gamma = gamma

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.alpha, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return (((1 - pt) ** self.gamma) * ce).mean()


# ===========================================================================
# F1: preprocessing fit on TRAIN only, applied to test
# ===========================================================================
def _normalise_attack(series):
    return (
        series.astype(str).str.strip()
        .replace("", np.nan)
        .replace("Backdoors", "Backdoor")
        .fillna("Normal")
    )


def _fit_cat(series):
    le = LabelEncoder()
    le.fit(np.append(series.astype(str).unique(), UNKNOWN))
    return le, set(le.classes_), int(le.transform([UNKNOWN])[0])


def _apply_cat(series, le, known, unk_idx):
    s = series.astype(str)
    s = s.where(s.isin(known), UNKNOWN)
    return le.transform(s)


def fit_preprocessor(df_train):
    """Fit scaler + categorical/attack encoders on the TRAIN dataframe only."""
    df = df_train.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "attack_cat" not in df.columns:
        raise ValueError("Missing column: attack_cat")
    df["attack_cat"] = _normalise_attack(df["attack_cat"])

    cat_enc = {}
    for c in CAT_COLS:
        le, known, unk = _fit_cat(df[c])
        cat_enc[c] = {"le": le, "known": known, "unk": unk}

    for c in NUMERIC_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    scaler = StandardScaler().fit(df[NUMERIC_FEATURE_COLS])

    attack_enc = LabelEncoder().fit(df["attack_cat"])

    pre = {"cat_enc": cat_enc, "scaler": scaler, "attack_enc": attack_enc}
    return pre


def transform_df(df_in, pre):
    """Apply a fitted preprocessor. Returns a processed dataframe (+ unseen attack rows dropped)."""
    df = df_in.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df["attack_cat"] = _normalise_attack(df["attack_cat"])

    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    else:
        df["label"] = (df["attack_cat"] != "Normal").astype(int)

    for c in CAT_COLS:
        e = pre["cat_enc"][c]
        df[c + "_enc"] = _apply_cat(df[c], e["le"], e["known"], e["unk"])

    for c in NUMERIC_FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df[NUMERIC_FEATURE_COLS] = pre["scaler"].transform(df[NUMERIC_FEATURE_COLS])

    # Drop rows whose attack_cat the train encoder never saw (keeps label space consistent)
    known_attacks = set(pre["attack_enc"].classes_)
    df = df[df["attack_cat"].isin(known_attacks)].reset_index(drop=True)
    df["attack_label"] = pre["attack_enc"].transform(df["attack_cat"])
    return df


# ===========================================================================
# F4: minority oversampling (train only)
# ===========================================================================
def oversample_minorities(df, floor, jitter=0.05):
    """Raise every class up to `floor` rows by resampling with Gaussian jitter on scaled
    continuous features. Categorical indices are left unchanged. Test set is NOT oversampled."""
    if floor <= 0:
        return df
    parts = [df]
    counts = df["attack_label"].value_counts().to_dict()
    for cls, cnt in counts.items():
        if cnt >= floor:
            continue
        need = floor - cnt
        pool = df[df["attack_label"] == cls]
        idx = RNG.integers(0, len(pool), size=need)
        extra = pool.iloc[idx].copy().reset_index(drop=True)
        noise = RNG.normal(0.0, jitter, size=(need, CONT_DIM)).astype(np.float32)
        extra[NUMERIC_FEATURE_COLS] = extra[NUMERIC_FEATURE_COLS].to_numpy() + noise
        parts.append(extra)
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


# ===========================================================================
# F3: non-leaking feature-similarity graphs
# ===========================================================================
def build_graphs(df, window=256, knn=5):
    """Shuffle rows (kills CSV-order artifact), window them, connect via kNN on scaled
    continuous features only. No temporal-chain edges."""
    cont = df[NUMERIC_FEATURE_COLS].to_numpy(np.float32)
    cats = df[[c + "_enc" for c in CAT_COLS]].to_numpy(np.float32)
    X = np.concatenate([cont, cats], axis=1)               # [N, CONT_DIM+3]
    y = df["attack_label"].to_numpy()

    order = RNG.permutation(len(df))
    graphs = []
    for s in range(0, len(order), window):
        wi = order[s : s + window]
        if len(wi) < 2:
            continue
        xs = X[wi]
        ys = y[wi]
        c = xs[:, :CONT_DIM]
        n = len(wi)
        k = min(knn, n - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(c).kneighbors(c, return_distance=False)
        src, dst = [], []
        for a in range(n):
            for b in nbrs[a][1:]:                          # skip self
                src += [a, int(b)]
                dst += [int(b), a]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        graphs.append(Data(
            x=torch.from_numpy(xs),
            edge_index=edge_index,
            y=torch.tensor(ys, dtype=torch.long),
        ))
    return graphs


# ===========================================================================
# F6: stratified IID client partitioning
# ===========================================================================
def create_clients(graphs, num_clients):
    """Round-robin shuffled multi-class graphs so every client sees every class."""
    g = list(graphs)
    random.shuffle(g)
    clients = {i: [] for i in range(num_clients)}
    for i, graph in enumerate(g):
        clients[i % num_clients].append(graph)
    return clients


# ===========================================================================
# F5: safe class weights
# ===========================================================================
def class_alpha(labels, num_classes, power=0.5):
    """sqrt-smoothed inverse-frequency weights; absent classes get the mean (neutral, not 0)."""
    counts = Counter(labels)
    total = max(len(labels), 1)
    w = torch.zeros(num_classes, dtype=torch.float32)
    present = []
    for i in range(num_classes):
        if counts.get(i, 0) > 0:
            w[i] = (total / (num_classes * counts[i])) ** power
            present.append(w[i].item())
    mean_w = float(np.mean(present)) if present else 1.0
    for i in range(num_classes):
        if counts.get(i, 0) == 0:
            w[i] = mean_w                                   # neutral, avoids unpenalised FP
    return w


# ===========================================================================
# F8: robust metrics
# ===========================================================================
def safe_macro_auc(y_true, y_prob, num_classes):
    """Macro OvR AUC over classes actually present (>=2). Returns None if undefined."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    present = [c for c in range(num_classes) if (y_true == c).sum() > 0]
    if len(present) < 2:
        return None
    Y = label_binarize(y_true, classes=list(range(num_classes)))
    aucs = []
    for c in present:
        col = Y[:, c]
        if 0 < col.sum() < len(col):
            try:
                aucs.append(roc_auc_score(col, y_prob[:, c]))
            except Exception:
                pass
    return float(np.mean(aucs)) if aucs else None


def evaluate(model, loader, num_classes, is_graph=True):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                batch = batch.to(DEVICE)
                out, target = model(batch), batch.y
            else:
                xb, target = batch[0].to(DEVICE), batch[1].to(DEVICE)
                out = model(xb)
            prob = torch.softmax(out, dim=1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(prob.argmax(1).cpu().numpy())
            y_prob.extend(prob.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, labels=list(range(num_classes)),
                        average="macro", zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=list(range(num_classes)),
                                    average=None, zero_division=0).tolist()
    auc = safe_macro_auc(y_true, y_prob, num_classes)
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "macro_f1": macro_f1, "auc": auc,
        "per_class_recall": per_class_recall,
        "yt": y_true, "yp": y_pred,
    }


def val_macro_f1(model, loader, num_classes, criterion, is_graph=True):
    """Cheap validation pass: returns (mean_loss, macro_f1). No AUC, no spam."""
    model.eval()
    losses, yt, yp = [], [], []
    with torch.no_grad():
        for batch in loader:
            if is_graph:
                batch = batch.to(DEVICE)
                out, target = model(batch), batch.y
            else:
                xb, target = batch[0].to(DEVICE), batch[1].to(DEVICE)
                out = model(xb)
            losses.append(criterion(out, target).item())
            yt.extend(target.cpu().numpy())
            yp.extend(out.argmax(1).cpu().numpy())
    mf1 = f1_score(yt, yp, labels=list(range(num_classes)), average="macro", zero_division=0)
    return float(np.mean(losses)), mf1


# ===========================================================================
# F7: local training with FedProx
# ===========================================================================
def train_local(graphs, global_state, num_classes, config, name):
    if len(graphs) < 3:
        print(f"  [{name}] Skipping — only {len(graphs)} graphs (need >=3)")
        return None, 0, None

    shuffled = random.sample(graphs, len(graphs))
    s1 = int(0.7 * len(shuffled))
    s2 = int(0.85 * len(shuffled))
    train_data, val_data, test_data = shuffled[:s1], shuffled[s1:s2], shuffled[s2:]
    if not train_data or not val_data or not test_data:
        train_data, val_data, test_data = shuffled, shuffled, shuffled

    model = GraphSAGEClassifier(
        config["num_protos"], config["num_services"], config["num_states"],
        CONT_DIM, config["hidden_dim"], num_classes,
    ).to(DEVICE)
    if global_state:
        model.load_state_dict(global_state, strict=False)

    # FedProx anchor
    global_params = None
    if global_state and config["fedprox_mu"] > 0:
        global_params = [p.detach().clone() for p in model.parameters()]

    labels = [int(v) for g in graphs for v in g.y.tolist()]
    alpha = class_alpha(labels, num_classes).to(DEVICE)
    criterion = FocalLoss(alpha, gamma=config["focal_gamma"])
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["wd"])

    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8)

    best = None
    wait, patience = 0, config["patience"]
    n_train_nodes = sum(int(g.y.numel()) for g in train_data)

    for ep in range(config["epochs"]):
        model.train()
        losses = []
        for b in train_loader:
            b = b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(b), b.y)
            if global_params is not None:
                prox = sum(((p - gp) ** 2).sum() for p, gp in zip(model.parameters(), global_params))
                loss = loss + 0.5 * config["fedprox_mu"] * prox
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        vloss, vmf1 = val_macro_f1(model, val_loader, num_classes, criterion)
        print(f"  [{name}] Epoch {ep+1} loss={np.mean(losses):.4f} "
              f"val_loss={vloss:.4f} val_macroF1={vmf1:.3f}")
        if best is None or vloss < best[0]:
            best = (vloss, copy.deepcopy(model.state_dict()))
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  [{name}] Early stop at epoch {ep+1}")
                break

    model.load_state_dict(best[1])
    test_loader = DataLoader(test_data, batch_size=8)
    m = evaluate(model, test_loader, num_classes)
    auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
    print(f"  [{name}] Test: acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} auc={auc_str}")
    return copy.deepcopy(model.state_dict()), n_train_nodes, m


# ===========================================================================
# F7: sample-weighted FedAvg
# ===========================================================================
def fedavg(states, weights):
    total = float(sum(weights)) or 1.0
    avg = copy.deepcopy(states[0])
    for k in avg.keys():
        avg[k] = sum(s[k].float() * w for s, w in zip(states, weights)) / total
    return avg


# ===========================================================================
# F9: baselines (trained centrally, evaluated on official test set)
# ===========================================================================
def _tensor_flow_dataset(df, num_classes):
    cont = df[NUMERIC_FEATURE_COLS].to_numpy(np.float32)
    cats = df[[c + "_enc" for c in CAT_COLS]].to_numpy(np.float32)
    X = torch.from_numpy(np.concatenate([cont, cats], axis=1))
    y = torch.tensor(df["attack_label"].to_numpy(), dtype=torch.long)
    return torch.utils.data.TensorDataset(X, y)


def train_mlp_baseline(train_df, test_df, config, num_classes, epochs=15):
    print("\n  [baseline] Per-flow MLP (no graph)...")
    tr = torch.utils.data.DataLoader(_tensor_flow_dataset(train_df, num_classes),
                                     batch_size=512, shuffle=True)
    te = torch.utils.data.DataLoader(_tensor_flow_dataset(test_df, num_classes),
                                     batch_size=1024)
    model = MLPClassifier(config["num_protos"], config["num_services"],
                          config["num_states"], CONT_DIM, config["hidden_dim"],
                          num_classes).to(DEVICE)
    alpha = class_alpha(train_df["attack_label"].tolist(), num_classes).to(DEVICE)
    crit = FocalLoss(alpha, gamma=config["focal_gamma"])
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    m = evaluate(model, te, num_classes, is_graph=False)
    auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
    print(f"  [baseline] MLP test: acc={m['acc']:.3f} macroF1={m['macro_f1']:.3f} auc={auc_str}")
    return m


def train_centralized_gnn(train_graphs, test_graphs, config, num_classes, epochs=15):
    print("\n  [baseline] Centralized GraphSAGE (no FL)...")
    model = GraphSAGEClassifier(config["num_protos"], config["num_services"],
                                config["num_states"], CONT_DIM, config["hidden_dim"],
                                num_classes).to(DEVICE)
    labels = [int(v) for g in train_graphs for v in g.y.tolist()]
    alpha = class_alpha(labels, num_classes).to(DEVICE)
    crit = FocalLoss(alpha, gamma=config["focal_gamma"])
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train_graphs, batch_size=8, shuffle=True)
    te = DataLoader(test_graphs, batch_size=8)
    for ep in range(epochs):
        model.train()
        for b in loader:
            b = b.to(DEVICE)
            opt.zero_grad()
            crit(model(b), b.y).backward()
            opt.step()
    m = evaluate(model, te, num_classes)
    auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
    print(f"  [baseline] Centralized GNN test: acc={m['acc']:.3f} "
          f"macroF1={m['macro_f1']:.3f} auc={auc_str}")
    return m


# ===========================================================================
# Helpers
# ===========================================================================
def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_json(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def resolve_paths(args):
    """F10: locate train/test CSVs from --train_csv/--test_csv + DATASET_PATH (file or dir),
    resolved against several anchors so a relative DATASET_PATH works regardless of cwd."""
    raw = os.environ.get("DATASET_PATH", "").strip().strip('"').strip("'").strip()
    here = os.path.dirname(os.path.abspath(__file__))
    anchors = [os.getcwd(), os.path.dirname(os.getcwd()), here, os.path.dirname(here)]

    candidates = []
    if raw:
        candidates.append(raw)                                   # raw may BE the train csv
        rawdir = raw if os.path.isdir(raw) else os.path.dirname(raw)
        for a in anchors:
            candidates.append(os.path.join(a, raw))              # raw (a file) relative to anchor
            candidates.append(os.path.join(a, rawdir, args.train_csv))
    candidates.append(args.train_csv)
    for a in anchors:
        candidates.append(os.path.join(a, args.train_csv))
        candidates.append(os.path.join(a, "data", args.train_csv))

    for tr in candidates:
        if tr and os.path.isfile(tr):
            base = os.path.dirname(os.path.abspath(tr))
            te = os.path.join(base, args.test_csv)
            return os.path.abspath(tr), (te if os.path.isfile(te) else None)

    # Fallback: recursively glob for the file under each anchor (robust to a broken .env)
    import glob
    for a in anchors:
        hits = glob.glob(os.path.join(a, "**", args.train_csv), recursive=True)
        if hits:
            base = os.path.dirname(os.path.abspath(hits[0]))
            te = os.path.join(base, args.test_csv)
            return os.path.abspath(hits[0]), (te if os.path.isfile(te) else None)

    raise FileNotFoundError(
        f"Could not locate {args.train_csv}. Set DATASET_PATH in .env or pass --train_csv."
    )


def save_confmat(yt, yp, classes, num_classes, title, path, cmap="Blues"):
    cm = confusion_matrix(yt, yp, labels=list(range(num_classes)))
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=90, cmap=cmap)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================
def main(args):
    t0 = time.time()
    train_csv, test_csv = resolve_paths(args)

    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Train CSV: {train_csv}")
    print(f"Test  CSV: {test_csv if test_csv else '(none — will split train)'}")

    # ---- Load ----
    df_train_raw = pd.read_csv(train_csv, low_memory=False)
    print(f"  Loaded train: {len(df_train_raw):,} rows, {len(df_train_raw.columns)} cols")

    # F1: fit preprocessing on TRAIN only
    pre = fit_preprocessor(df_train_raw)
    attack_enc = pre["attack_enc"]
    num_classes = len(attack_enc.classes_)
    num_protos = len(pre["cat_enc"]["proto"]["le"].classes_)
    num_services = len(pre["cat_enc"]["service"]["le"].classes_)
    num_states = len(pre["cat_enc"]["state"]["le"].classes_)
    print(f"  Classes ({num_classes}): {list(attack_enc.classes_)}")

    df_train = transform_df(df_train_raw, pre)

    # F2: official held-out test set (fall back to a 10% split only if absent)
    if test_csv:
        df_test = transform_df(pd.read_csv(test_csv, low_memory=False), pre)
    else:
        cut = int(0.9 * len(df_train))
        df_train, df_test = df_train.iloc[:cut].reset_index(drop=True), df_train.iloc[cut:].reset_index(drop=True)
    print(f"  Test rows: {len(df_test):,}")

    print("  Train class distribution (pre-oversample):")
    for c in attack_enc.classes_:
        print(f"    {c:18s}: {(df_train['attack_cat'] == c).sum():>7,}")

    # F4: oversample minorities in TRAIN only
    df_train_os = oversample_minorities(df_train, args.oversample_floor)
    if args.oversample_floor > 0:
        print(f"  After oversampling (floor={args.oversample_floor}): {len(df_train_os):,} train rows")

    # F3: build graphs
    print("\nBuilding graphs...")
    train_graphs = build_graphs(df_train_os, args.window, args.knn)
    test_graphs = build_graphs(df_test, args.window, args.knn)
    print(f"  Train graphs: {len(train_graphs):,}  Test graphs: {len(test_graphs):,}")

    config = {
        "epochs": args.epochs_local, "lr": 1e-3, "wd": 1e-4,
        "hidden_dim": args.hidden_dim, "num_protos": num_protos,
        "num_services": num_services, "num_states": num_states,
        "fedprox_mu": args.fedprox_mu, "focal_gamma": args.focal_gamma,
        "patience": args.patience,
    }
    test_loader = DataLoader(test_graphs, batch_size=8)

    # ---- F9: baselines ----
    comparison = {}
    if not args.skip_baselines:
        print("\n" + "=" * 60 + "\nBaselines (centralized, official test set)\n" + "=" * 60)
        mlp_m = train_mlp_baseline(df_train_os, df_test, config, num_classes)
        cen_m = train_centralized_gnn(train_graphs, test_graphs, config, num_classes)
        comparison["mlp_flow"] = {k: mlp_m[k] for k in ("acc", "macro_f1", "f1", "auc", "per_class_recall")}
        comparison["centralized_gnn"] = {k: cen_m[k] for k in ("acc", "macro_f1", "f1", "auc", "per_class_recall")}

    # ---- F6: clients ----
    clients = create_clients(train_graphs, args.num_clients)
    print(f"\nClient partition ({args.num_clients} clients):")
    for i in range(args.num_clients):
        print(f"  Client {i+1}: {len(clients[i]):,} graphs")

    # ---- Federated loop ----
    global_state = None
    allres = {}
    for r in range(args.rounds):
        print("\n" + "=" * 60 + f"\nRound {r+1}/{args.rounds}\n" + "=" * 60)
        states, sample_weights, roundm = [], [], {}
        for i in range(args.num_clients):
            name = f"Client{i+1}"
            print(f"\n  Training {name} ({len(clients[i])} graphs)...")
            state, n_nodes, met = train_local(clients[i], global_state, num_classes, config, name)
            if met is not None:
                states.append(state)
                sample_weights.append(n_nodes)
                roundm[name] = {k: met[k] for k in ("acc", "prec", "rec", "f1", "macro_f1", "auc")}
        if not states:
            print("  [ERROR] No client produced a model this round.")
            continue
        global_state = fedavg(states, sample_weights)        # F7: weighted
        torch.save(global_state, "results/global_model.pt")
        allres[f"Round{r+1}"] = roundm

    # ---- Final federated evaluation on official test set ----
    print("\n" + "=" * 60 + "\nFederated Global Model — Official Test Set\n" + "=" * 60)
    global_net = GraphSAGEClassifier(num_protos, num_services, num_states, CONT_DIM,
                                     args.hidden_dim, num_classes).to(DEVICE)
    global_net.load_state_dict(global_state, strict=False)
    fed_m = evaluate(global_net, test_loader, num_classes)
    auc_str = f"{fed_m['auc']:.4f}" if fed_m["auc"] is not None else "n/a"
    print(f"  Accuracy:        {fed_m['acc']:.4f}")
    print(f"  Weighted F1:     {fed_m['f1']:.4f}")
    print(f"  Macro F1:        {fed_m['macro_f1']:.4f}   <-- headline")
    print(f"  Macro AUC:       {auc_str}")

    report = classification_report(
        fed_m["yt"], fed_m["yp"], labels=list(range(num_classes)),
        target_names=attack_enc.classes_, zero_division=0,
    )
    print("\n" + report)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)
    save_confmat(fed_m["yt"], fed_m["yp"], attack_enc.classes_, num_classes,
                 "Confusion Matrix — Federated, Official Test Set",
                 "results/confmat_global.png", cmap="Greens")

    comparison["federated_gnn"] = {k: fed_m[k] for k in ("acc", "macro_f1", "f1", "auc", "per_class_recall")}

    # ---- Save artifacts ----
    with open("results/metrics.json", "w") as f:
        json.dump(clean_json(allres), f, indent=2)
    with open("results/global_model_metrics.json", "w") as f:
        json.dump(clean_json({
            "accuracy": fed_m["acc"], "weighted_f1": fed_m["f1"],
            "macro_f1": fed_m["macro_f1"], "auc": fed_m["auc"],
            "per_class_recall": dict(zip(list(attack_enc.classes_), fed_m["per_class_recall"])),
            "num_samples": len(fed_m["yt"]),
        }), f, indent=2)
    with open("results/baseline_comparison.json", "w") as f:
        json.dump(clean_json({
            "classes": list(attack_enc.classes_),
            "metric_note": "per_class_recall is aligned to 'classes'; headline = macro_f1",
            "models": comparison,
        }), f, indent=2)
    for nm, enc in (("attack", attack_enc), ("proto", pre["cat_enc"]["proto"]["le"]),
                    ("service", pre["cat_enc"]["service"]["le"]), ("state", pre["cat_enc"]["state"]["le"])):
        with open(f"results/{nm}_encoder.pkl", "wb") as f:
            pickle.dump(enc, f)
    with open("results/scaler.pkl", "wb") as f:
        pickle.dump(pre["scaler"], f)
    with open("results/model_config.json", "w") as f:
        json.dump({
            "num_protos": num_protos, "num_services": num_services,
            "num_states": num_states, "continuous_dim": CONT_DIM,
            "hidden_dim": args.hidden_dim, "num_classes": num_classes,
            "numeric_feature_cols": NUMERIC_FEATURE_COLS, "cat_cols": CAT_COLS,
        }, f, indent=2)
    print("\nSaved encoders, scaler, model_config.json, metrics.json, "
          "global_model_metrics.json, baseline_comparison.json")

    # ---- Learning curves (only valid AUCs averaged) ----
    if allres:
        rkeys = sorted(allres.keys(), key=lambda s: int(s.replace("Round", "")))
        accs, mf1s, aucs = [], [], []
        for rk in rkeys:
            cm = list(allres[rk].values())
            accs.append(np.mean([c["acc"] for c in cm]))
            mf1s.append(np.mean([c["macro_f1"] for c in cm]))
            valid = [c["auc"] for c in cm if c["auc"] is not None]
            aucs.append(np.mean(valid) if valid else np.nan)
        xs = list(range(1, len(rkeys) + 1))
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, vals, lab, col in zip(axes, [accs, mf1s, aucs],
                                      ["Accuracy", "Macro F1", "Macro AUC"],
                                      ["steelblue", "darkorange", "green"]):
            ax.plot(xs, vals, marker="o", color=col, linewidth=2)
            ax.set_title(f"Avg client {lab} per round")
            ax.set_xlabel("Round"); ax.set_ylabel(lab)
            ax.set_xticks(xs); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)
        plt.suptitle("Federated Learning Convergence", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("results/learning_curves.png", dpi=150)
        plt.close(fig)
        print("Saved results/learning_curves.png")

    # ---- Comparison summary ----
    print("\n" + "=" * 60 + "\nMODEL COMPARISON (official test set)\n" + "=" * 60)
    print(f"  {'model':18s} {'acc':>7s} {'macroF1':>8s} {'wF1':>7s} {'AUC':>7s}")
    for nm, m in comparison.items():
        a = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
        print(f"  {nm:18s} {m['acc']:7.3f} {m['macro_f1']:8.3f} {m['f1']:7.3f} {a:>7s}")

    h, rem = divmod(int(time.time() - t0), 3600)
    mnt, sec = divmod(rem, 60)
    print(f"\nDone in {h}h {mnt}m {sec}s. Model -> results/global_model.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Federated GraphSAGE APT Detection (robust)")
    p.add_argument("--train_csv", default="UNSW_NB15_training-set.csv")
    p.add_argument("--test_csv", default="UNSW_NB15_testing-set.csv")
    p.add_argument("--num_clients", type=int, default=7)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--epochs_local", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--window", type=int, default=256)
    p.add_argument("--knn", type=int, default=5)
    p.add_argument("--fedprox_mu", type=float, default=0.01)
    p.add_argument("--focal_gamma", type=float, default=1.5)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--oversample_floor", type=int, default=1500,
                   help="min train rows per class after oversampling (0 disables)")
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--log", default=None, help="log file path (default: timestamped)")
    args = p.parse_args()

    log_path = args.log or time.strftime("train-logs.txt")
    sys.stdout = Logger(log_path)
    main(args)
