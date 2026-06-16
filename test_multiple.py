#!/usr/bin/env python3
"""
Batch test: Run multiple synthetic attack scenarios + one real sample prediction.

Uses the saved model (full state_dict) and saved encoder.
"""

import os
import pickle

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.nn import SAGEConv, global_mean_pool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# --- Feature config (must match train.py) ---
NUMERIC_FEATURE_COLS = [
    "dur", "sbytes", "dbytes", "spkts", "dpkts", "sinpkt",
    "rate", "sttl", "dttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
]
IN_DIM = 14

# --- Model Definition ---
class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.fc(self.dropout(x))

# --- Device ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Load saved encoder (CODE-04 fix) ---
encoder_path = "results/attack_encoder.pkl"
if os.path.exists(encoder_path):
    print(f"Loading encoder from {encoder_path}")
    with open(encoder_path, "rb") as f:
        attack_enc = pickle.load(f)
else:
    # Fallback: alphabetical order
    print("[warn] Encoder file not found — using alphabetical class order")
    attack_enc = LabelEncoder()
    attack_enc.classes_ = np.array([
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
    ])

num_classes = len(attack_enc.classes_)
print(f"Loaded encoder with {num_classes} attack types.")

# --- Load Model (BUG-03 FIX: load full state_dict) ---
model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please train first at results/global_model.pt")

print("Loading trained model weights...")
net = GraphSAGEClassifier(IN_DIM, 64, num_classes).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
# BUG-03 FIX: Load the COMPLETE model — do NOT filter out fc. layer
net.load_state_dict(state_dict, strict=False)
print("Model loaded successfully (full state_dict).")
net.eval()

# --- Countermeasures ---
mitigation = {
    "Exploits": ["Patch vulnerable software", "Enable IPS", "Run vulnerability scans", "Harden systems"],
    "Reconnaissance": ["Deploy IDS", "Block ICMP sweeps", "Rate-limit unknown IPs"],
    "Fuzzers": ["Input validation", "Segment networks", "Limit error verbosity"],
    "DoS": ["Rate limiting", "Use CDN/WAF", "Monitor anomalies"],
    "Generic": ["Update antivirus", "Use sandboxing", "Inspect traffic anomalies"],
    "Analysis": ["Secure logging", "Restrict admin rights", "Monitor host activity"],
    "Backdoor": ["Re-image systems", "Rotate credentials", "Use MFA"],
    "Shellcode": ["Patch memory vulnerabilities", "Enable DEP/ASLR", "Deploy EDR"],
    "Worms": ["Isolate networks", "Patch systems", "Disable SMB/RPC"],
    "Normal": ["No attack detected", "Maintain routine monitoring"],
}

# --- Synthetic case generator ---
rng = np.random.RandomState(42)

def make_synthetic_case(kind="generic"):
    """Generate a synthetic 14-dim feature vector simulating different attack types."""
    base = rng.normal(loc=0.0, scale=0.5, size=12).astype(np.float32)
    proto_enc = float(rng.randint(0, 10))
    service_enc = float(rng.randint(0, 10))

    if kind == "exploit":
        base[1], base[2] = 3.0, 2.5   # high sbytes/dbytes
    elif kind == "dos":
        base[3], base[4] = 4.0, 3.5   # high spkts/dpkts
    elif kind == "recon":
        base[9], base[10] = 2.0, 2.0  # high ct_srv_src/ct_srv_dst
    elif kind == "fuzzer":
        base[11], base[5] = 2.0, 1.5  # high ct_dst_ltm/sinpkt
    elif kind == "normal":
        base = rng.normal(loc=0.0, scale=0.2, size=12).astype(np.float32)
    else:
        base += rng.normal(loc=1.0, scale=0.8, size=12).astype(np.float32)

    return np.concatenate([base, [proto_enc, service_enc]]).astype(np.float32)


# --- Run Synthetic Predictions ---
synthetic_kinds = ["normal", "exploit", "dos", "recon", "fuzzer", "generic", "normal"]

print("\n--- Running multiple synthetic predictions ---\n")
for i, kind in enumerate(synthetic_kinds, 1):
    feats = make_synthetic_case(kind)
    x = torch.tensor(feats).view(1, -1).to(DEVICE)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(data.num_nodes, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out = torch.softmax(net(data), dim=1).detach().cpu().numpy()[0]
    pred = int(np.argmax(out))
    pred_label = attack_enc.classes_[pred] if pred < len(attack_enc.classes_) else f"Type{pred}"

    print(f"Prediction {i} (simulated {kind}): {pred_label}")
    print("Countermeasures:")
    for step in mitigation.get(pred_label, ["No data available."]):
        print(" -", step)
    print()


# --- Real Sample Prediction ---
print("--- Running prediction on a real dataset sample ---\n")
from dotenv import load_dotenv
load_dotenv()
dataset_path = os.environ.get("DATASET_PATH", "data")
csv_file = os.path.join(dataset_path, "UNSW_NB15_training-set.csv")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, low_memory=True, nrows=5000)
    df.columns = [c.strip().lower() for c in df.columns]
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip().replace("", "Normal").fillna("Normal")

    # Encode
    df["proto_enc"] = LabelEncoder().fit_transform(df["proto"].astype(str))
    df["service_enc"] = LabelEncoder().fit_transform(df["service"].astype(str))

    # Scale numeric
    for c in NUMERIC_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    scaler = StandardScaler()
    df[NUMERIC_FEATURE_COLS] = scaler.fit_transform(df[NUMERIC_FEATURE_COLS])

    sample = df.sample(1).iloc[0]
    features = np.concatenate([
        sample[NUMERIC_FEATURE_COLS].to_numpy(),
        [sample["proto_enc"], sample["service_enc"]],
    ]).astype(np.float32)

    x = torch.tensor(features).view(1, -1).to(DEVICE)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
    real_data = Data(x=x, edge_index=edge_index)
    real_data.batch = torch.zeros(real_data.num_nodes, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out = torch.softmax(net(real_data), dim=1).detach().cpu().numpy()[0]

    pred = int(np.argmax(out))
    pred_label = attack_enc.classes_[pred] if pred < len(attack_enc.classes_) else f"Type{pred}"

    print(f"Predicted Attack Type: {pred_label}")
    print(f"Actual:                {sample['attack_cat']}")
    print("Countermeasures:")
    for step in mitigation.get(pred_label, ["No data available."]):
        print(" -", step)
else:
    print(f"[warn] CSV not found at {csv_file} — skipping real sample test")
