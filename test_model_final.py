#!/usr/bin/env python3
"""
Test the trained federated GraphSAGE model on real data from the CSV dataset.

Loads the saved model (with full state_dict including fc layer) and the saved
LabelEncoder, then runs predictions on synthetic and real samples.
"""

import os
import pickle
import json

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.nn import SAGEConv, global_mean_pool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

NUMERIC_FEATURE_COLS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "is_sm_ips_ports"
]
IN_DIM = 42  # 39 numeric + proto_enc + service_enc + state_enc

# --- Load model config ---
config_path = "results/model_config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        model_config = json.load(f)
else:
    model_config = {
        "num_protos": 134,
        "num_services": 14,
        "num_states": 12,
        "continuous_dim": 39,
        "hidden_dim": 64,
        "num_classes": 10
    }

# --- Load saved encoders ---
with open("results/proto_encoder.pkl", "rb") as f:
    le_proto = pickle.load(f)
with open("results/service_encoder.pkl", "rb") as f:
    le_service = pickle.load(f)
with open("results/state_encoder.pkl", "rb") as f:
    le_state = pickle.load(f)

def safe_encode(le, series, unseen_val="<unknown>"):
    classes = set(le.classes_)
    if unseen_val not in classes:
        le.classes_ = np.append(le.classes_, unseen_val)
        classes.add(unseen_val)
    series_clean = series.map(lambda x: x if x in classes else unseen_val)
    return le.transform(series_clean)

# --- Model Definition ---
class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, num_protos, num_services, num_states, continuous_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proto_emb = torch.nn.Embedding(num_protos, 16)
        self.service_emb = torch.nn.Embedding(num_services, 8)
        self.state_emb = torch.nn.Embedding(num_states, 8)
        
        in_dim = continuous_dim + 16 + 8 + 8
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, num_classes)
        self.dropout = torch.nn.Dropout(dropout)
        self.relu = torch.nn.ReLU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        cont_feats = x[:, :39]
        proto_idx = x[:, 39].long()
        service_idx = x[:, 40].long()
        state_idx = x[:, 41].long()
        
        p_emb = self.proto_emb(proto_idx)
        s_emb = self.service_emb(service_idx)
        st_emb = self.state_emb(state_idx)
        
        x_emb = torch.cat([cont_feats, p_emb, s_emb, st_emb], dim=1)
        
        x = self.relu(self.conv1(x_emb, edge_index))
        x = self.dropout(x)
        x = self.relu(self.conv2(x, edge_index))
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
    # Fallback: alphabetical order (matches LabelEncoder default)
    print("[warn] Encoder file not found — using alphabetical class order")
    attack_enc = LabelEncoder()
    attack_enc.classes_ = np.array([
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
    ])

num_classes = len(attack_enc.classes_)
print(f"Loaded encoder with {num_classes} attack types: {list(attack_enc.classes_)}")

# --- Load Model (BUG-03 FIX: load FULL state_dict including fc layer) ---
model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please train first.")

print("Loading trained model weights...")
net = GraphSAGEClassifier(
    num_protos=model_config["num_protos"],
    num_services=model_config["num_services"],
    num_states=model_config["num_states"],
    continuous_dim=39,
    hidden_dim=model_config["hidden_dim"],
    num_classes=num_classes
).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
# BUG-03 FIX: Load the complete model — do NOT filter out fc. layer
net.load_state_dict(state_dict, strict=False)
print("Model loaded successfully (full state_dict including fc layer).")
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

# --- Create Dummy Graph Input ---
dummy_cont = torch.randn(10, 39).to(DEVICE)
dummy_proto = torch.randint(0, model_config["num_protos"], (10, 1)).float().to(DEVICE)
dummy_service = torch.randint(0, model_config["num_services"], (10, 1)).float().to(DEVICE)
dummy_state = torch.randint(0, model_config["num_states"], (10, 1)).float().to(DEVICE)
dummy_x = torch.cat([dummy_cont, dummy_proto, dummy_service, dummy_state], dim=1)

dummy_edge_index = torch.randint(0, 10, (2, 20)).to(DEVICE)
data = Data(x=dummy_x, edge_index=dummy_edge_index)
data.batch = torch.zeros(data.num_nodes, dtype=torch.long).to(DEVICE)

with torch.no_grad():
    out = torch.softmax(net(data), dim=1).detach().cpu().numpy()[0]

pred = int(np.argmax(out))
attack_type = attack_enc.inverse_transform([pred])[0]
malicious = attack_type != "Normal"

print(f"Predicted attack type: {attack_type}")
print(f"Malicious: {'Yes' if malicious else 'No'}")
print("\nRecommended Countermeasures:")
for step in mitigation.get(attack_type, ["No data."]):
    print(" -", step)

# --- Real Sample Prediction ---
print("\n--- Testing model on a real sample from dataset ---")
from dotenv import load_dotenv
load_dotenv()
dataset_path = os.environ.get("DATASET_PATH", "data").strip('"').strip("'")
csv_file = os.path.join(dataset_path, "UNSW_NB15_training-set.csv")

if os.path.exists(csv_file):
    df = pd.read_csv(csv_file, low_memory=True, nrows=5000)
    df.columns = [c.strip().lower() for c in df.columns]
    df["attack_cat"] = df["attack_cat"].astype(str).str.strip().replace("", "Normal").fillna("Normal")

    # Encode
    df["proto_enc"] = safe_encode(le_proto, df["proto"].astype(str))
    df["service_enc"] = safe_encode(le_service, df["service"].astype(str))
    df["state_enc"] = safe_encode(le_state, df["state"].astype(str))

    # Scale numeric
    for c in NUMERIC_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            df[c] = 0.0
    scaler = StandardScaler()
    df[NUMERIC_FEATURE_COLS] = scaler.fit_transform(df[NUMERIC_FEATURE_COLS])

    sample = df.sample(1).iloc[0]
    print(f"Sample attack_cat: {sample['attack_cat']} | Label: {sample['label']}")

    features = np.concatenate([
        sample[NUMERIC_FEATURE_COLS].to_numpy(),
        [sample["proto_enc"], sample["service_enc"], sample["state_enc"]],
    ]).astype(np.float32)

    x = torch.tensor(features).view(1, -1).to(DEVICE)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
    real_data = Data(x=x, edge_index=edge_index)
    real_data.batch = torch.zeros(real_data.num_nodes, dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        out = torch.softmax(net(real_data), dim=1).detach().cpu().numpy()[0]

    pred = int(np.argmax(out))
    attack_type = attack_enc.inverse_transform([pred])[0]

    print(f"Model predicted: {attack_type}")
    print(f"Actual label:    {sample['attack_cat']}")
    print("\nRecommended Countermeasures:")
    for step in mitigation.get(attack_type, ["No data."]):
        print(" -", step)
else:
    print(f"[warn] CSV not found at {csv_file} — skipping real sample test")
