#!/usr/bin/env python3
"""
Lightweight CPU-only model test — no CSV required.

Loads the saved model and encoder, runs inference on a synthetic graph,
and outputs the predicted attack type with countermeasures.
"""

import os
import pickle
import json

import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

# --- Safe Environment Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

print("Running model test on CPU...")

# --- Feature config (must match train.py) ---
IN_DIM = 14

# --- Load saved encoder (CODE-04 + CODE-05 fix) ---
encoder_path = "results/attack_encoder.pkl"
if os.path.exists(encoder_path):
    print(f"Loading encoder from {encoder_path}")
    with open(encoder_path, "rb") as f:
        attack_enc = pickle.load(f)
else:
    # CODE-05 FIX: use correct ALPHABETICAL order (matches LabelEncoder)
    print("[warn] Encoder file not found — using alphabetical class order")
    attack_enc = LabelEncoder()
    attack_enc.classes_ = np.array([
        "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers",
        "Generic", "Normal", "Reconnaissance", "Shellcode", "Worms"
    ])

# --- Load Model Config ---
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

num_classes = len(attack_enc.classes_)

# --- Model Definition ---
class GraphSAGEClassifier(nn.Module):
    def __init__(self, num_protos, num_services, num_states, continuous_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.proto_emb = nn.Embedding(num_protos, 16)
        self.service_emb = nn.Embedding(num_services, 8)
        self.state_emb = nn.Embedding(num_states, 8)
        
        in_dim = continuous_dim + 16 + 8 + 8
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

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

# --- Load Model ---
model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

print(f"Loading model from {model_path} ...")
state = torch.load(model_path, map_location=DEVICE, weights_only=True)
net = GraphSAGEClassifier(
    num_protos=model_config["num_protos"],
    num_services=model_config["num_services"],
    num_states=model_config["num_states"],
    continuous_dim=39,
    hidden_dim=model_config["hidden_dim"],
    num_classes=num_classes
).to(DEVICE)
net.load_state_dict(state, strict=False)
net.eval()
print("Model loaded successfully")

# --- Create Dummy Graph Input ---
dummy_cont = torch.randn(10, 39).to(DEVICE)
dummy_proto = torch.randint(0, model_config["num_protos"], (10, 1)).float().to(DEVICE)
dummy_service = torch.randint(0, model_config["num_services"], (10, 1)).float().to(DEVICE)
dummy_state = torch.randint(0, model_config["num_states"], (10, 1)).float().to(DEVICE)
dummy_x = torch.cat([dummy_cont, dummy_proto, dummy_service, dummy_state], dim=1)

dummy_edge_index = torch.randint(0, 10, (2, 20)).to(DEVICE)
dummy_batch = torch.zeros(10, dtype=torch.long).to(DEVICE)
data = Data(x=dummy_x, edge_index=dummy_edge_index, batch=dummy_batch)

# --- Run Inference ---
with torch.no_grad():
    out = torch.softmax(net(data), dim=1)[0].cpu().numpy()
pred = int(np.argmax(out))
attack_type = attack_enc.classes_[pred]

print(f"\nPredicted Attack Type: {attack_type}")
print(f"Malicious: {'Yes' if attack_type != 'Normal' else 'No'}")

# --- Countermeasure Suggestions ---
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

print("\nRecommended Countermeasures:")
for step in mitigation.get(attack_type, ["No data available."]):
    print(" -", step)
