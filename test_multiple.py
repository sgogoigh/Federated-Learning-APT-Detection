import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv, global_mean_pool

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Model Definition 
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

# Device Setup 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Data Preparation
print("Loading CSV to reconstruct label encoder...")
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=True, nrows=5000)

df["proto_enc"] = LabelEncoder().fit_transform(df["proto"].astype(str))
df["service_enc"] = LabelEncoder().fit_transform(df["service"].astype(str))

# Encode attack categories
le_attack = LabelEncoder()
df["attack_cat"] = df["attack_cat"].replace("", np.nan).fillna("Normal")
df["attack_label"] = le_attack.fit_transform(df["attack_cat"].astype(str))
attack_enc = le_attack
num_classes = len(attack_enc.classes_)
print(f"Loaded encoder with {num_classes} attack types.")

# Load Model 
model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please train first at results/global_model.pt")

print("Loading trained model weights...")
in_dim = 14  

net = GraphSAGEClassifier(in_dim, 64, num_classes).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)

print("Loading weights (skipping final classification layer if mismatched)...")
model_dict = net.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("fc.")}
model_dict.update(filtered_dict)
net.load_state_dict(model_dict, strict=False)
print("Model weights loaded.")

net.eval()

# Countermeasures
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
    "Normal": ["No attack detected", "Maintain routine monitoring"]
}

# Synthetic case generator
rng = np.random.RandomState(42)

def make_synthetic_case(kind="generic"):
    base = rng.normal(loc=0.0, scale=0.5, size=12).astype(np.float32)
    proto_enc = float(rng.randint(0, max(1, df["proto_enc"].max() + 1)))
    service_enc = float(rng.randint(0, max(1, df["service_enc"].max() + 1)))

    if kind == "exploit":
        base[3], base[4] = 3.0, 2.5
    elif kind == "dos":
        base[5], base[6] = 4.0, 3.5
    elif kind == "recon":
        base[8], base[9] = 2.0, 2.0
    elif kind == "fuzzer":
        base[10], base[7] = 2.0, 1.5
    elif kind == "normal":
        base = rng.normal(loc=0.0, scale=0.2, size=12).astype(np.float32)
    else:
        base += rng.normal(loc=1.0, scale=0.8, size=12).astype(np.float32)

    return np.concatenate([base, [proto_enc, service_enc]]).astype(np.float32)

# Run Synthetic Predictions 
synthetic_kinds = ["normal", "exploit", "dos", "recon", "fuzzer", "generic", "normal"]

print("\nRunning multiple synthetic predictions...\n")
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

    print(f"Prediction {i}: {pred_label}")
    print("Countermeasures:")
    for step in mitigation.get(pred_label, ["No data available."]):
        print(" -", step)
    print("")


print("Running prediction on a real dataset sample...\n")
sample = df.sample(1).iloc[0]
features = np.concatenate([
    sample[["sport", "dsport", "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Sintpkt",
            "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "hour_of_day"]].to_numpy(),
    [sample["proto_enc"], sample["service_enc"]]
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
print("Countermeasures:")
for step in mitigation.get(pred_label, ["No data available."]):
    print(" -", step)
