import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
from torch_geometric.nn import SAGEConv, global_mean_pool
import os

# Limit PyTorch CPU threads to avoid memory spikes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_num_threads(1)


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

#DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

print("Loading CSV to reconstruct label encoder...")
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=True, nrows=5000)

le_attack = LabelEncoder()
df["attack_cat"] = df["attack_cat"].replace("", np.nan).fillna("Normal")
df["attack_label"] = le_attack.fit_transform(df["attack_cat"].astype(str))
attack_enc = le_attack
num_classes = len(attack_enc.classes_)
print(f"Loaded encoder with {num_classes} attack types.")

model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Please train first.")

print("Loading trained model weights...")
in_dim = 16
net = GraphSAGEClassifier(in_dim, 64, num_classes).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)

try:
    net.load_state_dict(state_dict)
    print("Model weights loaded successfully.")
except RuntimeError as e:
    print("Warning: Shape mismatch while loading model:", e)
    net.load_state_dict(state_dict, strict=False)

net.eval()

print("Running demo prediction on synthetic input...")
dummy_x = torch.randn(10, in_dim).to(DEVICE)
dummy_edge_index = torch.randint(0, 10, (2, 20)).to(DEVICE)
dummy_batch = torch.zeros(10, dtype=torch.long).to(DEVICE)

data = Data(x=dummy_x, edge_index=dummy_edge_index, batch=dummy_batch)
with torch.no_grad():
    out = torch.softmax(net(data.unsqueeze(0)), dim=1)[0].detach().cpu().numpy()
pred = int(np.argmax(out))
attack_type = attack_enc.inverse_transform([pred])[0]
malicious = attack_type != "Normal"

print(f"Predicted attack type: {attack_type}")
print(f"Malicious: {'Yes' if malicious else 'No'}")

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

print("\nRecommended Countermeasures:")
for step in mitigation.get(attack_type, ["No data."]):
    print(" -", step)

print("\nTesting model on a real sample from dataset...")
sample = df.sample(1).iloc[0]
print(f"Sample attack_cat: {sample['attack_cat']} | Label: {sample['Label']}")

features = np.concatenate([
    sample[["sport", "dsport", "dur", "sbytes", "dbytes", "Spkts", "Dpkts", "Sintpkt",
            "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "hour_of_day"]].to_numpy(),
    [sample["proto_enc"], sample["service_enc"]]
]).astype(np.float32)

x = torch.tensor(features).view(1, -1).to(DEVICE)
edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(DEVICE)
batch = torch.zeros(1, dtype=torch.long).to(DEVICE)
real_data = Data(x=x, edge_index=edge_index, batch=batch)

with torch.no_grad():
    out = torch.softmax(net(real_data.unsqueeze(0)), dim=1)[0].detach().cpu().numpy()
pred = int(np.argmax(out))
attack_type = attack_enc.inverse_transform([pred])[0]

print(f"Model predicted: {attack_type}")
print(f"Actual label: {sample['attack_cat']}")
