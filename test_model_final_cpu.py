import os
import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool

# ---------- Safe Environment Setup ----------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
DEVICE = torch.device("cpu")

print("Running model test on CPU...")

# ---------- Label Encoder (same as during training) ----------
attack_enc = LabelEncoder()
attack_enc.classes_ = np.array([
    "Normal", "Exploits", "Reconnaissance", "Fuzzers",
    "DoS", "Generic", "Analysis", "Backdoor",
    "Shellcode", "Worms"
])

# ---------- Model Definition ----------
class GraphSAGEClassifier(nn.Module):
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

# ---------- Load Model ----------
model_path = "results/global_model.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

print(f"Loading model from {model_path} ...")
state = torch.load(model_path, map_location=DEVICE)

in_dim, num_classes = 14, len(attack_enc.classes_)
net = GraphSAGEClassifier(in_dim, 64, num_classes).to(DEVICE)
net.load_state_dict(state, strict=False)
net.eval()

print("✅ Model loaded successfully")

# ---------- Create Dummy Graph Input ----------
dummy_x = torch.randn(10, in_dim).to(DEVICE)
dummy_edge_index = torch.randint(0, 10, (2, 20)).to(DEVICE)
dummy_batch = torch.zeros(10, dtype=torch.long).to(DEVICE)
data = Data(x=dummy_x, edge_index=dummy_edge_index, batch=dummy_batch)

# ---------- Run Inference ----------
with torch.no_grad():
    out = torch.softmax(net(data), dim=1)[0].cpu().numpy()
pred = int(np.argmax(out))
attack_type = attack_enc.classes_[pred]

print(f"\nPredicted Attack Type: {attack_type}")
print(f"Malicious: {'Yes' if attack_type != 'Normal' else 'No'}")

# ---------- Countermeasure Suggestions ----------
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
for step in mitigation.get(attack_type, ["No data available."]):
    print(" -", step)
