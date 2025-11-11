
# APT Detection Using Federated GraphSAGE

This repository contains an implementation of an Advanced Persistent Threat (APT) detection framework using a Federated Graph Neural Network (GraphSAGE) model. The solution leverages the UNSW-NB15 dataset to identify network intrusions and recommend countermeasures based on attack classification.

---

## 1. Introduction

Advanced Persistent Threats (APTs) represent a sophisticated form of cyberattack characterized by stealth, persistence, and high-impact objectives. Traditional signature-based systems often fail to detect such evolving threats. This project introduces a federated learning-based approach that enables decentralized APT detection without sharing raw data, preserving privacy while improving collaborative intelligence.

The model utilizes GraphSAGE, a graph neural network architecture that captures topological and feature relationships within network flow data. The model is trained in a federated setting, where multiple clients collaboratively contribute to a global model without direct data exchange.

---

## 2. Project Overview

This project includes the following components:

- **apt-fl-model.ipynb** – A Jupyter notebook for federated training of the GraphSAGE model.
- **test.py** – A Python script for evaluating the trained global model on synthetic and real data.
- **data/** – Directory containing the preprocessed UNSW-NB15 dataset.
- **results/** – Directory containing the trained global model weights (`global_model.pt`).

---

## 3. Model Architecture

The APT detection model is based on **GraphSAGE (Graph Sample and Aggregate)**, implemented using PyTorch Geometric. The architecture consists of:

- Two GraphSAGE convolutional layers for neighborhood aggregation.
- A ReLU activation function to introduce non-linearity.
- Global mean pooling to condense graph-level features.
- A fully connected output layer for classification.
- Dropout regularization to mitigate overfitting.

### Model Definition

```python
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
```

---

## 4. Dataset Description

The model uses the **UNSW-NB15 Reduced Features Dataset**, which contains network traffic records labeled by attack category.  
Key attributes include:

- `sport`, `dsport`: Source and destination ports
- `dur`: Duration of flow
- `sbytes`, `dbytes`: Byte counts sent and received
- `Spkts`, `Dpkts`: Number of packets sent and received
- `proto`, `service`: Encoded categorical features
- `attack_cat`: Attack category label

Preprocessing includes label encoding for categorical variables (`proto`, `service`, `attack_cat`) using `LabelEncoder` from scikit-learn.

---

## 5. Code Explanation

### 5.1 Data Preparation

```python
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=True, nrows=5000)
df["proto_enc"] = LabelEncoder().fit_transform(df["proto"].astype(str))
df["service_enc"] = LabelEncoder().fit_transform(df["service"].astype(str))
le_attack = LabelEncoder()
df["attack_cat"] = df["attack_cat"].replace("", np.nan).fillna("Normal")
df["attack_label"] = le_attack.fit_transform(df["attack_cat"].astype(str))
```

This section loads the dataset, performs encoding on categorical columns, and prepares numeric features for the model.

### 5.2 Model Loading and Initialization

```python
model_path = "results/global_model.pt"
net = GraphSAGEClassifier(in_dim=14, hidden_dim=64, num_classes=num_classes).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)
model_dict = net.state_dict()
filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
model_dict.update(filtered_dict)
net.load_state_dict(model_dict)
```

The trained model is loaded from disk with safety checks to ensure only matching parameters are updated. This prevents loading errors due to shape mismatches.

### 5.3 Synthetic Data Generation

```python
def make_synthetic_case(kind="generic"):
    if kind == "normal":
        feats = rng.normal(loc=0.0, scale=0.5, size=14)
    elif kind == "exploit":
        feats = rng.normal(loc=1.0, scale=0.8, size=14)
    elif kind == "dos":
        feats = rng.normal(loc=2.0, scale=1.0, size=14)
    elif kind == "recon":
        feats = rng.normal(loc=-1.0, scale=0.7, size=14)
    elif kind == "fuzzer":
        feats = rng.normal(loc=0.3, scale=1.2, size=14)
    else:
        feats = rng.normal(loc=0.0, scale=1.0, size=14)
    return feats.astype(np.float32)
```

This function generates synthetic network flow data for different attack categories by sampling from normal distributions with varying means and variances.

### 5.4 Inference and Countermeasure Generation

The model predicts the attack category for each sample and provides relevant countermeasures based on a predefined dictionary.

```python
with torch.no_grad():
    logits = net(data)
    out = torch.softmax(logits / temperature, dim=1).detach().cpu().numpy()[0]
pred = int(np.argmax(out))
attack_type = attack_enc.inverse_transform([pred])[0].strip()
```

Countermeasures are retrieved as:

```python
for step in mitigation.get(attack_type, ["No data available."]):
    print(" -", step)
```

---

## 6. Countermeasure Mapping

| Attack Type     | Example Countermeasures |
|------------------|-------------------------|
| Exploits         | Patch vulnerable software, enable IPS, run vulnerability scans |
| Reconnaissance   | Deploy IDS, block ICMP sweeps, rate-limit unknown IPs |
| Fuzzers          | Input validation, segment networks, limit error verbosity |
| DoS              | Rate limiting, use CDN/WAF, monitor anomalies |
| Generic          | Update antivirus, sandbox unknown apps, inspect traffic anomalies |
| Analysis         | Secure logging, restrict admin rights, monitor host activity |
| Backdoor         | Re-image systems, rotate credentials, use MFA |
| Shellcode        | Patch memory vulnerabilities, enable DEP/ASLR, deploy EDR |
| Worms            | Isolate networks, patch systems, disable SMB/RPC |
| Normal           | No attack detected, maintain routine monitoring |

---

## 7. How to Run

### 7.1 Install Dependencies

```bash
pip install torch torchvision torchaudio torch-geometric scikit-learn pandas numpy
```

### 7.2 Prepare Dataset

Ensure the dataset file is located at:

```
data/UNSW_NB15_reduced_features.csv
```

### 7.3 Train the Model

Open and execute the following notebook to train the federated model:

```
apt-fl-model.ipynb
```

This generates a global model checkpoint:

```
results/global_model.pt
```

### 7.4 Test the Model

To run inference and obtain attack predictions:

```bash
python test.py
```

---

## 8. Example Output

```
Using device: cuda:0
Loaded encoder with 10 attack types: ['Analysis', 'Backdoor', ..., 'Worms']

Running predictions on synthetic samples...

Synthetic Case 2:
Predicted Attack Type: Exploits
Countermeasures:
 - Patch vulnerable software
 - Enable IPS
 - Run vulnerability scans
 - Harden systems
```

---