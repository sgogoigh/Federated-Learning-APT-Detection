# UNSW-NB15 Dataset — Comprehensive Reference

> **Kaggle Source**: [mrwellsdavid/unsw-nb15](https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15)  
> **Downloaded via**: `kagglehub.dataset_download("mrwellsdavid/unsw-nb15")`   
> **Environment Variable**: `DATASET_PATH` (set in `.env`)

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [File Inventory](#2-file-inventory)
3. [Column Schema (45 Features)](#3-column-schema-45-features)
4. [Attack Categories](#4-attack-categories)
5. [Attack Subcategories and Event Counts](#5-attack-subcategories-and-event-counts)
6. [Class Distribution Analysis](#6-class-distribution-analysis)
7. [Training vs. Test Set — Why Test is Larger](#7-training-vs-test-set--why-test-is-larger)
8. [Raw Partition Files (UNSW-NB15_1 to _4)](#8-raw-partition-files-unsw-nb15_1-to-_4)
9. [Kaggle Download Process](#9-kaggle-download-process)
10. [Usage in This Project](#10-usage-in-this-project)
11. [Known Data Quality Notes](#11-known-data-quality-notes)

---

## 1. Dataset Overview

The **UNSW-NB15** dataset was created by the Australian Centre for Cyber Security (ACCS) at the University of New South Wales Canberra. It was generated using the IXIA PerfectStorm tool in the Cyber Range Lab, producing a hybrid of real normal network traffic and synthesised attack traffic.

The dataset captures **network flow records** (similar to NetFlow) extracted from raw PCAP files using the Argus and Bro-IDS tools, with 49 features computed per flow.

### Key Facts

| Property | Value |
|---|---|
| Total records (all raw parts) | ~2,540,044 |
| Features per record | 49 (raw) / 45 (train/test sets) |
| Attack categories | 9 attack types + 1 Normal |
| Normal traffic records | ~2,218,761 (87.3% of total) |
| Attack traffic records | ~321,283 (12.7% of total) |
| Time period | Jan–Feb 2015, captured over 31 hours |
| Traffic generator | IXIA PerfectStorm |
| Feature extraction | Argus + Bro-IDS tools |

---

## 2. File Inventory


| File | Size | Rows | Description |
|---|---|---|---|
| `UNSW_NB15_training-set.csv` | 15.4 MB | **82,332** | Official training partition |
| `UNSW_NB15_testing-set.csv` | 32.3 MB | **175,341** | Official test partition |
| `UNSW-NB15_1.csv` | 169.0 MB | ~700,000 | Raw partition 1 (all 49 features) |
| `UNSW-NB15_2.csv` | 165.2 MB | ~700,000 | Raw partition 2 |
| `UNSW-NB15_3.csv` | 154.6 MB | ~600,000 | Raw partition 3 |
| `UNSW-NB15_4.csv` | 97.6 MB | ~540,044 | Raw partition 4 |
| `NUSW-NB15_features.csv` | 4.0 KB | 49 rows | Feature name + description schema |
| `UNSW-NB15_LIST_EVENTS.csv` | 4.6 KB | 209 rows | Attack categories, subcategories, event counts |

**Total dataset size**: ~634 MB across all files  
**Combined raw data**: ~2.54 million records

---

## 3. Column Schema (45 Features)

The official training and test sets each have **45 columns**. The first column is a row ID; the last two are the labels.

### Identifier
| # | Column | Type | Description |
|---|---|---|---|
| 1 | `id` | int | Sequential row identifier |

### Network Flow Features (Basic)
| # | Column | Type | Description |
|---|---|---|---|
| 2 | `dur` | float | Record total duration (seconds) |
| 3 | `proto` | str | Transaction protocol (e.g., tcp, udp, arp, ospf, icmp) |
| 4 | `service` | str | Service type (http, ftp, smtp, dns, ssh, ftp-data, irc; `-` = none detected) |
| 5 | `state` | str | Transaction state and its dependent protocol (e.g., FIN, INT, CON, REQ, ECO) |
| 6 | `spkts` | int | Source-to-destination packet count |
| 7 | `dpkts` | int | Destination-to-source packet count |
| 8 | `sbytes` | int | Source-to-destination byte count |
| 9 | `dbytes` | int | Destination-to-source byte count |
| 10 | `rate` | float | Total packets per second (bidirectional) |
| 11 | `sttl` | int | Source-to-destination time-to-live value |
| 12 | `dttl` | int | Destination-to-source time-to-live value |
| 13 | `sload` | float | Source bits per second (load) |
| 14 | `dload` | float | Destination bits per second (load) |
| 15 | `sloss` | int | Source packets retransmitted or dropped |
| 16 | `dloss` | int | Destination packets retransmitted or dropped |
| 17 | `sinpkt` | float | Source inter-packet arrival time (ms) |
| 18 | `dinpkt` | float | Destination inter-packet arrival time (ms) |
| 19 | `sjit` | float | Source jitter (ms) |
| 20 | `djit` | float | Destination jitter (ms) |

### TCP Window & Connection Features
| # | Column | Type | Description |
|---|---|---|---|
| 21 | `swin` | int | Source TCP window advertisement value |
| 22 | `stcpb` | int | Source TCP base sequence number |
| 23 | `dtcpb` | int | Destination TCP base sequence number |
| 24 | `dwin` | int | Destination TCP window advertisement value |
| 25 | `tcprtt` | float | TCP connection setup round-trip time (syn → syn-ack → ack) |
| 26 | `synack` | float | TCP connection setup time (syn → syn-ack) |
| 27 | `ackdat` | float | TCP connection setup time (syn-ack → ack) |
| 28 | `smean` | int | Mean of flow packet size transmitted by the source |
| 29 | `dmean` | int | Mean of flow packet size transmitted by the destination |
| 30 | `trans_depth` | int | Pipelined depth into the connection of http request/response transaction |
| 31 | `response_body_len` | int | Actual uncompressed content size of the data transferred from http |

### Connection Table Features (ct_)
These are computed per-connection time-window statistics:

| # | Column | Type | Description |
|---|---|---|---|
| 32 | `ct_srv_src` | int | # connections with same service and source address in last 100 records |
| 33 | `ct_state_ttl` | int | # records with same protocol/state/TTL combination |
| 34 | `ct_dst_ltm` | int | # connections with same destination address in last 100 records |
| 35 | `ct_src_dport_ltm` | int | # connections with same source address and destination port in last 100 records |
| 36 | `ct_dst_sport_ltm` | int | # connections with same destination address and source port in last 100 records |
| 37 | `ct_dst_src_ltm` | int | # connections with same source and destination addresses in last 100 records |

### FTP/HTTP Specific Features
| # | Column | Type | Description |
|---|---|---|---|
| 38 | `is_ftp_login` | int | Binary: 1 if the session has an FTP login, else 0 |
| 39 | `ct_ftp_cmd` | int | # of FTP commands in the session (0 if not FTP) |
| 40 | `ct_flw_http_mthd` | int | # of HTTP methods (GET, POST) in the session |
| 41 | `ct_src_ltm` | int | # connections with same source address in last 100 records |
| 42 | `ct_srv_dst` | int | # connections with same service and destination address in last 100 records |
| 43 | `is_sm_ips_ports` | int | Binary: 1 if source and destination IPs and ports are equal |

### Labels
| # | Column | Type | Description |
|---|---|---|---|
| 44 | `attack_cat` | str | Attack category name (10 classes including Normal) |
| 45 | `label` | int | Binary label: 0 = Normal, 1 = Attack |

> **Note**: The raw partition files (UNSW-NB15_1.csv through _4.csv) have **49 columns** including `srcip`, `sport`, `dstip`, `dsport`, and `Ltime`/`Stime` (timestamps) which are removed from the official train/test sets. The train/test sets also rename some columns (e.g. `sinpkt` → was `Sintpkt` in some versions).

---

## 4. Attack Categories

The dataset contains **10 classes** total: 9 attack types plus Normal traffic.

| Label Value | Class Name | Description |
|---|---|---|
| — | **Normal** | Legitimate network traffic |
| — | **Fuzzers** | Attempts to cause crashes by sending semi-random data to protocols |
| — | **Analysis** | Port scanning, spam, and HTML file penetration |
| — | **Backdoors** | Remote access technique that bypasses normal authentication |
| — | **DoS** | Denial-of-Service attacks (resource exhaustion) |
| — | **Exploits** | Known software or OS vulnerabilities being exploited |
| — | **Generic** | Generic attacks targeting block ciphers regardless of key |
| — | **Reconnaissance** | Information gathering through probes and scanning |
| — | **Shellcode** | Attacks injecting/executing shell code |
| — | **Worms** | Self-propagating malicious code |

**LabelEncoder alphabetical ordering** (used in training code):
```
0: Analysis
1: Backdoor      ← note: dataset uses "Backdoors" in some files
2: DoS
3: Exploits
4: Fuzzers
5: Generic
6: Normal
7: Reconnaissance
8: Shellcode
9: Worms
```
> **CRITICAL**: This is the correct class order after `LabelEncoder.fit_transform()`. The `test_model_final_cpu.py` script had this order wrong (it placed Normal at index 0). Always save and reload the fitted encoder from `results/attack_encoder.pkl`.

---

## 5. Attack Subcategories and Event Counts

From `UNSW-NB15_LIST_EVENTS.csv` — the full raw dataset contains **2,540,044** total records:

### Normal
| Count |
|---|
| 2,218,761 |

### Fuzzers (total: ~23,247)
Subcategories: FTP, HTTP, RIP, SMB, Syslog, PPTP, DCERPC, OSPF, TFTP, BGP

### Reconnaissance (total: ~17,356)
Subcategories: Telnet, SNMP, SunRPC (TCP/UDP), NetBIOS, DNS, HTTP, ICMP, SCTP, MSSQL, SMTP

### Shellcode (total: ~1,296)
Subcategories by OS: FreeBSD, HP-UX, NetBSD, AIX, SCO Unix, Linux, Decoders, IRIX, OpenBSD, Mac OS X, BSD, Windows, BSDi, Multiple OS, Solaris

### Analysis (total: ~2,677)
Subcategories: HTML, Port Scanner, Spam

### Backdoors (total: 2,329)
Single subcategory (generic backdoor traffic)

### DoS (total: ~14,079)
Subcategories: Ethernet, Microsoft Office, VNC, IRC, RDP, TCP, FTP, LDAP, Oracle, TFTP, DCERPC, XINETD, SNMP, ISAKMP, NTP, Telnet, CUPS, Hypervisor, ICMP, SunRPC, IMAP, Asterisk, Browser, Cisco Skinny, SIP, SMTP, SSL, DNS, IIS Web Server, Miscellaneous, RTSP, IGMP, NetBIOS/SMB, Oracle, Windows Explorer, HTTP, LDAP, SNMP

### Exploits (total: ~43,396)
Subcategories include: Browser, Apache, Microsoft IIS, SMTP, RADIUS, RTSP, Web Application, Office Document, Miscellaneous, PHP, SMB, SIP, DNS, FTP, SCADA, SunRPC, IMAP, Webserver, Clientside variants, etc.

### Generic (total: ~215,481)
Subcategories: IXIA (dominant — 207,243 records), SIP, SMTP, HTTP, TFTP, Superflow

### Worms (total: 174)
Single subcategory

---

## 6. Class Distribution Analysis

### Official Train/Test Sets

| Class | Training Rows | Test Rows | Train % | Test % |
|---|---|---|---|---|
| Normal | ~56,000 | ~93,000 | 68.0% | 53.1% |
| Analysis | ~2,000 | ~677 | 2.4% | 0.4% |
| Backdoor | ~1,746 | ~583 | 2.1% | 0.3% |
| DoS | ~4,089 | ~4,089 | 5.0% | 2.3% |
| Exploits | ~8,876 | ~11,132 | 10.8% | 6.4% |
| Fuzzers | ~6,062 | ~6,062 | 7.4% | 3.5% |
| Generic | ~18,871 | ~58,871 | 22.9% | 33.6% |
| Reconnaissance | ~3,496 | ~3,496 | 4.2% | 2.0% |
| Shellcode | ~378 | ~1,133 | 0.5% | 0.6% |
| Worms | ~174 | ~174 | 0.2% | 0.1% |

> **Note**: Exact counts vary; above are approximate based on total rows and class proportions documented in original UNSW-NB15 papers.

### Class Imbalance

The dataset has **significant class imbalance**:
- Normal class dominates at ~53–68% of samples
- Worms, Shellcode, and Backdoor are very rare (< 2%)
- This is the primary reason the model collapsed to predicting "Normal" for all inputs

**Mitigation required**:
1. Inverse-frequency class weights in `CrossEntropyLoss` ✅ (already implemented)
2. Loading the full dataset (all ~257k rows) instead of `nrows=5000` — **pending fix**
3. Optionally: SMOTE oversampling on minority classes before graph construction

---

## 7. Training vs. Test Set — Why Test is Larger

This is a deliberate design decision by the UNSW-NB15 authors, not a mistake.

**Training set**: 82,332 rows (~15 MB)  
**Test set**: 175,341 rows (~32 MB) — **2.13× larger**

### Reason

The official train/test split was designed to:
1. **Mimic real-world deployment**: In intrusion detection, models are trained on a smaller, curated set but must generalise to much larger, continuously incoming traffic
2. **Stress-test generalisation**: A larger test set with different proportions tests whether models truly learn attack patterns rather than memorising training data
3. **Ensure minority class coverage**: Rare attack types (Worms, Shellcode) that appear in tiny numbers in training are more broadly represented across the test set for evaluation completeness

### Practical Implications for This Project

Because we are using **federated learning** (no pre-defined train/test split at the global level), we do **not** use the official train/test partition directly. Instead:
- We load the training set CSV as our data source
- We build graphs from it and split within each client (70% train / 15% val / 15% test per client)
- The official test set is available as a **held-out global benchmark** for final evaluation after federated training

**Recommendation**: After federated training completes, evaluate the global model on `UNSW_NB15_testing-set.csv` for a true unbiased benchmark.

---

## 8. Raw Partition Files (UNSW-NB15_1 to _4)

The four raw CSV files contain the **original unprocessed network flow records** with **49 columns** (vs. 45 in the train/test sets).

### Additional columns in raw files (not in train/test sets)

| Column | Description |
|---|---|
| `srcip` | Source IP address |
| `sport` | Source port number |
| `dstip` | Destination IP address |
| `dsport` | Destination port number |
| `Stime` | Record start time (Unix epoch) |
| `Ltime` | Record last time (Unix epoch) |

> These are removed from the train/test sets to prevent IP/port memorisation and to ensure the model learns flow-level statistical features rather than topology.

### File Sizes and Estimated Row Counts

| File | Size | Approx. Rows |
|---|---|---|
| `UNSW-NB15_1.csv` | 169.0 MB | ~700,000 |
| `UNSW-NB15_2.csv` | 165.2 MB | ~700,000 |
| `UNSW-NB15_3.csv` | 154.6 MB | ~600,000 |
| `UNSW-NB15_4.csv` | 97.6 MB | ~540,044 |
| **Total** | **586.4 MB** | **~2,540,044** |

### When to use raw files vs. train/test sets

| Use Case | Recommended File |
|---|---|
| Federated training (current project) | `UNSW_NB15_training-set.csv` |
| Global benchmark evaluation | `UNSW_NB15_testing-set.csv` |
| Feature engineering research | Raw `UNSW-NB15_1.csv` through `_4.csv` |
| IP/network graph analysis | Raw files (has `srcip`, `dstip`) |
| Reproducing original paper results | Train + Test sets |

---

## 9. Kaggle Download Process

The dataset is downloaded using the `kagglehub` library.

### Download Notebook

Located at: `Federated-Learning-APT-Detection/.ipynb_checkpoints/dataset_download.ipynb`

```python
import kagglehub

# Download latest version (v1)
path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")

print("Path to dataset files:", path)
# Output: C:\Users\sgogo\.cache\kagglehub\datasets\mrwellsdavid\unsw-nb15\versions\1
```

### Download Details

| Property | Value |
|---|---|
| Dataset identifier | `mrwellsdavid/unsw-nb15` |
| Download size (compressed) | ~149 MB |
| Moved to project location | `C:\...\APT Detection Fed Learning\datasets\mrwellsdavid\unsw-nb15\versions\1` |
| Environment variable | `DATASET_PATH` in `.env` file |

### Prerequisites

```bash
pip install kagglehub   # already added to requirements.txt
```

A Kaggle API key must be configured at `~/.kaggle/kaggle.json`:
```json
{"username": "your_kaggle_username", "key": "your_api_key"}
```

---

## 10. Usage in This Project

### Which file the model uses

The federated training notebook (`apt-fl-model.ipynb`) loads:

```python
# Current (before fix) — capped at 5,000 rows
df = pd.read_csv("data/UNSW_NB15_reduced_features.csv", low_memory=False, nrows=5000)

# After IMPROVEMENTS.md fix — load full training set, no cap
df = pd.read_csv(os.environ["DATASET_PATH"] + "/UNSW_NB15_training-set.csv", low_memory=False)
```

The path should be resolved from the `.env` file:
```
DATASET_PATH=C:\Users\sgogo\OneDrive\Desktop\APT Detection Fed Learning\datasets\mrwellsdavid\unsw-nb15\versions\1
```

### Features used by the model (14 of 45)

The project uses a **subset of 14 features** from the training set:

```python
feature_cols = [
    "sport",        # Source port (NOTE: not in train/test — use proto instead)
    "dsport",       # Destination port (same note)
    "dur",          # Flow duration
    "sbytes",       # Source bytes
    "dbytes",       # Destination bytes
    "Spkts",        # Source packets (note capitalisation)
    "Dpkts",        # Destination packets
    "Sintpkt",      # Source inter-packet time (note capitalisation)
    "ct_srv_src",   # Connection table feature
    "ct_srv_dst",   # Connection table feature
    "ct_dst_ltm",   # Connection table feature
    "hour_of_day",  # Derived from timestamp (not in raw data — must be added)
    "proto_enc",    # Label-encoded proto
    "service_enc",  # Label-encoded service
]
```

> **Important**: `sport`, `dsport`, and `hour_of_day` are **not** present in `UNSW_NB15_training-set.csv`. The training set removes IP/port columns. The project originally used a "reduced features" CSV (`UNSW_NB15_reduced_features.csv`) which may have preserved these or engineered them. When switching to the full training set, the preprocessing pipeline must be updated to:
> 1. Use `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm` from the available columns
> 2. Drop `sport`/`dsport` references (or substitute with `ct_src_dport_ltm`/`ct_dst_sport_ltm`)
> 3. Either drop `hour_of_day` or derive it from Stime in the raw files

### Preprocessing Steps Applied

1. Strip whitespace from column names
2. Fill empty `attack_cat` with `"Normal"`
3. Coerce `Label`, `hour_of_day`, and all numeric columns to float (fillna 0)
4. `LabelEncoder` on `proto` → `proto_enc`
5. `LabelEncoder` on `service` → `service_enc`
6. `StandardScaler` on 12 numeric feature columns
7. `LabelEncoder` on `attack_cat` → `attack_label` (0–9, alphabetical)

---

## 11. Known Data Quality Notes

### 1. Column name inconsistencies across files

The train/test sets use lowercase column names (`spkts`, `dpkts`), while the project code uses mixed-case (`Spkts`, `Dpkts`, `Sintpkt`). The `preprocess_df()` function handles this with `.strip()` but does **not** case-normalise. Always verify column names with `df.columns.tolist()` after loading.

### 2. Missing `attack_cat` values

Empty strings in `attack_cat` represent Normal traffic in some file versions. The preprocessing correctly fills these with `"Normal"`.

### 3. `sport`/`dsport` absent from official train/test sets

The project's 14-feature list references `sport` and `dsport`, which exist only in the raw partition files (`UNSW-NB15_1.csv` through `_4.csv`), not in the official `UNSW_NB15_training-set.csv`. When loading the full training set, these features must be substituted or dropped.

**Suggested substitutes**:
- `sport` → `ct_src_dport_ltm` (connection count from same source/dest-port pair)
- `dsport` → `ct_dst_sport_ltm` (connection count from same dest/source-port pair)

### 4. `hour_of_day` is a derived feature

The training-set CSV does not contain a timestamp column. `hour_of_day` must be either:
- Dropped from the feature set (reduce to 12 features)
- Derived from the raw file `Stime` column and joined in

### 5. Generic class contains IXIA-generated traffic

The Generic class is dominated by `IXIA` subcategory traffic (207,243 of ~215,481 Generic records), which is synthetic load-test traffic from the IXIA tool — not real-world attack traffic. This means Generic is semantically different from the other attack categories.

### 6. Label name inconsistency

The dataset uses `"Backdoors"` (plural) in some files and `"Backdoor"` (singular) in others. The `LabelEncoder` will treat these as different classes. The preprocessing code should normalise: `df["attack_cat"] = df["attack_cat"].str.strip().str.replace("Backdoors", "Backdoor")`.

---

*This document is a living reference. Update it whenever the dataset path, preprocessing logic, or feature selection changes.*
