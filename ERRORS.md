# APT Detection Federated Learning - Known Errors and Fixes

This document serves as a catalogue of the architectural, programmatic, and pipeline errors we encountered during the development of this Federated Learning system, and the corresponding fixes we implemented.

---

### PROBLEM NO.1: Missing / Dropped Fully Connected (`fc`) Layer during Inference
**Description:** The test scripts (`test_model_final.py` and `test_multiple.py`) were manually filtering out the `fc.` layer parameters when loading the model's `state_dict`. As a result, the model's classification head was initialized with random weights during testing, leading to completely garbage outputs.
**FIX 1:** We updated all inference scripts to use `strict=False` and properly load the complete state dictionary (including the trained `fc.` layer) to retain the learned classification capabilities.

---

### PROBLEM NO.2: Inconsistent Label Decoding
**Description:** The test scripts were hardcoding label arrays or instantiating a brand-new `LabelEncoder` during inference based on a small batch of test data. Because the encoding order changes depending on the data subset, the network's predictions were completely misaligned from the training labels.
**FIX 2:** Modified the main training script (`train.py`) to persist the fitted `LabelEncoder` as a pickle object (`results/attack_encoder.pkl`). The testing scripts now load this object to decode predictions consistently.

---

### PROBLEM NO.3: AUC Computation Crashing on Partial Batches
**Description:** The `roc_auc_score` function failed and returned a silent `0.0` or threw an error because partial evaluation sets (e.g., during validation) didn't always contain samples representing every single attack class.
**FIX 3:** Passed the explicit argument `labels=list(range(num_classes))` to the `roc_auc_score` function inside the `evaluate()` loop. This explicitly informs scikit-learn of all the possible classes, allowing it to correctly compute the Macro AUC even if certain classes have zero support in that specific batch.

---

### PROBLEM NO.4: Environment Variable File Path Quotation Error
**Description:** The script loaded the dataset path from the `.env` file using `python-dotenv`. However, Windows paths enclosed in quotes (`DATASET_PATH="C:\..."`) were parsed literally. When appended to the filename using `os.path.join`, it formed an invalid path like `"C:\..."\UNSW_NB15_training-set.csv` and threw a `FileNotFoundError`.
**FIX 4:** Added `.strip('"').strip("'")` right after fetching the environment variable in all scripts to sanitize the string before executing file operations.

---

### PROBLEM NO.5: Classification Report ValueError
**Description:** During the Global Test Set evaluation, the `classification_report` function crashed with `ValueError: Number of classes, 5, does not match size of target_names, 10`. This happened because the random 10% test split did not naturally contain the rarest APT classes (e.g., Worms, Shellcode).
**FIX 5:** Passed `labels=list(range(num_classes))` to `classification_report()` to force it to evaluate and display all 10 target classes, outputting a `0` score for any missing categories rather than crashing.

---

### PROBLEM NO.6: Graph-Level Classification Causing Minority Class Erasure
**Description:** The initial architecture built graphs by taking 120 chronological network flows, connecting them, and assigning a single "majority" label to the entire graph. Because "Normal" and "Generic" classes dominate the dataset (75%), the minority APT classes were entirely erased during graph construction, resulting in massive class collapse.
**FIX 6:** Swapped the architectural design from **Graph-Level Classification** to **Node-Level Classification**. We removed `global_mean_pool` from the `GraphSAGEClassifier` and assigned labels per individual node, forcing the network to evaluate and classify all 82,332 original flows independently while preserving minority attacks.

---

### PROBLEM NO.7: FedAvg Catastrophic Forgetting via Non-IID Partitioning
**Description:** The client partitioning function (`create_clients`) used KMeans clustering on specific connection features (like Protocols and Services). This clustered the dataset into highly restricted, Non-IID partitions. One client would see only HTTP traffic, while another saw only DNS traffic. Local models rapidly "forgot" classes they weren't seeing, and when averaged globally, the competing extreme biases destroyed the network.
**FIX 7:** Replaced the KMeans logic with a randomized shuffle (IID distribution). Distributing graphs uniformly ensures every client gets a representative baseline of all protocols and attacks, eliminating the catastrophic forgetting.

---

### PROBLEM NO.8: Exploding Loss Gradients due to Missing Classes
**Description:** The `train_local` script calculated inverse-frequency class weights using the formula `1.0 / (count + 1e-6)`. If a local client partition happened to be missing an attack class entirely (count = 0), the loss penalty weight for that class skyrocketed to `1,000,000`, destabilizing gradients and halting convergence.
**FIX 8:** Implemented a mathematically stable balanced weighting formula (`total_samples / (num_classes * class_count)`). If a class is completely absent, its weight is gracefully set to `0.0`.

---

### PROBLEM NO.9: Graph Over-smoothing via Cliques
**Description:** The training pipeline grouped flows by `service_enc` and connected flows sharing `service_enc`. Since all flows in a sliced subgraph shared the same service, every subgraph became a fully connected clique. Mean neighborhood aggregation in a clique averages all node features, erasing individual flow details and causing the model to collapse to predicting only the majority class.
**FIX 9:** Chunked the dataset chronologically (retaining protocol/service diversity) and constructed K-Nearest Neighbors ($k=3$) graphs on scaled continuous features.

---

### PROBLEM NO.10: Edge Discarding via Index Mismatch
**Description:** Nodes were added to NetworkX using their original DataFrame indices. When converting to PyTorch Geometric, boundary checks discarded all edges with indices `>= 120`. Consequently, all subgraphs after the first chunk were treated as having zero edges (isolated nodes).
**FIX 10:** Used local node indices `0` to `len(sub) - 1` when adding nodes to the graph, preserving all edges.

---

### PROBLEM NO.11: Raw Categorical Float Input
**Description:** Label-encoded protocol and service variables were fed directly as continuous floats into the GraphSAGE model. Since categories have no ordered numeric meaning, this confused the neural network.
**FIX 11:** Refactored the `GraphSAGEClassifier` to use `nn.Embedding` for protocol and service features, mapping them to dense learned vectors, and updated the testing scripts to load saved encoders and config.

---

### PROBLEM NO.12: Sub-optimal Feature Subset (12 of 39 numeric columns)
**Description:** The training pipeline only used 12 out of 39 available numeric features in the CSV, ignoring 27 highly descriptive attributes (like `sload`, `dload`, `tcprtt`, `smean`, `dmean`, and connection tables) which are vital for classifying network attacks.
**FIX 12:** Expanded `NUMERIC_FEATURE_COLS` to include all 39 numeric features in the dataset.

---

### PROBLEM NO.13: Ignored `state` Categorical Column
**Description:** The `state` categorical attribute (indicating connection status like CON, FIN, INT) was completely ignored by the model, despite being highly informative of DoS and scanning activities.
**FIX 13:** Added label encoding and a 3rd `nn.Embedding` layer in the GNN model for connection state, bringing total features utilized to 42.

---

### PROBLEM NO.14: Hyper-aggressive Class Weighting (Class Collapse)
**Description:** The balanced class weight formula created extreme ratios (e.g. $850:1$ for Worms vs. Normal). This penalized rare class errors so severely that the model over-predicted them at the slightest similarity, leading to high false positives and capping overall accuracy at 62.4%.
**FIX 14:** Implemented square-root weight smoothing ($w_i = \sqrt{total / (num\_classes \times count)}$) to reduce the ratio to $29:1$. This stabilized loss and boosted accuracy to 82.3%.

---

### PROBLEM NO.15: Graph Topology Lacking Chronological Sequence
**Description:** Constructing graphs purely on continuous feature similarity (KNN) made the GNN message passing redundant with the raw features, since it only averaged nodes that were already similar in feature space.
**FIX 15:** Created a hybrid graph topology combining chronological temporal chain edges (connecting sequentially adjacent flows in the window) with KNN similarity edges.

---

### PROBLEM NO.16: Sub-optimal Early Stopping Metric (Validation Accuracy)
**Description:** Local training early stopping was based on raw validation accuracy. Because accuracy is heavily dominated by majority classes (Normal/Generic), the local models early-stopped (often at epoch 5) before they could learn minority classes.
**FIX 16:** Changed the early stopping metric to validation loss, which is smooth and represents true learning progress across all classes. We increased the patience to `5` and adjusted the class weight smoothing factor to a power of `0.6` to give slightly higher weight to minority classes.

---

> **Errors below were encountered during the June 2026 robustness rewrite (`train.py`).**

### PROBLEM NO.17: `.env` Backslash Paths Silently Corrupted by dotenv Escape Parsing
**Description:** `DATASET_PATH="datasets\mrwellsdavid\unsw-nb15\versions\1\UNSW_NB15_training-set.csv"` parsed to `...unsw-nb15\x0bersions\1\...` because `python-dotenv` interprets C-style escape sequences inside **double-quoted** values: `\v` became a vertical-tab (`\x0b`) and `\1` is also an escape. The path silently pointed nowhere → `FileNotFoundError`. This is distinct from PROBLEM NO.4 (which was about quotes forming an invalid prefix); here the *body* of the path was mangled.
**FIX 17:** Rewrote `.env` to use **forward slashes** and point to the dataset *directory* (`DATASET_PATH=../datasets/mrwellsdavid/unsw-nb15/versions/1`). Forward slashes work on Windows and contain no escapable characters. Rule of thumb: never put Windows backslash paths inside double-quoted `.env` values.

---

### PROBLEM NO.18: Relative `DATASET_PATH` Not Resolved Against the Right Anchor
**Description:** The dataset lives at `<project-parent>/datasets/...`, but `train.py` runs from `<project-parent>/Federated-Learning-APT-Detection/`. A relative `DATASET_PATH` joined against `os.getcwd()` resolved to the wrong directory, so the CSV was "not found" even though it existed one level up.
**FIX 18:** Made `resolve_paths()` try the path against several anchors (cwd, cwd's parent, the script directory and its parent) and, as a final fallback, **recursively `glob`** for the CSV filename under each anchor. The pipeline now finds the dataset regardless of where it is launched from or whether `.env` is correct.

---

### PROBLEM NO.19: `AttributeError: 'list' object has no attribute 'to'` in MLP Baseline Eval
**Description:** The first full run crashed (exit 1) in `evaluate()`. The graph models yield a PyG `Data` batch (which has `.to(DEVICE)` and `.y`), but the per-flow MLP baseline uses a plain `torch.utils.data.TensorDataset`, whose DataLoader yields a **`[X, y]` list**. The shared `evaluate()` called `batch.to(DEVICE)` unconditionally → `AttributeError`.
**FIX 19:** Branched both `evaluate()` and `val_macro_f1()` on the `is_graph` flag: graph batches do `batch.to(DEVICE)` / `batch.y`; tensor batches unpack `xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)`. Lesson: when one eval helper serves two dataloader types, handle the batch shape explicitly rather than assuming a PyG object.

---

### PROBLEM NO.20: Windows Console `UnicodeEncodeError` Risk on Non-ASCII Log Output
**Description:** Print strings contained `—` (em-dash) and similar Unicode. The log file opens as UTF-8, but the Windows terminal stream is often cp1252; writing `—` to it can raise `UnicodeEncodeError` and abort a long run mid-training.
**FIX 20:** Hardened `Logger.write()` to catch `UnicodeEncodeError` and re-encode the message to the terminal's encoding with `errors="replace"` (the UTF-8 log file still gets the original text). Long runs can no longer die on a stray glyph.



