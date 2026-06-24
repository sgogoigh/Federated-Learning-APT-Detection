# LANL Dataset — Acquisition & Preparation

The pivot dataset (IMPROVEMENTS.md Part IV) is the **LANL "Comprehensive, Multi-Source
Cyber-Security Events"** (Kent, 2015) — the graph-native, red-team-labeled benchmark for
lateral-movement / APT detection.

> ⚠️ The old `lanl_conversion.ipynb` downloaded the **wrong** release
> (`lanl-auth-dataset-1-00`, the 2017 unlabeled `time,user,computer` bipartite file — "good for
> only DSA questions"). We do **not** use that one. We need the 2015 comprehensive set, which has
> directed computer→computer logons **and** ground-truth red-team labels.

## 1. Download (data-fence gated — requires a browser, one-time)

1. Open **https://csr.lanl.gov/data/cyber1/**
2. Read and **accept the data-use agreement** on the page. It then generates **time-limited
   signed download URLs** (this is the `data-fence/<token>/...` mechanism; tokens expire in hours,
   so download promptly).
3. Download at minimum:
   - **`redteam.txt.gz`** (~20 KB — the APT labels; grab this first, it's tiny)
   - **`auth.txt.gz`** (~9 GB compressed — the authentication graph edges)
   - *(optional, for richer features later)* `proc.txt.gz`, `flows.txt.gz`, `dns.txt.gz`
4. Place them in a folder, e.g.:
   ```
   datasets/lanl/auth.txt.gz
   datasets/lanl/redteam.txt.gz
   ```

You do **not** need to decompress them — `lanl_prep.py` streams the `.gz` directly.

### Disk / scope note
`auth.txt.gz` is ~9 GB compressed (~70 GB / 1.05 B rows uncompressed). You have ~164 GB free, so
the compressed file is fine. **Do not process all 58 days at once.** Start with a time window that
contains early red-team activity (see §3). The streaming reader stops at `--t_end`, so a windowed
run never reads the whole file.

## 2. Verify the files

```bash
# first red-team rows (tiny)
python -c "import gzip;f=gzip.open('datasets/lanl/redteam.txt.gz','rt');[print(next(f).strip()) for _ in range(5)]"
# first auth rows
python -c "import gzip;f=gzip.open('datasets/lanl/auth.txt.gz','rt');[print(next(f).strip()) for _ in range(3)]"
```
Expected auth columns: `time,src_user@dom,dst_user@dom,src_comp,dst_comp,auth_type,logon_type,auth_orientation,success`.

## 3. Build the authentication graphs

```bash
# Prototype: first 14 days, hourly snapshots, keep all malicious + 200:1 benign (train scope)
python lanl_prep.py --data_dir datasets/lanl --t_start 0 --t_end 1209600 \
                    --bucket 3600 --benign_ratio 200 --out_dir results/lanl_graphs
```
- `--bucket 3600` = hourly graphs (use `86400` for daily, fewer/larger graphs).
- `--benign_ratio 200` keeps every malicious edge + 200× as many benign (training scope only —
  drop it / set 0 for the held-out test window to preserve the true base rate).
- Output: `results/lanl_graphs/snap_*.pt` (PyG graphs) + `manifest.json`.

The red-team events span the dataset; pick `--t_start/--t_end` for an **early train window** and a
**later disjoint test window** (temporal split, IMPROVEMENTS §17) — never shuffle across time.

## 4. Sanity check without the data
```bash
python lanl_prep.py --selftest      # synthetic LANL-format data; verifies parse/bucket/label/PyG
```

## 5. What each graph contains
- **Nodes** = computers active in the window. Node features (`NODE_FEAT_NAMES`):
  out/in event counts, out/in degree, out/in fail counts.
- **Edges** = directed `src_comp → dst_comp` auth aggregated in the window. Edge features
  (`EDGE_FEAT_NAMES`): count, fail_count, fail_ratio, #users, #auth_types, off-hours fraction,
  is_new_pair (first-ever occurrence of this pair).
- **`edge_y`** = 1 if the edge matches a red-team event in that window, else 0. This is the
  edge-level lateral-movement label the federated GNN will be trained to predict.

Once `results/lanl_graphs/` exists, the next step is the federated edge-classification trainer
(IMPROVEMENTS §21 stage 3), which reuses the FedAdam machinery from `train.py`.
