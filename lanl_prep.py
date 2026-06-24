#!/usr/bin/env python3
"""
LANL Comprehensive Cyber-Security Events -> time-windowed authentication graphs.

Implements IMPROVEMENTS.md Part IV (§16): stream auth.txt(.gz) line-by-line, bucket into
time-windowed snapshots, build directed computer->computer authentication graphs with
edge/node features, and label edges malicious from redteam.txt(.gz). Output is a set of PyG
snapshot graphs for edge-level lateral-movement (APT) detection.

The dataset (data-fence gated) is NOT downloaded here. Fetch `auth.txt.gz` and `redteam.txt.gz`
from https://csr.lanl.gov/data/cyber1/ (accept the agreement -> signed URLs) and place them in a
directory, then:

    python lanl_prep.py --data_dir <dir> --t_start 0 --t_end 1209600 --bucket 3600
    python lanl_prep.py --selftest          # no data needed: synthetic correctness check

auth schema  : time,src_user@dom,dst_user@dom,src_comp,dst_comp,auth_type,logon_type,auth_orient,success
redteam      : time,user@dom,src_comp,dst_comp
"""

import os
import gzip
import csv
import json
import argparse
import tempfile
from collections import defaultdict

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    HAVE_TORCH = True
except Exception:                                   # selftest of parsing/labeling can run without torch
    HAVE_TORCH = False

EDGE_FEAT_NAMES = ["count", "fail_count", "fail_ratio", "n_users",
                   "n_auth_types", "off_hours_frac", "is_new_pair"]
NODE_FEAT_NAMES = ["out_count", "in_count", "out_deg", "in_deg", "fail_out", "fail_in"]


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _open(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="ignore") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8", errors="ignore")


def load_redteam(path):
    """Return dict keyed (src_comp, dst_comp) -> list of (time, user). Tiny file."""
    idx = defaultdict(list)
    n = 0
    with _open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 4:
                continue
            try:
                t = int(p[0])
            except ValueError:
                continue                            # skip header if present
            user, src, dst = p[1], p[2], p[3]
            idx[(src, dst)].append((t, user))
            n += 1
    return idx, n


def stream_auth(path, t_start=0, t_end=None, logon_only=True):
    """Yield (time, fields) for auth events within [t_start, t_end). Assumes time-sorted file
    (true for LANL auth.txt) so it can stop early at t_end."""
    with _open(path) as f:
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) < 9:
                continue
            try:
                t = int(p[0])
            except ValueError:
                continue
            if t < t_start:
                continue
            if t_end is not None and t >= t_end:
                break
            # auth_orientation is column 7 ("LogOn"/"LogOff"/"TGS"/...); red-team are LogOns.
            # (column 6 is logon_type = Network/Interactive/... — do NOT filter on it.)
            if logon_only and p[7] != "LogOn":
                continue
            yield t, p


# ---------------------------------------------------------------------------
# Bucketing + graph construction
# ---------------------------------------------------------------------------
def iter_buckets(rows, bucket):
    """Group a time-sorted (time, fields) stream into consecutive buckets of `bucket` seconds."""
    cur, buf = None, []
    for t, p in rows:
        b = t // bucket
        if cur is None:
            cur = b
        if b != cur:
            yield cur, buf
            buf, cur = [], b
        buf.append((t, p))
    if buf:
        yield cur, buf


def _redteam_hit(rt_idx, src, dst, b_start, b_end):
    """True if any red-team event for (src,dst) falls in [b_start, b_end)."""
    for (t, _user) in rt_idx.get((src, dst), ()):
        if b_start <= t < b_end:
            return True
    return False


def build_snapshot(bucket_id, rows, bucket, rt_idx, seen_pairs):
    """Aggregate one bucket's auth events into a labeled directed graph dict.
    Mutates `seen_pairs` (global novelty tracking). Returns a dict (torch-free)."""
    b_start = bucket_id * bucket
    b_end = b_start + bucket

    agg = defaultdict(lambda: {"count": 0, "fail": 0, "users": set(),
                               "atypes": set(), "off": 0})
    node_out = defaultdict(int); node_in = defaultdict(int)
    node_out_dst = defaultdict(set); node_in_src = defaultdict(set)
    node_fail_out = defaultdict(int); node_fail_in = defaultdict(int)

    for t, p in rows:
        src, dst = p[3], p[4]
        if not src or not dst or src == "?" or dst == "?":
            continue
        success = p[8]
        is_fail = success.lower().startswith("fail")
        hour = (t // 3600) % 24
        off = 1 if (hour < 7 or hour >= 19) else 0
        e = agg[(src, dst)]
        e["count"] += 1
        e["fail"] += int(is_fail)
        e["users"].add(p[1])
        e["atypes"].add(p[5])
        e["off"] += off
        node_out[src] += 1; node_in[dst] += 1
        node_out_dst[src].add(dst); node_in_src[dst].add(src)
        node_fail_out[src] += int(is_fail); node_fail_in[dst] += int(is_fail)

    nodes = sorted(set(node_out) | set(node_in))
    nidx = {c: i for i, c in enumerate(nodes)}

    x = np.zeros((len(nodes), len(NODE_FEAT_NAMES)), dtype=np.float32)
    for c, i in nidx.items():
        x[i] = [node_out[c], node_in[c], len(node_out_dst[c]),
                len(node_in_src[c]), node_fail_out[c], node_fail_in[c]]

    src_idx, dst_idx, eattr, elabel = [], [], [], []
    n_mal = 0
    for (src, dst), e in agg.items():
        is_new = 0 if (src, dst) in seen_pairs else 1
        seen_pairs.add((src, dst))
        cnt = e["count"]
        eattr.append([cnt, e["fail"], e["fail"] / cnt, len(e["users"]),
                      len(e["atypes"]), e["off"] / cnt, is_new])
        src_idx.append(nidx[src]); dst_idx.append(nidx[dst])
        lab = 1 if _redteam_hit(rt_idx, src, dst, b_start, b_end) else 0
        elabel.append(lab); n_mal += lab

    return {
        "bucket": int(bucket_id), "t_start": int(b_start), "t_end": int(b_end),
        "nodes": nodes,
        "x": x,
        "edge_index": np.array([src_idx, dst_idx], dtype=np.int64) if src_idx
                      else np.zeros((2, 0), dtype=np.int64),
        "edge_attr": np.array(eattr, dtype=np.float32) if eattr
                     else np.zeros((0, len(EDGE_FEAT_NAMES)), dtype=np.float32),
        "edge_y": np.array(elabel, dtype=np.int64),
        "n_mal": int(n_mal),
    }


def downsample_benign(snap, ratio, rng):
    """Keep all malicious edges; subsample benign edges to `ratio`x malicious (train only).
    ratio<=0 disables. Operates on the snapshot dict's edge arrays in place; returns it."""
    y = snap["edge_y"]
    if ratio <= 0 or y.size == 0:
        return snap
    mal = np.where(y == 1)[0]
    ben = np.where(y == 0)[0]
    if len(mal) == 0:
        keep_ben = rng.choice(ben, size=min(len(ben), ratio), replace=False) if len(ben) else ben
        keep = np.sort(keep_ben)
    else:
        k = min(len(ben), ratio * len(mal))
        keep_ben = rng.choice(ben, size=k, replace=False)
        keep = np.sort(np.concatenate([mal, keep_ben]))
    snap["edge_index"] = snap["edge_index"][:, keep]
    snap["edge_attr"] = snap["edge_attr"][keep]
    snap["edge_y"] = snap["edge_y"][keep]
    return snap


def to_pyg(snap):
    if not HAVE_TORCH:
        raise RuntimeError("torch/torch_geometric not available")
    d = Data(
        x=torch.from_numpy(snap["x"]),
        edge_index=torch.from_numpy(snap["edge_index"]),
        edge_attr=torch.from_numpy(snap["edge_attr"]),
        edge_y=torch.from_numpy(snap["edge_y"]),
    )
    d.num_nodes = snap["x"].shape[0]
    return d


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_all(auth_path, redteam_path, t_start, t_end, bucket,
              benign_ratio=0, logon_only=True, out_dir=None, min_edges=1):
    rt_idx, n_rt = load_redteam(redteam_path)
    print(f"  redteam events loaded: {n_rt} ({len(rt_idx)} distinct src->dst pairs)")
    rng = np.random.default_rng(42)
    seen_pairs = set()
    snaps, total_e, total_mal = [], 0, 0

    rows = stream_auth(auth_path, t_start, t_end, logon_only=logon_only)
    for bid, buf in iter_buckets(rows, bucket):
        snap = build_snapshot(bid, buf, bucket, rt_idx, seen_pairs)
        if snap["edge_index"].shape[1] < min_edges:
            continue
        if benign_ratio > 0:
            snap = downsample_benign(snap, benign_ratio, rng)
        total_e += snap["edge_y"].size
        total_mal += int((snap["edge_y"] == 1).sum())
        snaps.append(snap)

    print(f"  built {len(snaps)} snapshots | edges={total_e:,} | malicious edges={total_mal}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if HAVE_TORCH:
            for s in snaps:
                torch.save(to_pyg(s), os.path.join(out_dir, f"snap_{s['bucket']}.pt"))
        manifest = {
            "n_snapshots": len(snaps), "total_edges": int(total_e),
            "malicious_edges": int(total_mal), "bucket_seconds": bucket,
            "t_start": t_start, "t_end": t_end, "benign_ratio": benign_ratio,
            "edge_feat_names": EDGE_FEAT_NAMES, "node_feat_names": NODE_FEAT_NAMES,
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  saved snapshots + manifest to {out_dir}")
    return snaps


# ---------------------------------------------------------------------------
# Self-test (synthetic LANL-format data; no download required)
# ---------------------------------------------------------------------------
def selftest():
    print("[selftest] generating synthetic LANL-format data...")
    d = tempfile.mkdtemp(prefix="lanl_selftest_")
    auth_p = os.path.join(d, "auth.txt")
    rt_p = os.path.join(d, "redteam.txt")
    bucket = 3600

    # Bucket 0 (t in [0,3600)): benign C1->C2, C2->C3 logons.
    # Bucket 1 (t in [3600,7200)): benign C1->C2 + MALICIOUS C1->C3 (matches redteam).
    rows = [
        # t, src_user, dst_user, src_comp, dst_comp, auth_type, logon_type, auth_orientation, success
        (10,   "U1@D", "U1@D", "C1", "C2", "Kerberos", "Network",     "LogOn",  "Success"),
        (20,   "U1@D", "U1@D", "C1", "C2", "Kerberos", "Network",     "LogOn",  "Fail"),
        (30,   "U2@D", "U2@D", "C2", "C3", "NTLM",     "Network",     "LogOn",  "Success"),
        (35,   "U2@D", "U2@D", "C2", "C9", "NTLM",     "Network",     "LogOff", "Success"),  # filtered (not LogOn)
        (3700, "U1@D", "U1@D", "C1", "C2", "Kerberos", "Interactive", "LogOn",  "Success"),
        (3800, "U2@D", "U2@D", "C1", "C3", "Kerberos", "Network",     "LogOn",  "Success"),  # malicious
        (3850, "U2@D", "U2@D", "C1", "C3", "Kerberos", "Network",     "LogOn",  "Success"),  # same edge, same bucket
    ]
    with open(auth_p, "w") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    with open(rt_p, "w") as f:
        csv.writer(f).writerow((3800, "U2@D", "C1", "C3"))

    snaps = build_all(auth_p, rt_p, t_start=0, t_end=None, bucket=bucket,
                      benign_ratio=0, logon_only=True, out_dir=None)

    assert len(snaps) == 2, f"expected 2 snapshots, got {len(snaps)}"
    b0, b1 = snaps[0], snaps[1]

    # Bucket 0: nodes {C1,C2,C3} (C9 row filtered), edges C1->C2, C2->C3, both benign
    assert set(b0["nodes"]) == {"C1", "C2", "C3"}, b0["nodes"]
    assert b0["edge_index"].shape[1] == 2, b0["edge_index"].shape
    assert b0["edge_y"].sum() == 0, "bucket 0 should have no malicious edges"

    # Bucket 0 C1->C2 edge: count=2, fail=1, fail_ratio=0.5, is_new=1
    eidx = {(b0["nodes"][s], b0["nodes"][dd]): i
            for i, (s, dd) in enumerate(zip(*b0["edge_index"]))}
    i12 = eidx[("C1", "C2")]
    fa = dict(zip(EDGE_FEAT_NAMES, b0["edge_attr"][i12]))
    assert fa["count"] == 2 and fa["fail_count"] == 1 and abs(fa["fail_ratio"] - 0.5) < 1e-6, fa
    assert fa["is_new_pair"] == 1, fa

    # Bucket 1: C1->C3 aggregated (count=2) and labeled malicious; C1->C2 is NOT new now
    assert b1["edge_y"].sum() == 1, "bucket 1 should have exactly 1 malicious edge"
    eidx1 = {(b1["nodes"][s], b1["nodes"][dd]): i
             for i, (s, dd) in enumerate(zip(*b1["edge_index"]))}
    i13 = eidx1[("C1", "C3")]
    assert b1["edge_y"][i13] == 1, "C1->C3 must be malicious"
    fa13 = dict(zip(EDGE_FEAT_NAMES, b1["edge_attr"][i13]))
    assert fa13["count"] == 2, fa13
    i12b = eidx1[("C1", "C2")]
    assert dict(zip(EDGE_FEAT_NAMES, b1["edge_attr"][i12b]))["is_new_pair"] == 0, "C1->C2 seen before"

    # benign downsampling keeps all malicious
    rng = np.random.default_rng(0)
    ds = downsample_benign({k: (v.copy() if isinstance(v, np.ndarray) else v)
                            for k, v in b1.items()}, ratio=1, rng=rng)
    assert int((ds["edge_y"] == 1).sum()) == 1, "downsampling must keep malicious edges"

    if HAVE_TORCH:
        g = to_pyg(b1)
        assert g.edge_index.shape[1] == g.edge_attr.shape[0] == g.edge_y.shape[0]
        assert g.x.shape[1] == len(NODE_FEAT_NAMES)
        print("[selftest] PyG conversion OK:", g)

    print("[selftest] ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LANL auth -> labeled authentication graphs")
    ap.add_argument("--data_dir", help="dir containing auth.txt(.gz) and redteam.txt(.gz)")
    ap.add_argument("--auth", default=None, help="override path to auth file")
    ap.add_argument("--redteam", default=None, help="override path to redteam file")
    ap.add_argument("--t_start", type=int, default=0)
    ap.add_argument("--t_end", type=int, default=None, help="end time (s); default = whole file")
    ap.add_argument("--bucket", type=int, default=3600, help="snapshot window seconds")
    ap.add_argument("--benign_ratio", type=int, default=0,
                    help="benign:malicious edge ratio to keep (0 = keep all)")
    ap.add_argument("--all_logons", action="store_true", help="do not filter to LogOn events")
    ap.add_argument("--out_dir", default="results/lanl_graphs")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        selftest()
        raise SystemExit(0)

    def _find(name_opts, override):
        if override:
            return override
        for n in name_opts:
            p = os.path.join(a.data_dir, n)
            if os.path.isfile(p):
                return p
        raise FileNotFoundError(f"none of {name_opts} found in {a.data_dir}")

    if not a.data_dir and not (a.auth and a.redteam):
        ap.error("provide --data_dir (or --auth and --redteam), or use --selftest")
    auth = _find(["auth.txt.gz", "auth.txt"], a.auth)
    redteam = _find(["redteam.txt.gz", "redteam.txt"], a.redteam)
    print(f"auth   : {auth}\nredteam: {redteam}")
    build_all(auth, redteam, a.t_start, a.t_end, a.bucket,
              benign_ratio=a.benign_ratio, logon_only=not a.all_logons, out_dir=a.out_dir)
