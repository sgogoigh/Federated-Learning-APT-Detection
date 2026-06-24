#!/usr/bin/env python3
"""
Federated GNN for lateral-movement (APT) detection on LANL authentication graphs.

Implements IMPROVEMENTS.md Part IV, stage 3. Loads the labeled snapshot graphs produced by
`lanl_prep.py`, trains an edge-level classifier (benign vs red-team authentication) in a
federated setting (FedAdam), and reports the rare-event metric suite (ROC-AUC, PR-AUC /
average precision, detection@low-FPR, precision@k) on a TEMPORAL test split. Includes a
no-graph baseline (edge features only) and a centralized-GNN baseline so the graph's
contribution is provable.

Usage:
    python train_lanl.py --graph_dir ../datasets/lanl_graphs --t_split 950400 --t_test_end 1382400
"""
import os, sys, glob, json, argparse, copy, time, random
import numpy as np

class Logger(object):
    def __init__(self, path):
        self.terminal = sys.stdout; self.log = open(path, "w", encoding="utf-8")
    def write(self, m):
        try: self.terminal.write(m)
        except UnicodeEncodeError:
            enc = getattr(self.terminal, "encoding", None) or "ascii"
            self.terminal.write(m.encode(enc, "replace").decode(enc))
        self.log.write(m); self.log.flush()          # flush so progress is visible live
    def flush(self): self.terminal.flush(); self.log.flush()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Model: GraphSAGE node embeddings -> edge classifier
# ---------------------------------------------------------------------------
class EdgeGNN(nn.Module):
    def __init__(self, node_dim, edge_dim, hid=64, dropout=0.3, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.conv1 = SAGEConv(node_dim, hid)
            self.conv2 = SAGEConv(hid, hid)
            edge_in = 2 * hid + edge_dim
        else:
            edge_in = edge_dim                      # no-graph baseline: edge features only
        self.relu = nn.ReLU(); self.drop = nn.Dropout(dropout)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in, hid), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hid, 1))

    def forward(self, data):
        ei = data.edge_index
        if self.use_graph:
            h = self.relu(self.conv1(data.x, ei)); h = self.drop(h)
            h = self.relu(self.conv2(h, ei))
            z = torch.cat([h[ei[0]], h[ei[1]], data.edge_attr], dim=1)
        else:
            z = data.edge_attr
        return self.edge_mlp(z).squeeze(-1)          # [E] logits


# ---------------------------------------------------------------------------
# Server optimizer (FedAdam) — same idea as train.py's ServerOpt
# ---------------------------------------------------------------------------
class ServerOpt:
    def __init__(self, init_state, mode="fedadam", lr=0.05, b1=0.9, b2=0.99, tau=1e-3):
        self.mode, self.lr, self.b1, self.b2, self.tau = mode, lr, b1, b2, tau
        self.g = {k: v.detach().float().clone() for k, v in init_state.items()}
        self.m = {k: torch.zeros_like(v) for k, v in self.g.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.g.items()}
    def step(self, states, weights):
        tot = float(sum(weights)) or 1.0
        avg = {k: sum(s[k].float() * w for s, w in zip(states, weights)) / tot for k in self.g}
        if self.mode == "fedavg":
            self.g = avg; return self.g
        for k in self.g:
            d = avg[k] - self.g[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * d
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * d * d
            self.g[k] = self.g[k] + self.lr * self.m[k] / (self.v[k].sqrt() + self.tau)
        return self.g


# ---------------------------------------------------------------------------
# Data loading / feature scaling / split
# ---------------------------------------------------------------------------
def load_snaps(graph_dir, bucket):
    files = glob.glob(os.path.join(graph_dir, "snap_*.pt"))
    snaps = []
    for fp in files:
        bid = int(os.path.basename(fp)[5:-3])
        g = torch.load(fp, weights_only=False)
        g.t_start = bid * bucket
        snaps.append(g)
    snaps.sort(key=lambda g: g.t_start)
    return snaps


def fit_scalers(train_snaps):
    xs = np.concatenate([g.x.numpy() for g in train_snaps if g.x.shape[0]], 0)
    es = np.concatenate([g.edge_attr.numpy() for g in train_snaps if g.edge_attr.shape[0]], 0)
    xs = np.log1p(np.clip(xs, 0, None)); es = np.log1p(np.clip(es, 0, None))
    return ((xs.mean(0), xs.std(0) + 1e-6), (es.mean(0), es.std(0) + 1e-6))


def apply_scalers(snaps, scx, sce):
    for g in snaps:
        if g.x.shape[0]:
            x = np.log1p(np.clip(g.x.numpy(), 0, None)); g.x = torch.tensor((x - scx[0]) / scx[1], dtype=torch.float32)
        if g.edge_attr.shape[0]:
            e = np.log1p(np.clip(g.edge_attr.numpy(), 0, None)); g.edge_attr = torch.tensor((e - sce[0]) / sce[1], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training (local) with negative sampling
# ---------------------------------------------------------------------------
def sample_idx(y, neg_per_pos, min_neg):
    pos = (y == 1).nonzero(as_tuple=True)[0]; neg = (y == 0).nonzero(as_tuple=True)[0]
    if len(neg) == 0: return pos
    k = neg_per_pos * len(pos) if len(pos) > 0 else min_neg
    k = min(len(neg), max(k, 1))
    sel = neg[torch.randperm(len(neg))[:k]]
    return torch.cat([pos, sel])


def train_local(snaps, global_state, cfg, name):
    model = EdgeGNN(cfg["node_dim"], cfg["edge_dim"], cfg["hid"], use_graph=cfg["use_graph"]).to(DEVICE)
    if global_state: model.load_state_dict(global_state, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    pos_w = torch.tensor([cfg["neg_per_pos"]], dtype=torch.float32, device=DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    n_edges = 0
    for ep in range(cfg["epochs"]):
        model.train(); order = list(range(len(snaps))); random.shuffle(order)
        losses = []
        for gi in order:
            g = snaps[gi].to(DEVICE)
            if g.edge_index.shape[1] == 0: continue
            logit = model(g)
            idx = sample_idx(g.edge_y, cfg["neg_per_pos"], cfg["min_neg"]).to(DEVICE)
            loss = crit(logit[idx], g.edge_y[idx].float())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            if ep == 0: n_edges += int(g.edge_index.shape[1])
            g.to("cpu")
    return copy.deepcopy(model.state_dict()), n_edges


@torch.no_grad()
def predict(model, snaps):
    model.eval(); ys, ps = [], []
    for g in snaps:
        if g.edge_index.shape[1] == 0: continue
        g = g.to(DEVICE)
        p = torch.sigmoid(model(g)).cpu().numpy()
        ps.append(p); ys.append(g.edge_y.cpu().numpy()); g.to("cpu")
    return np.concatenate(ys), np.concatenate(ps)


def rare_metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "base_rate": float(y.mean())}
    if 0 < y.sum() < len(y):
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        fpr, tpr, _ = roc_curve(y, p)
        for tgt in (1e-2, 1e-3, 1e-4):
            m = fpr <= tgt
            out[f"tpr@fpr={tgt:g}"] = float(tpr[m].max()) if m.any() else 0.0
        k = int(y.sum())
        out[f"prec@{k}"] = float(y[np.argsort(-p)[:k]].mean()) if k else 0.0
    else:
        out["roc_auc"] = out["pr_auc"] = None
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(a):
    t0 = time.time()
    print(f"Device: {DEVICE}\nLoading snapshots from {a.graph_dir} ...")
    snaps = load_snaps(a.graph_dir, a.bucket)
    print(f"  {len(snaps)} snapshots, t in [{snaps[0].t_start}, {snaps[-1].t_start}]")

    trainval = [g for g in snaps if g.t_start < a.t_split]
    test = [g for g in snaps if a.t_split <= g.t_start < a.t_test_end]
    # Stratified 85/15 train/val within the trainval window so val gets a proportional share of
    # malicious snapshots (PR-AUC selection is useless with only a handful of positives in val).
    mal_s = [g for g in trainval if int(g.edge_y.sum()) > 0]
    ben_s = [g for g in trainval if int(g.edge_y.sum()) == 0]
    random.shuffle(mal_s); random.shuffle(ben_s)
    nvm, nvb = max(1, int(a.val_frac * len(mal_s))), int(a.val_frac * len(ben_s))
    val = mal_s[:nvm] + ben_s[:nvb]
    train = mal_s[nvm:] + ben_s[nvb:]
    random.shuffle(val); random.shuffle(train)

    def pos(snaps_): return int(sum(int(g.edge_y.sum()) for g in snaps_))
    print(f"  train: {len(train)} snaps ({pos(train)} mal edges) | "
          f"val: {len(val)} snaps ({pos(val)} mal) | test: {len(test)} snaps ({pos(test)} mal)")
    if pos(test) == 0:
        print("  [WARN] test window has no malicious edges — adjust --t_split/--t_test_end")

    scx, sce = fit_scalers(train)
    for grp in (train, val, test): apply_scalers(grp, scx, sce)
    node_dim = train[0].x.shape[1]; edge_dim = train[0].edge_attr.shape[1]

    base_cfg = dict(node_dim=node_dim, edge_dim=edge_dim, hid=a.hidden_dim, lr=1e-3,
                    epochs=a.epochs_local, neg_per_pos=a.neg_per_pos, min_neg=256, use_graph=True)
    comparison = {}

    # ---- Baseline 1: no-graph (edge features only), centralized ----
    if not a.skip_baselines:
        print("\n[baseline] no-graph edge-MLP (centralized)...")
        cfg = dict(base_cfg, use_graph=False, epochs=a.baseline_epochs)
        st, _ = train_local(train, None, cfg, "mlp")
        m = EdgeGNN(node_dim, edge_dim, a.hidden_dim, use_graph=False).to(DEVICE); m.load_state_dict(st)
        comparison["nograph_mlp"] = rare_metrics(*predict(m, test))
        print("   ", {k: comparison["nograph_mlp"][k] for k in ("roc_auc", "pr_auc")})

        print("[baseline] centralized GNN...")
        cfg = dict(base_cfg, use_graph=True, epochs=a.baseline_epochs)
        st, _ = train_local(train, None, cfg, "cen")
        m = EdgeGNN(node_dim, edge_dim, a.hidden_dim, use_graph=True).to(DEVICE); m.load_state_dict(st)
        comparison["centralized_gnn"] = rare_metrics(*predict(m, test))
        print("   ", {k: comparison["centralized_gnn"][k] for k in ("roc_auc", "pr_auc")})

    # ---- Federated GNN ----
    print(f"\nFederated GNN: {a.num_clients} clients, {a.rounds} rounds, "
          f"server_opt={a.server_opt}, neg_per_pos={a.neg_per_pos}")
    clients = {i: [] for i in range(a.num_clients)}
    for i, g in enumerate(train): clients[i % a.num_clients].append(g)  # round-robin (IID over snapshots)

    init = EdgeGNN(node_dim, edge_dim, a.hidden_dim, use_graph=True).to(DEVICE)
    server = ServerOpt(init.state_dict(), mode=a.server_opt, lr=a.server_lr)
    gstate = server.g
    gnet = EdgeGNN(node_dim, edge_dim, a.hidden_dim, use_graph=True).to(DEVICE)
    best_val, best_state, best_round, traj = -1.0, copy.deepcopy(gstate), 0, []

    for r in range(a.rounds):
        states, weights = [], []
        for i in range(a.num_clients):
            if not clients[i]: continue
            st, ne = train_local(clients[i], gstate, base_cfg, f"C{i+1}")
            states.append(st); weights.append(ne)
        gstate = server.step(states, weights)
        gnet.load_state_dict(gstate, strict=False)
        vy, vp = predict(gnet, val); ty, tp = predict(gnet, test)
        vm, tm = rare_metrics(vy, vp), rare_metrics(ty, tp)
        traj.append({"round": r + 1, "val_pr_auc": vm["pr_auc"], "test_pr_auc": tm["pr_auc"],
                     "test_roc_auc": tm["roc_auc"]})
        print(f"  Round {r+1}/{a.rounds}: val PR-AUC={vm['pr_auc']} | "
              f"test PR-AUC={tm['pr_auc']} ROC-AUC={tm['roc_auc']}")
        if vm["pr_auc"] is not None and vm["pr_auc"] > best_val:
            best_val, best_state, best_round = vm["pr_auc"], copy.deepcopy(gstate), r + 1
            torch.save(best_state, "results/lanl_global_model.pt")

    # ---- Final test on best-val checkpoint ----
    gnet.load_state_dict(best_state, strict=False)
    ty, tp = predict(gnet, test)
    comparison["federated_gnn"] = rare_metrics(ty, tp)
    print(f"\nSelected round {best_round} (val PR-AUC={best_val:.4f})")

    # ---- Save + report ----
    with open("results/lanl_comparison.json", "w") as f:
        json.dump({"models": comparison, "trajectory": traj,
                   "split": {"t_split": a.t_split, "t_test_end": a.t_test_end}}, f, indent=2)

    print("\n" + "=" * 64)
    print("LANL lateral-movement detection — official temporal test split")
    print("=" * 64)
    hdr = ["model", "ROC-AUC", "PR-AUC", "TPR@1e-3", "TPR@1e-4"]
    print(f"  {hdr[0]:16s} {hdr[1]:>8s} {hdr[2]:>8s} {hdr[3]:>9s} {hdr[4]:>9s}")
    for nm, m in comparison.items():
        ra = f"{m['roc_auc']:.4f}" if m["roc_auc"] is not None else "n/a"
        pa = f"{m['pr_auc']:.4f}" if m["pr_auc"] is not None else "n/a"
        t3 = f"{m.get('tpr@fpr=0.001', 0):.3f}"; t4 = f"{m.get('tpr@fpr=0.0001', 0):.3f}"
        print(f"  {nm:16s} {ra:>8s} {pa:>8s} {t3:>9s} {t4:>9s}")
    print(f"  (test base rate = {comparison['federated_gnn']['base_rate']:.2e}, "
          f"{comparison['federated_gnn']['n_pos']} malicious / {comparison['federated_gnn']['n']:,} edges)")

    # PR + trajectory plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    rs = [t["round"] for t in traj]
    ax[0].plot(rs, [t["val_pr_auc"] or 0 for t in traj], "s-", label="val PR-AUC")
    ax[0].plot(rs, [t["test_pr_auc"] or 0 for t in traj], "o-", label="test PR-AUC")
    if "nograph_mlp" in comparison and comparison["nograph_mlp"]["pr_auc"]:
        ax[0].axhline(comparison["nograph_mlp"]["pr_auc"], ls="--", color="crimson",
                      label=f"no-graph MLP ({comparison['nograph_mlp']['pr_auc']:.3f})")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("PR-AUC"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Federated convergence (PR-AUC)")
    ty2, tp2 = predict(gnet, test)
    from sklearn.metrics import precision_recall_curve
    pr, rc, _ = precision_recall_curve(ty2, tp2)
    ax[1].plot(rc, pr); ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
    ax[1].set_title("Federated GNN PR curve (test)"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig("results/lanl_pr_curves.png", dpi=150); plt.close(fig)

    h, rem = divmod(int(time.time() - t0), 3600); mn, sc = divmod(rem, 60)
    print(f"\nSaved results/lanl_comparison.json, lanl_pr_curves.png, lanl_global_model.pt")
    print(f"Done in {h}h {mn}m {sc}s.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Federated GNN APT detection on LANL")
    p.add_argument("--graph_dir", default="../datasets/lanl_graphs")
    p.add_argument("--bucket", type=int, default=3600)
    p.add_argument("--t_split", type=int, default=950400, help="train/test boundary (s); train < split")
    p.add_argument("--t_test_end", type=int, default=1382400)
    p.add_argument("--num_clients", type=int, default=7)
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--epochs_local", type=int, default=2)
    p.add_argument("--baseline_epochs", type=int, default=10)
    p.add_argument("--hidden_dim", type=int, default=64)
    p.add_argument("--neg_per_pos", type=int, default=50)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--server_opt", choices=["fedadam", "fedavg"], default="fedadam")
    p.add_argument("--server_lr", type=float, default=0.05)
    p.add_argument("--skip_baselines", action="store_true")
    p.add_argument("--log", default="lanl-train-logs.txt")
    a = p.parse_args()
    sys.stdout = Logger(a.log)
    main(a)
