#!/usr/bin/env python3
"""
Experiment orchestrator for the LANL federated APT detector (IMPROVEMENTS.md Part IV §22).

Runs three experiment groups as resumable subprocess calls to train_lanl.py and aggregates the
per-run JSONs into mean +- std tables:

  1. Multi-seed comparison (seeds 42/7/123): centralized, no-graph, and federated under
     non-IID host partition with edges / positives / val_signal aggregation.
  2. pos_smooth sweep (seed 42): positive-aware weighting from sharp (1) to flat (1000),
     interpolating toward the broken uniform regime.
  3. Privacy-preserving val_signal variant is included in group 1.

Each run writes results/exp/<tag>.json. Re-running skips completed runs (resumable). After the
matrix completes, writes results/lanl_experiment_summary.{json,md}.

    python lanl_experiments.py            # run the full matrix + aggregate
    python lanl_experiments.py --aggregate_only
"""
import os, sys, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "results", "exp")
PY = sys.executable
SEEDS = [42, 7, 123]
ROUNDS, EPOCHS, CLIENTS, NEG = 15, 2, 7, 50
METRICS = ["roc_auc", "pr_auc", "tpr@fpr=0.001", "tpr@fpr=0.0001"]


def run_spec(tag, agg, seed, baselines, pos_smooth=1.0):
    return {"tag": tag, "agg": agg, "seed": seed, "baselines": baselines, "pos_smooth": pos_smooth}


def build_matrix():
    runs = []
    for s in SEEDS:
        runs.append(run_spec(f"seed{s}_positives", "positives", s, True, 1.0))   # + baselines
        runs.append(run_spec(f"seed{s}_edges", "edges", s, False))
        runs.append(run_spec(f"seed{s}_valsignal", "val_signal", s, False))
    for ps in (10, 100, 1000):
        runs.append(run_spec(f"seed42_possmooth{ps}", "positives", 42, False, float(ps)))
    return runs


def done(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return "federated_gnn" in d.get("models", {})
    except Exception:
        return False


def launch(spec):
    out = os.path.join(EXP, spec["tag"] + ".json")
    if done(out):
        print(f"  [skip] {spec['tag']} (already done)"); return out
    cmd = [PY, os.path.join(HERE, "train_lanl.py"),
           "--partition", "host", "--agg_weight", spec["agg"], "--seed", str(spec["seed"]),
           "--rounds", str(ROUNDS), "--epochs_local", str(EPOCHS), "--num_clients", str(CLIENTS),
           "--neg_per_pos", str(NEG), "--pos_smooth", str(spec["pos_smooth"]),
           "--out_json", out, "--log", os.path.join(EXP, spec["tag"] + ".log")]
    if not spec["baselines"]:
        cmd.append("--skip_baselines")
    t = time.time()
    print(f"  [run ] {spec['tag']} ...", flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  [FAIL] {spec['tag']} rc={r.returncode}\n{r.stderr[-800:]}", flush=True)
        return None
    print(f"  [done] {spec['tag']} in {int(time.time()-t)}s", flush=True)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    n = len(vals); mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5 if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n, "vals": vals}


def _load(tag):
    p = os.path.join(EXP, tag + ".json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def aggregate():
    # group 1: multi-seed mean+-std
    collect = {m: {} for m in METRICS}   # metric -> model -> [vals across seeds]
    models = ["nograph_mlp", "centralized_gnn", "fed_edges", "fed_positives", "fed_valsignal"]
    per = {m: {mod: [] for mod in models} for m in METRICS}
    for s in SEEDS:
        dp = _load(f"seed{s}_positives"); de = _load(f"seed{s}_edges"); dv = _load(f"seed{s}_valsignal")
        for m in METRICS:
            if dp:
                per[m]["nograph_mlp"].append(dp["models"].get("nograph_mlp", {}).get(m))
                per[m]["centralized_gnn"].append(dp["models"].get("centralized_gnn", {}).get(m))
                per[m]["fed_positives"].append(dp["models"].get("federated_gnn", {}).get(m))
            if de:
                per[m]["fed_edges"].append(de["models"].get("federated_gnn", {}).get(m))
            if dv:
                per[m]["fed_valsignal"].append(dv["models"].get("federated_gnn", {}).get(m))
    multiseed = {m: {mod: _stats(per[m][mod]) for mod in models} for m in METRICS}

    # group 2: pos_smooth sweep (seed 42 single)
    sweep = []
    smooth_runs = [("1", "seed42_positives")] + [(str(ps), f"seed42_possmooth{ps}") for ps in (10, 100, 1000)]
    for ps, tag in smooth_runs:
        d = _load(tag)
        if d:
            fg = d["models"].get("federated_gnn", {})
            sweep.append({"pos_smooth": ps, **{m: fg.get(m) for m in METRICS}})

    summary = {"seeds": SEEDS, "metrics": METRICS, "multiseed": multiseed, "pos_smooth_sweep": sweep}
    with open(os.path.join(HERE, "results", "lanl_experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # markdown + console
    lines = ["# LANL Federated APT — Experiment Summary\n",
             f"Seeds: {SEEDS} | non-IID host partition | {CLIENTS} clients, {ROUNDS} rounds\n",
             "## Multi-seed (mean ± std)\n",
             "| model | ROC-AUC | PR-AUC | TPR@0.1%FPR | TPR@0.01%FPR |",
             "|---|---|---|---|---|"]

    def cell(st):
        return f"{st['mean']:.4f} ± {st['std']:.4f}" if st else "n/a"
    labels = {"nograph_mlp": "no-graph MLP", "centralized_gnn": "centralized GNN",
              "fed_edges": "fed (edges)", "fed_positives": "fed (positives)",
              "fed_valsignal": "fed (val_signal)"}
    for mod in models:
        row = [labels[mod]] + [cell(multiseed[m][mod]) for m in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## pos_smooth sweep (seed 42, positive-aware)\n",
              "| pos_smooth | ROC-AUC | PR-AUC | TPR@0.1%FPR |", "|---|---|---|---|"]
    for r in sweep:
        def f(x): return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"
        lines.append(f"| {r['pos_smooth']} | {f(r['roc_auc'])} | {f(r['pr_auc'])} | {f(r['tpr@fpr=0.001'])} |")
    md = "\n".join(lines) + "\n"
    with open(os.path.join(HERE, "results", "lanl_experiment_summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("\n" + md)
    print("Saved results/lanl_experiment_summary.{json,md}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate_only", action="store_true")
    a = ap.parse_args()
    os.makedirs(EXP, exist_ok=True)
    if not a.aggregate_only:
        matrix = build_matrix()
        print(f"Experiment matrix: {len(matrix)} runs (resumable). Seeds={SEEDS}\n")
        t0 = time.time()
        for i, spec in enumerate(matrix, 1):
            print(f"[{i}/{len(matrix)}] {spec['tag']}", flush=True)
            launch(spec)
        print(f"\nMatrix finished in {int((time.time()-t0)/60)} min.")
    aggregate()
