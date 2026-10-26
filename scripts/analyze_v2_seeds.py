from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2 as A
import analyze_v2_extra as X

SETTINGS = [("gsm8k", "Qwen3-32B"), ("mmlu_pro", "Qwen3-8B")]
POLICIES = ["Fixed", "LearnStop", "LS+Cap", "Static|LS", "BestScalar"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", default="results/v2,results/v2_seed1,results/v2_seed2")
    ap.add_argument("--out", default="results/v2_analysis_seeds")
    ap.add_argument("--splits", type=int, default=50)
    args = ap.parse_args()
    A.STATIC_GRID = "ckpt"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    recs = []
    for seed, root in enumerate(args.roots.split(",")):
        runs = A.load_runs(Path(root))
        for task, model in SETTINGS:
            if (task, model, "eval") not in runs or (task, model, "train") not in runs:
                print(f"  [skip] {root}: {task}/{model} not found")
                continue
            P = A.prepare(runs[(task, model, "eval")], "line", 48)
            Tr = A.prepare(runs[(task, model, "train")], "line", 48)
            C, divs = X.build(P, Tr, with_features=False)
            rr = X.sweep(P, C, POLICIES, [0.05, 0.15], [0.0, 0.025], X.LTT if hasattr(X, "LTT") else [("binom", "seq")],
                         divs, args.splits, 0.5, 0.05, "seed")
            for r in rr:
                r.update(setting=f"{task}/{model}", seed=seed, acc_full=float(P["F"].mean()),
                         mean_think=float(P["think"].mean()))
            recs += rr
            print(f"  seed {seed} {task}/{model}: full acc {P['F'].mean():.3f}, mean think {P['think'].mean():.0f}")
    df = pd.DataFrame(recs)
    if df.empty:
        print("no seed runs found"); return
    df.to_csv(out / "seed_records.csv.gz", index=False)
    agg = df.groupby(["setting", "seed", "alpha", "price", "policy"]).agg(
        save_mean=("save", "mean"), risk_mean=("risk", "mean"), acc_mean=("acc", "mean"),
        acc_full=("acc_full", "first"), mean_think=("mean_think", "first")).reset_index()
    agg.to_csv(out / "seed_summary.csv", index=False)
    print(agg[(agg.price == 0) & np.isclose(agg.alpha, 0.05)].pivot_table(
        index=["setting", "seed"], columns="policy", values="save_mean").round(1))


if __name__ == "__main__":
    main()
