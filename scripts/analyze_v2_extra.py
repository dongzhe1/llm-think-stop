from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v2 as A

PRICES = [0.0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
SCHEDULES = {"S14": None, "S7": [0, 256, 512, 1024, 2048, 4096, 8192], "S4": [0, 512, 2048, 8192],
             "S2": [1024, 4096]}
FEAT_GROUPS = {"budget": [0, 1], "confidence": [2, 3], "stability": [4, 5, 6], "markers": [7]}
CAPS = {512, 1024, 2048, 4096, 8192}
SKIP = ("adaptthink", "bbh_free")
ALPHA_GRID = [round(0.01 * i, 2) for i in range(1, 26)]
TRAIN_SIZES = [50, 100, 200]


def pvalues(k: np.ndarray, n: int, alpha: float, bound: str) -> np.ndarray:
    if bound == "binom":
        return A.binom_cdf(k, n, alpha)
    rhat = k / n
    return np.where(rhat < alpha, np.exp(-2.0 * n * (alpha - rhat) ** 2), 1.0)


def certified(C, idx, k_cal, n, alpha, delta, bound, mult, div=1):
    p = pvalues(k_cal[idx], n, alpha, bound)
    if mult == "union":
        return p <= delta / (len(idx) * div)
    fams = np.unique(C.fam[idx])
    ok = p <= delta / (len(fams) * div)
    out = np.zeros(len(idx), dtype=bool)
    for f in fams:
        sel = np.where(C.fam[idx] == f)[0]
        sel = sel[np.argsort(C.order[idx][sel])]
        out[sel] = np.cumprod(ok[sel]).astype(bool)
    return out


def run_subset(run: dict, budgets_keep):
    if budgets_keep is None:
        return run
    b = list(run["budgets"])
    cols = [b.index(x) for x in budgets_keep if x in b]
    m = len(b)
    colsE = cols + [m]
    r = dict(run)
    r["budgets"] = run["budgets"][cols]
    r["active"] = run["active"][:, cols]
    for k in ("txt_line", "txt_full", "len_line", "len_full", "mkr"):
        r[k] = run[k][:, colsE]
    for k in ("lp", "ent"):
        r[k] = run[k][:, colsE, :]
    return r


def train_rows_target(Tr, target=None):
    return A.train_rows([Tr], target)


def ls_scores(P, Tr, cols=None, kind="lr", target=None):
    S = np.full((P["N"], P["m"]), -np.inf); St = np.full((Tr["N"], Tr["m"]), -np.inf)
    if not Tr["m"] or not Tr["active"].any():
        return S, np.array([])
    X, y = train_rows_target(Tr, target)
    if len(np.unique(y)) < 2:
        return S, np.array([])
    use = list(range(X.shape[1])) if cols is None else cols
    fn = A.fit(kind, X[:, use], y)
    if P["active"].any():
        S[P["active"]] = fn(P["X"][P["active"]][:, use])
    if Tr["active"].any():
        St[Tr["active"]] = fn(Tr["X"][Tr["active"]][:, use])
    return S, A.grid_from(St)


def build(P, Tr, with_features=True, with_scalars=True):
    C = A.Cands(P)
    m = P["m"]
    budgets = [int(x) for x in P["budgets"]]
    sb = [int(x) for x in P["st_budgets"]]
    C.off("Fixed", 0)
    for j in range(len(sb) - 1, -1, -1):
        C.static("Fixed", 0, j)
    for g in ("Fixed-caps", "Fixed-caps-d6"):
        C.off(g, 0)
        for j in range(len(sb) - 1, -1, -1):
            if sb[j] in CAPS:
                C.static(g, 0, j)
    S, grid = ls_scores(P, Tr)
    C.off("LearnStop", 0); C.thresholds("LearnStop", 0, S, grid, name="LS")
    C.off("LS+Cap", 0); C.thresholds("LS+Cap", 0, S, grid, name="LS")
    for c in [j for j in range(m) if budgets[j] in CAPS]:
        C.fixed("LS+Cap", c + 1, c); C.thresholds("LS+Cap", c + 1, S, grid, cap=c, name="LS")
    C.off("Static|LS", 0)
    for j in range(len(sb) - 1, -1, -1):
        C.static("Static|LS", 0, j)
    C.off("Static|LS", 1); C.thresholds("Static|LS", 1, S, grid, name="LS")
    if with_scalars:
        for f, (pol, key) in enumerate(A.SCALARS.items()):
            src = Tr[f"S_{key}"] if Tr["m"] else P[f"S_{key}"]
            C.off("BestScalar", f); C.thresholds("BestScalar", f, P[f"S_{key}"], A.grid_from(src), name=pol)
    if with_features:
        allc = list(range(len(A.FEATURES)))
        for gname, gcols in FEAT_GROUPS.items():
            for mode, cols in ((f"LS-no-{gname}", [c for c in allc if c not in gcols]), (f"LS-only-{gname}", gcols)):
                Sv, gv = ls_scores(P, Tr, cols)
                C.off(mode, 0); C.thresholds(mode, 0, Sv, gv, name=mode)
    return C.finalize(), {"Fixed-caps-d6": 6}


def sweep(P, C, groups, alphas, prices, certs, divs, n_splits, cal_frac, delta, tag):
    N = P["N"]; n_cal = int(round(N * cal_frac)); ref = P["ref_dec"]
    recs = []
    gidx = {g: np.where(C.group == g)[0] for g in groups}
    offs = {g: int(gidx[g][np.array([C.label[t] == "off" for t in gidx[g]])][0]) for g in groups}
    for sp in range(n_splits):
        perm = np.random.default_rng(sp).permutation(N)
        cal, te = perm[:n_cal], perm[n_cal:]
        k = C.loss[:, cal].sum(1)
        dcal, ecal = C.dec[:, cal].mean(1), C.ext[:, cal].mean(1)
        dte, ete = C.dec[:, te].mean(1), C.ext[:, te].mean(1)
        risk, acc = C.loss[:, te].mean(1), C.corr[:, te].mean(1)
        rte = ref[te].mean()
        for a in alphas:
            for (bound, mult) in certs:
                for g in groups:
                    idx = gidx[g]
                    mask = certified(C, idx, k, n_cal, a, delta, bound, mult, divs.get(g, 1))
                    valid = idx[mask]
                    cert_any = bool(any(C.label[t] != "off" for t in valid))
                    for p in prices:
                        if valid.size:
                            c = int(valid[np.argmin(dcal[valid] + p * ecal[valid])])
                        else:
                            c = offs[g]
                        recs.append(dict(tag=tag, split=sp, alpha=a, cert=f"{bound}+{mult}", policy=g, price=p,
                                         save=100 * (1 - (dte[c] + p * ete[c]) / rte), risk=risk[c], acc=acc[c],
                                         feasible=C.label[c] != "off", cert_any=cert_any))
    return recs


def bootstrap(P, C, pairs, alphas, delta, cal_frac, n_boot=2000, split=0):
    N = P["N"]; n_cal = int(round(N * cal_frac)); ref = P["ref_dec"]
    perm = np.random.default_rng(split).permutation(N)
    cal, te = perm[:n_cal], perm[n_cal:]
    k = C.loss[:, cal].sum(1); dcal = C.dec[:, cal].mean(1)
    rng = np.random.default_rng(1000 + split)
    bi = te[rng.integers(0, len(te), size=(n_boot, len(te)))]
    out = []
    for a in alphas:
        pick = {}
        for g in set(sum(pairs, ())):
            idx = np.where(C.group == g)[0]
            valid = idx[certified(C, idx, k, n_cal, a, delta, "binom", "seq")]
            pick[g] = int(valid[np.argmin(dcal[valid])]) if valid.size else \
                int(idx[np.array([C.label[t] == "off" for t in idx])][0])
        for ga, gb in pairs:
            r = ref[bi].mean(1)
            d = 100 * (C.dec[pick[gb]][bi].mean(1) - C.dec[pick[ga]][bi].mean(1)) / r
            point = 100 * (C.dec[pick[gb]][te].mean() - C.dec[pick[ga]][te].mean()) / ref[te].mean()
            out.append(dict(alpha=a, A=ga, B=gb, split=split, delta_point=point, ci_lo=float(np.percentile(d, 2.5)),
                            ci_hi=float(np.percentile(d, 97.5)), n_test=len(te)))
    return out


def merge(parts):
    rows = {k: [] for k in ("loss", "corr", "dec", "ext", "group", "fam", "label", "order")}
    for C, g, ng, nf in parts:
        idx = np.where(C.group == g)[0]
        assert len(np.unique(C.fam[idx])) == 1, g
        rows["loss"].append(C.loss[idx]); rows["corr"].append(C.corr[idx])
        rows["dec"].append(C.dec[idx]); rows["ext"].append(C.ext[idx])
        rows["group"] += [ng] * len(idx); rows["fam"] += [nf] * len(idx)
        rows["label"] += [C.label[t] for t in idx]
        rows["order"].append(C.order[idx])
    M = type("Merged", (), {})()
    M.loss = np.concatenate(rows["loss"]); M.corr = np.concatenate(rows["corr"])
    M.dec = np.concatenate(rows["dec"]); M.ext = np.concatenate(rows["ext"])
    M.group = np.array(rows["group"]); M.fam = np.array(rows["fam"]); M.label = rows["label"]
    M.order = np.concatenate(rows["order"])
    return M


def subset_rows(P: dict, idx) -> dict:
    N = P["N"]
    out = {k: (v[idx] if isinstance(v, np.ndarray) and k != "budgets" and v.ndim >= 1 and v.shape[0] == N else v)
           for k, v in P.items()}
    out["N"] = len(idx)
    return out


def collection_cost(Tr) -> np.ndarray:
    m = Tr["m"]
    return Tr["think"] + (Tr["L"][:, :m] * Tr["active"]).sum(1) + Tr["L"][:, m]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v2")
    ap.add_argument("--out", default="results/v2_analysis_extra")
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--train_curve", action="store_true",
                    help="only retrain LearnStop on subsets of the training pool and write train_curve.csv")
    ap.add_argument("--train_reps", type=int, default=5)
    ap.add_argument("--fine_static", action="store_true",
                    help="only compare the static budget on the fine grid (fine_probes.npz from probe_fine_grid.py) "
                         "with the 14-point static budget and LearnStop; writes fine_summary.csv")
    ap.add_argument("--ls_target", choices=["safe", "correct"], default="safe", help="LearnStop training label")
    ap.add_argument("--static_grid", choices=["fine", "ckpt"], default="fine", help="static-budget grid")
    ap.add_argument("--label_check", action="store_true",
                    help="only compare LearnStop trained on the probe-correct label with LearnStop trained on the "
                         "certified loss (safe-to-stop label), and split each policy's saving by full-thinking "
                         "correctness; writes label_check.csv and label_split.csv")
    ap.add_argument("--natural_curves", default="",
                    help="comma-separated model tags: only compute alpha curves of the main policies with the free "
                         "response as the reference (for the AdaptThink comparison) and write natural_curves.csv")
    args = ap.parse_args()
    A.LS_TARGET, A.STATIC_GRID = args.ls_target, args.static_grid
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    runs = A.load_runs(Path(args.root))
    evals = sorted(k for k in runs if k[2] == "eval" and not any(s in (k[0] + k[1]).lower() for s in SKIP))
    LTT = [("binom", "seq")]
    if args.natural_curves:
        want = set(args.natural_curves.split(","))
        recs = []
        for task, model, _ in evals:
            if model not in want or (task, model, "train") not in runs:
                continue
            P = A.prepare(runs[(task, model, "eval")], "line", 48, ref="natural")
            Tr = A.prepare(runs[(task, model, "train")], "line", 48, ref="natural")
            C, divs = build(P, Tr, with_features=False, with_scalars=False)
            rr = sweep(P, C, ["Fixed", "LearnStop", "LS+Cap", "Static|LS"], [0.05, 0.10, 0.15, 0.20], [0.0], LTT,
                       divs, args.splits, args.cal_frac, args.delta, "natural")
            for r in rr:
                r["setting"] = f"{task}/{model}"
            recs += rr
            print(f"  natural curves {task}/{model}", flush=True)
        df = pd.DataFrame(recs)
        df.groupby(["setting", "alpha", "policy"]).agg(save_mean=("save", "mean"), acc_mean=("acc", "mean"),
                                                        risk_mean=("risk", "mean")).reset_index() \
          .to_csv(out / "natural_curves.csv", index=False)
        print(f"[extra] natural curves -> {out / 'natural_curves.csv'}")
        return
    if args.fine_static:
        recs, agree = [], {}
        for task, model, _ in evals:
            if (task, model, "train") not in runs:
                continue
            sid = f"{task}/{model}"
            P = A.prepare(runs[(task, model, "eval")], "line", 48)
            if len(P["st_budgets"]) == P["m"]:
                print(f"  [skip] {sid}: no fine grid"); continue
            Tr = A.prepare(runs[(task, model, "train")], "line", 48)
            run = runs[(task, model, "eval")]; fz = run["_fine"]; fb = [int(x) for x in fz["budgets"]]
            m0 = run["active"].shape[1]; r_ = {k: v for k, v in run.items() if k != "_fine"}
            r_["budgets"] = fz["budgets"]; r_["active"] = fz["active"]
            for k in ("txt_line", "txt_full", "len_line", "len_full"):
                r_[k] = np.concatenate([fz[k], run[k][:, m0:m0 + 1]], axis=1)
            r_["mkr"] = np.zeros((P["N"], len(fb) + 1)); A_ = run["lp"].shape[2]
            r_["lp"] = np.full((P["N"], len(fb) + 1, A_), np.nan); r_["ent"] = np.full((P["N"], len(fb) + 1, A_), np.nan)
            sg, A.STATIC_GRID = A.STATIC_GRID, "ckpt"
            Pr = A.prepare(r_, "line", 48)
            A.STATIC_GRID = sg
            same = tot = 0
            for jo, B in enumerate(int(x) for x in run["budgets"]):
                if B in fb:
                    a = P["active"][:, jo]
                    same += int((P["correct"][a, jo] == Pr["correct"][a, fb.index(B)]).sum()); tot += int(a.sum())
            agree[sid] = same / max(tot, 1)
            C, _ = build(P, Tr, with_features=False, with_scalars=False)
            C14 = A.Cands(P)
            C14.off("Fixed-14", 0)
            for jj in range(P["m"] - 1, -1, -1):
                C14.fixed("Fixed-14", 0, jj)
            C14.finalize()
            M = merge([(C14, "Fixed-14", "Fixed-14", 0), (C, "Fixed", "Fixed", 0), (C, "LearnStop", "LearnStop", 0)])
            groups = ["Fixed-14", "Fixed", "LearnStop"]
            rr = sweep(P, M, groups, sorted(set(ALPHA_GRID) | {0.05, 0.10, 0.15, 0.20}), [0.0], LTT, {},
                       args.splits, args.cal_frac, args.delta, "fine")
            rr += sweep(P, M, groups, [0.05], PRICES, LTT, {}, args.splits, args.cal_frac, args.delta, "fineprice")
            for r in rr:
                r["setting"] = sid
            recs += rr
            print(f"  fine grid {sid}: {len(P['st_budgets'])} static budgets ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(recs)
        df.groupby(["setting", "tag", "alpha", "policy", "price"]).agg(
            save_mean=("save", "mean"), risk_mean=("risk", "mean"), acc_mean=("acc", "mean"),
            feasible_rate=("feasible", "mean"), cert_rate=("cert_any", "mean")).reset_index() \
          .to_csv(out / "fine_summary.csv", index=False)
        json.dump(agree, open(out / "fine_agreement.json", "w"), indent=1)
        print(f"[extra] fine grid -> {out / 'fine_summary.csv'} ({time.time() - t0:.0f}s)")
        return
    if args.label_check:
        recs, splits_ = [], []
        for task, model, _ in evals:
            if (task, model, "train") not in runs:
                continue
            sid = f"{task}/{model}"
            P = A.prepare(runs[(task, model, "eval")], "line", 48)
            Tr = A.prepare(runs[(task, model, "train")], "line", 48)
            C = A.Cands(P)
            C.off("Fixed", 0)
            for j in range(len(P["st_budgets"]) - 1, -1, -1):
                C.static("Fixed", 0, j)
            for g, kind, target in (("LS-correct", "lr", "correct"), ("LearnStop", "lr", "safe"),
                                    ("LS-correct-GBM", "gbm", "correct"), ("LS-safe-GBM", "gbm", "safe")):
                S, grid = ls_scores(P, Tr, kind=kind, target=target)
                C.off(g, 0); C.thresholds(g, 0, S, grid, name=g)
            C.finalize()
            groups = ["Fixed", "LearnStop", "LS-correct", "LS-correct-GBM", "LS-safe-GBM"]
            rr = sweep(P, C, groups, [0.05, 0.15], [0.0], LTT, {}, args.splits, args.cal_frac, args.delta, "label")
            for r in rr:
                r["setting"] = sid
            recs += rr
            N = P["N"]; n_cal = int(round(N * args.cal_frac)); ref = P["ref_dec"]; F0 = P["F"] == 0
            for sp in range(args.splits):
                perm = np.random.default_rng(sp).permutation(N)
                cal, te = perm[:n_cal], perm[n_cal:]
                k = C.loss[:, cal].sum(1); dcal = C.dec[:, cal].mean(1)
                for g in groups:
                    idx = np.where(C.group == g)[0]
                    valid = idx[certified(C, idx, k, n_cal, 0.05, args.delta, "binom", "seq")]
                    c = int(valid[np.argmin(dcal[valid])]) if valid.size else idx[0]
                    tot = ref[te].sum()
                    for part, mask in (("F0", F0[te]), ("F1", ~F0[te])):
                        q = te[mask]
                        splits_.append(dict(setting=sid, split=sp, policy=g, part=part, share=float(mask.mean()),
                                            save=100 * float((ref[q] - C.dec[c, q]).sum() / tot)))
            print(f"  label check {sid} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(recs)
        df.groupby(["setting", "alpha", "policy"]).agg(save_mean=("save", "mean"), risk_mean=("risk", "mean"),
                                                        acc_mean=("acc", "mean"), feasible_rate=("feasible", "mean")) \
          .reset_index().to_csv(out / "label_check.csv", index=False)
        pd.DataFrame(splits_).groupby(["setting", "policy", "part"]).agg(save=("save", "mean"), share=("share", "mean")) \
          .reset_index().to_csv(out / "label_split.csv", index=False)
        print(f"[extra] label check -> {out / 'label_check.csv'} ({time.time() - t0:.0f}s)")
        return
    if args.train_curve:
        recs = []
        for task, model, _ in evals:
            if (task, model, "train") not in runs:
                continue
            sid = f"{task}/{model}"
            P = A.prepare(runs[(task, model, "eval")], "line", 48)
            Tfull = A.prepare(runs[(task, model, "train")], "line", 48)
            cost_all = collection_cost(Tfull)
            nT = Tfull["N"]
            plan = [(n, r) for n in TRAIN_SIZES if n < nT for r in range(args.train_reps)] + [(nT, 0)]
            for n, r in plan:
                idx = np.arange(nT) if n == nT else np.sort(np.random.default_rng(7919 * n + r).choice(nT, n, replace=False))
                Tr = subset_rows(Tfull, idx)
                C, divs = build(P, Tr, with_features=False, with_scalars=False)
                rr = sweep(P, C, ["Fixed", "LearnStop"], [0.05, 0.15], [0.0], LTT, divs, args.splits, args.cal_frac,
                           args.delta, "train")
                for x in rr:
                    x.update(setting=sid, n_train=n, rep=r, collect_tokens=float(cost_all[idx].sum()),
                             collect_trace=float(Tfull["think"][idx].sum()), ref_mean=float(P["ref_dec"].mean()))
                recs += rr
            print(f"  train curve {sid} ({time.time() - t0:.0f}s)", flush=True)
        df = pd.DataFrame(recs)
        df.groupby(["setting", "alpha", "policy", "n_train", "rep"]).agg(
            save_mean=("save", "mean"), risk_mean=("risk", "mean"), feasible_rate=("feasible", "mean"),
            collect_tokens=("collect_tokens", "first"), collect_trace=("collect_trace", "first"),
            ref_mean=("ref_mean", "first")).reset_index().to_csv(out / "train_curve.csv", index=False)
        print(f"[extra] training-size curve -> {out / 'train_curve.csv'} ({time.time() - t0:.0f}s)")
        return
    CERT4 = [("binom", "seq"), ("binom", "union"), ("hoeff", "seq"), ("hoeff", "union")]
    recs, boots = [], []
    for task, model, _ in evals:
        sid = f"{task}/{model}"
        if (task, model, "train") not in runs:
            continue
        P = A.prepare(runs[(task, model, "eval")], "line", 48)
        Tr = A.prepare(runs[(task, model, "train")], "line", 48)
        C, divs = build(P, Tr)
        base = ["Fixed", "LearnStop", "LS+Cap", "Static|LS", "BestScalar"]
        r1 = sweep(P, C, base, [0.05, 0.15], PRICES, LTT, divs, args.splits, args.cal_frac, args.delta, "price")
        r2 = sweep(P, C, ["Fixed-caps", "Fixed-caps-d6"], [0.05, 0.15], [0.0], LTT, divs, args.splits,
                   args.cal_frac, args.delta, "decomp")
        feats = [g for g in set(C.group.tolist()) if g.startswith("LS-")]
        r3 = sweep(P, C, feats, [0.05], [0.0], LTT, divs, args.splits, args.cal_frac, args.delta, "features")
        r4 = sweep(P, C, ["Fixed", "LearnStop", "BestScalar"], [0.05, 0.10, 0.15, 0.20], [0.0], CERT4, divs,
                   args.splits, args.cal_frac, args.delta, "cert")
        r6 = sweep(P, C, ["Fixed", "LearnStop", "Static|LS"], ALPHA_GRID, [0.0], LTT, divs, args.splits,
                   args.cal_frac, args.delta, "riskcurve")
        rr = r1 + r2 + r3 + r4 + r6
        for sp in range(args.splits):
            pairs = [("LearnStop", "Fixed"), ("Static|LS", "Fixed"), ("Static|LS", "LearnStop")] if sp == 0 else \
                [("LearnStop", "Fixed")]
            for b in bootstrap(P, C, pairs, [0.05, 0.15] if sp == 0 else [0.05], args.delta, args.cal_frac, split=sp):
                b["setting"] = sid
                boots.append(b)
        sched_parts = [(C, "LearnStop", "LS-sched", 0)]
        for sname, keep in SCHEDULES.items():
            if keep is None:
                continue
            Ps = A.prepare(run_subset(runs[(task, model, "eval")], keep), "line", 48)
            Ts = A.prepare(run_subset(runs[(task, model, "train")], keep), "line", 48)
            Cs, dv = build(Ps, Ts, with_features=False, with_scalars=False)
            r5 = sweep(Ps, Cs, ["LearnStop", "Static|LS"], [0.05], PRICES, LTT, dv, args.splits, args.cal_frac,
                       args.delta, f"schedule:{sname}")
            rr += r5
            sched_parts.append((Cs, "LearnStop", "LS-sched", len(sched_parts)))
        M = merge([(C, "Fixed", "Fixed", 0)] + sched_parts +
                  [(C, "Fixed", "Static|LS-sched", 0)] + [(c_, g_, "Static|LS-sched", f_ + 1)
                                                          for c_, g_, _, f_ in sched_parts])
        rr += sweep(P, M, ["Fixed", "LS-sched", "Static|LS-sched"], [0.05], PRICES, LTT, {}, args.splits,
                    args.cal_frac, args.delta, "schedsel")
        for r in rr:
            r["setting"] = sid
        recs += rr
        print(f"  {sid}: {len(rr)} records ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(recs)
    df["viol"] = (df.risk > df.alpha).astype(float)
    df.to_csv(out / "extra_records.csv.gz", index=False)
    agg = df.groupby(["setting", "tag", "alpha", "cert", "policy", "price"]).agg(
        save_mean=("save", "mean"), risk_mean=("risk", "mean"), acc_mean=("acc", "mean"),
        feasible_rate=("feasible", "mean"), cert_rate=("cert_any", "mean"), risk_viol=("viol", "mean")).reset_index()
    agg.to_csv(out / "extra_summary.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
    print(f"[extra] done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
