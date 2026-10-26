from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grading_v2 as G

FEATURES = ["log_budget", "ckpt_frac", "conf_lp", "conf_ent", "stable", "run_len", "vote_share", "marker_density"]
SCALARS = {"Confidence": "conf", "Entropy": "ent", "AnswerConv": "run", "ConfLeap": "leap"}
QUANT_CATS = {"math", "physics", "chemistry", "engineering"}
REGIMES = ("kv_fork", "prefix_cache", "black_box")
MAIN = dict(acct="line", regime="kv_fork", alpha=0.05)
LS_TARGET = "safe"
STATIC_GRID = "fine"


def load_runs(root: Path):
    runs = {}
    for d in sorted(root.glob("*__*")):
        if not (d / "raw.npz").exists() or not (d / "meta.json").exists():
            continue
        meta = json.load(open(d / "meta.json"))
        z = np.load(d / "raw.npz", allow_pickle=False)
        run = {k: z[k] for k in z.files}
        run["meta"] = meta
        if (d / "fine_probes.npz").exists():
            zf = np.load(d / "fine_probes.npz", allow_pickle=False)
            run["_fine"] = {k: zf[k] for k in zf.files}
        split = meta.get("split") or d.name.split("__")[0].rsplit("_", 1)[-1]
        runs[(meta["task"], meta["model_tag"], split)] = run
    return runs


def gold_alternates(run: dict) -> list[tuple]:
    N = len(run["gold"])
    alts = [()] * N
    cats = [str(c) for c in run["category"]]
    if "dyck_languages" not in cats:
        return alts
    path = Path(run["meta"].get("data", ""))
    if not path.exists():
        print(f"  [warn] {path} not found: dyck answers graded against the closing brackets only")
        return alts
    q = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            q[r["id"]] = r["question"]
    for i, (qid, c) in enumerate(zip(run["qid"], cats)):
        if c == "dyck_languages" and str(qid) in q:
            inp = q[str(qid)].split("Input:")[-1].strip()
            alts[i] = (inp + " " + str(run["gold"][i]),)
    return alts


def prepare(run: dict, acct: str, ans_cap: int, ref: str = "probe") -> dict:
    budgets = run["budgets"].astype(int)
    N, m = len(run["think_len"]), len(budgets)
    active = run["active"].astype(bool).reshape(N, m)
    txt = run["txt_line"] if acct == "line" else run["txt_full"]
    Lgen = (run["len_line"] if acct == "line" else run["len_full"]).astype(float)
    gold, mode, nopt = run["gold"], run["mode"], run["n_options"]
    alts = gold_alternates(run)
    keys = np.full((N, m + 1), "", dtype=object)
    correct = np.zeros((N, m + 1), dtype=np.int8)
    for i in range(N):
        for j in range(m + 1):
            if j < m and not active[i, j]:
                continue
            keys[i, j], correct[i, j] = G.grade(str(txt[i, j]), str(gold[i]), str(mode[i]), int(nopt[i]),
                                                alts=alts[i])
    nat = np.array([G.grade(str(run["resp_text"][i]), str(gold[i]), str(mode[i]), int(nopt[i]),
                            where="natural", alts=alts[i])[1] for i in range(N)], dtype=np.int8)
    gen_len = run["gen_len"].astype(float)
    if ref == "natural":
        correct[:, m] = nat
    Lconf = (run["len_line"] if acct == "line" else run["len_full"]).astype(int)
    A = run["lp"].shape[2]
    mask = np.arange(A)[None, None, :] < Lconf[:, :, None]
    lp = np.where(mask, np.nan_to_num(run["lp"], nan=0.0), 0.0).sum(2) / np.maximum(Lconf, 1)
    en = np.where(mask, np.nan_to_num(run["ent"], nan=0.0), 0.0).sum(2) / np.maximum(Lconf, 1)
    L = np.full_like(Lgen, float(ans_cap)) if acct == "cap" else Lgen.copy()
    if ref == "natural":
        L[:, m] = np.maximum(gen_len - run["think_len"].astype(float), 0.0)
    P = dict(N=N, m=m, budgets=budgets, active=active, keys=keys, correct=correct, F=correct[:, m].copy(),
             nat=nat, gen_len=gen_len, conf_lp=lp, conf_ent=en, mkr=run["mkr"].astype(float), L=L,
             think=run["think_len"].astype(float), prompt=run["prompt_len"].astype(float),
             mode=mode, category=run["category"], qid=run["qid"], closed=run["closed"].astype(bool))
    P["X"] = features(P)
    P.update(stop_tables(P))
    P.update(scalar_scores(P))
    if STATIC_GRID == "fine" and "_fine" in run:
        Pf = prepare(fine_run(run, run["_fine"]), acct, ans_cap, ref=ref)
        assert np.allclose(Pf["ref_dec"], P["ref_dec"]) and (Pf["F"] == P["F"]).all()
        P.update(st_budgets=Pf["budgets"], st_fcorr=Pf["fcorr"], st_fdec=Pf["fdec"], st_fext=Pf["fext"])
    else:
        P.update(st_budgets=budgets, st_fcorr=P["fcorr"], st_fdec=P["fdec"], st_fext=P["fext"])
    return P


def fine_run(run: dict, fz) -> dict:
    N = len(run["think_len"]); m = run["active"].shape[1]
    assert len(fz["qid"]) == N and list(fz["qid"]) == list(run["qid"]), "fine grid / run mismatch"
    mf = len(fz["budgets"]); fb = [int(x) for x in fz["budgets"]]
    r = {k: v for k, v in run.items() if k != "_fine"}
    r["budgets"] = fz["budgets"]; r["active"] = fz["active"]
    for k in ("txt_line", "txt_full", "len_line", "len_full"):
        fine = fz[k].copy()
        for jo, B in enumerate(int(x) for x in run["budgets"]):
            if B in fb:
                fine[:, fb.index(B)] = run[k][:, jo]
        r[k] = np.concatenate([fine, run[k][:, m:m + 1]], axis=1)
    r["mkr"] = np.zeros((N, mf + 1), np.float32)
    A_ = run["lp"].shape[2]
    r["lp"] = np.full((N, mf + 1, A_), np.nan, np.float32); r["ent"] = np.full((N, mf + 1, A_), np.nan, np.float32)
    return r


def features(P) -> np.ndarray:
    N, m = P["N"], P["m"]
    X = np.full((N, m, len(FEATURES)), np.nan)
    lb = np.log1p(P["budgets"]) / np.log1p(max(P["budgets"].max(), 1)) if m else np.zeros(0)
    for i in range(N):
        prev, run, counts = None, 0, Counter()
        for j in range(m):
            if not P["active"][i, j]:
                break
            a = P["keys"][i, j]
            same = a != "" and a == prev
            run = run + 1 if same else (1 if a != "" else 0)
            if a != "":
                counts[a] += 1
            X[i, j] = [lb[j], j / max(m - 1, 1), P["conf_lp"][i, j], P["conf_ent"][i, j], float(same), run,
                       counts[a] / (j + 1) if a != "" else 0.0, P["mkr"][i, j]]
            prev = a
    return X


def scalar_scores(P) -> dict:
    X, act = P["X"], P["active"]
    ninf = -np.inf
    conf = np.where(act, np.exp(X[..., 2]), ninf)
    ent = np.where(act, -X[..., 3], ninf)
    run = np.where(act, X[..., 5], ninf)
    c0 = np.where(act, np.exp(X[..., 2]), 0.0)
    prev = np.concatenate([np.zeros((P["N"], 1)), c0[:, :-1]], axis=1) if P["m"] else c0
    leap = np.where(act, np.maximum(c0 - prev, 0.0) * c0, ninf)
    return {"S_conf": conf, "S_ent": ent, "S_run": run, "S_leap": leap}


def stop_tables(P) -> dict:
    N, m, B, act, L = P["N"], P["m"], P["budgets"].astype(float), P["active"], P["L"]
    think, pr = P["think"], P["prompt"]
    ref_dec = think + L[:, m]
    if m == 0:
        return dict(dec=ref_dec[:, None], ext=np.zeros((N, 1)), fdec=np.zeros((N, 0)), fext=np.zeros((N, 0)),
                    fcorr=np.zeros((N, 0)), ref_dec=ref_dec)
    Lact = np.where(act, L[:, :m], 0.0)
    cumL = np.cumsum(Lact, axis=1)
    PB = np.where(act, pr[:, None] + B[None, :], 0.0)
    cumPB = np.cumsum(PB, axis=1)
    dec = np.zeros((N, m + 1)); ext = np.zeros((N, m + 1))
    dec[:, :m] = B[None, :] + cumL
    dec[:, m] = think + cumL[:, -1] + L[:, m]
    ext[:, :m] = cumPB + np.concatenate([np.zeros((N, 1)), cumPB[:, :-1]], axis=1)
    ext[:, m] = 2.0 * cumPB[:, -1]
    fdec = np.where(act, B[None, :] + L[:, :m], (think + L[:, m])[:, None])
    fext = np.where(act, pr[:, None] + B[None, :], 0.0)
    fcorr = np.where(act, P["correct"][:, :m], P["correct"][:, m:m + 1])
    return dict(dec=dec, ext=ext, fdec=fdec, fext=fext, fcorr=fcorr, ref_dec=ref_dec)


def train_rows(Ps: list[dict], target: str | None = None):
    target = target or LS_TARGET
    Xs, ys = [], []
    for P in Ps:
        a, m = P["active"], P["m"]
        y = P["correct"][:, :m].astype(bool)
        if target == "safe":
            y = y | (P["F"][:, None] == 0)
        Xs.append(P["X"][a]); ys.append(y[a].astype(int))
    return np.concatenate(Xs), np.concatenate(ys)


def fit(kind: str, X, y):
    from sklearn.preprocessing import StandardScaler
    X = np.nan_to_num(X)
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X), y)
        return lambda Z: clf.predict_proba(sc.transform(np.nan_to_num(Z)))[:, 1]
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0).fit(X, y)
    return lambda Z: clf.predict_proba(np.nan_to_num(Z))[:, 1]


def score(model_fn, P):
    S = np.full((P["N"], P["m"]), -np.inf)
    a = P["active"]
    if a.any():
        S[a] = model_fn(P["X"][a])
    return S


def grid_from(scores_train: np.ndarray, n_q: int = 100) -> np.ndarray:
    v = scores_train[np.isfinite(scores_train)]
    if v.size == 0:
        return np.array([])
    u = np.unique(v)
    g = u if u.size <= n_q else np.unique(np.quantile(v, np.linspace(0, 1, n_q + 1)))
    return np.sort(g)[::-1]


class Cands:

    def __init__(self, P):
        self.P = P
        self.loss, self.corr, self.dec, self.ext = [], [], [], []
        self.group, self.fam, self.label = [], [], []

    def _add(self, group, fam, label, corr, dec, ext):
        self.loss.append(((self.P["F"] == 1) & (corr == 0)).astype(np.int8))
        self.corr.append(corr.astype(np.int8)); self.dec.append(dec); self.ext.append(ext)
        self.group.append(group); self.fam.append(fam); self.label.append(label)

    def off(self, group, fam):
        P = self.P
        self._add(group, fam, "off", P["F"], P["ref_dec"], np.zeros(P["N"]))

    def fixed(self, group, fam, j):
        P = self.P
        self._add(group, fam, f"fixed@{P['budgets'][j]}", P["fcorr"][:, j], P["fdec"][:, j], P["fext"][:, j])

    def static(self, group, fam, j):
        P = self.P
        self._add(group, fam, f"static@{P['st_budgets'][j]}", P["st_fcorr"][:, j], P["st_fdec"][:, j],
                  P["st_fext"][:, j])

    def thresholds(self, group, fam, S, grid, cap=None, name=""):
        P = self.P
        N, m, ar = P["N"], P["m"], np.arange(P["N"])
        for tau in grid:
            hit = S >= tau
            s = np.where(hit.any(1), hit.argmax(1), m)
            if cap is not None:
                s = np.where(P["active"][:, cap] & (s > cap), cap, s)
            self._add(group, fam, f"{name}>={tau:.4g}" + (f"|cap{P['budgets'][cap]}" if cap is not None else ""),
                      P["correct"][ar, s], P["dec"][ar, s], P["ext"][ar, s])

    def finalize(self):
        self.loss = np.array(self.loss); self.corr = np.array(self.corr)
        self.dec = np.array(self.dec, dtype=np.float64); self.ext = np.array(self.ext, dtype=np.float64)
        self.group = np.array(self.group); self.fam = np.array(self.fam)
        self.order = np.zeros(len(self.fam), dtype=int)
        seen = Counter()
        for k, (g, f) in enumerate(zip(self.group, self.fam)):
            self.order[k] = seen[(g, f)]; seen[(g, f)] += 1
        return self


def build_cands(P, scores: dict, grids: dict, caps=None) -> Cands:
    C = Cands(P)
    m = P["m"]
    cap_idx = [j for j in range(m) if caps is None or int(P["budgets"][j]) in caps]
    C.off("Fixed", 0)
    for j in range(len(P["st_budgets"]) - 1, -1, -1):
        C.static("Fixed", 0, j)
    for pol, key in SCALARS.items():
        S, g = scores[f"S_{key}"], grids[key]
        for grp, fam in ((pol, 0), ("BestScalar", list(SCALARS).index(pol))):
            C.off(grp, fam); C.thresholds(grp, fam, S, g, name=pol)
    for name in [k for k in scores if k.startswith("LS")]:
        grp = {"LS_lr": "LearnStop", "LS_gbm": "LearnStop-GBM", "LS_pooled": "LearnStop-Pooled",
               "LS_loto": "LearnStop-LOTO"}[name]
        C.off(grp, 0); C.thresholds(grp, 0, scores[name], grids[name], name=grp)
    if "LS_lr" in scores:
        C.off("LS+Cap", 0); C.thresholds("LS+Cap", 0, scores["LS_lr"], grids["LS_lr"], name="LS")
        for c in cap_idx:
            C.fixed("LS+Cap", c + 1, c)
            C.thresholds("LS+Cap", c + 1, scores["LS_lr"], grids["LS_lr"], cap=c, name="LS")
    return C.finalize()


def binom_cdf(k: np.ndarray, n: int, p: float) -> np.ndarray:
    try:
        from scipy.stats import binom
        return binom.cdf(k, n, p)
    except ImportError:
        out = np.empty(len(k))
        for t, kk in enumerate(k):
            out[t] = sum(math.comb(n, r) * p ** r * (1 - p) ** (n - r) for r in range(int(kk) + 1))
        return out


def select(C: Cands, group: str, k_cal: np.ndarray, n_cal: int, cost_cal: np.ndarray, alpha: float,
           delta: float, cert: str):
    idx = np.where(C.group == group)[0]
    if idx.size == 0:
        return None
    if cert == "hoeffding":
        ucb = k_cal[idx] / n_cal + math.sqrt(math.log(idx.size / delta) / (2 * n_cal))
        valid = idx[ucb <= alpha]
    else:
        fams = np.unique(C.fam[idx])
        pv = binom_cdf(k_cal[idx], n_cal, alpha)
        ok = pv <= delta / len(fams)
        valid = []
        for f in fams:
            sel = np.where(C.fam[idx] == f)[0]
            sel = sel[np.argsort(C.order[idx][sel])]
            passed = np.cumprod(ok[sel]).astype(bool)
            valid.extend(idx[sel[passed]].tolist())
        valid = np.array(valid, dtype=int)
    if valid.size == 0:
        return int(idx[np.array([C.label[t] == "off" for t in idx])][0])
    return int(valid[np.argmin(cost_cal[valid])])


def regime_cost(C_dec, C_ext, regime, w, disc):
    if regime == "kv_fork":
        return C_dec
    return C_dec + w * (disc if regime == "prefix_cache" else 1.0) * C_ext


def run_splits(P, C: Cands, subset, groups, alphas, regimes, n_splits, cal_frac, delta, cert, w, disc,
               seed0=0, keep_choices=False):
    subset = np.asarray(subset)
    n_cal = int(round(len(subset) * cal_frac))
    ref = P["ref_dec"]
    recs, choices = [], {}
    costs = {r: regime_cost(C.dec, C.ext, r, w, disc) for r in regimes}
    for sp in range(n_splits):
        perm = np.random.default_rng(seed0 + sp).permutation(subset)
        cal, te = perm[:n_cal], perm[n_cal:]
        k_cal = C.loss[:, cal].sum(1)
        risk_te = C.loss[:, te].mean(1); acc_te = C.corr[:, te].mean(1)
        ref_te = ref[te].mean()
        for r in regimes:
            cost_cal = costs[r][:, cal].mean(1)
            save_te = 100.0 * (1.0 - costs[r][:, te].mean(1) / ref_te)
            for a in alphas:
                for g in groups:
                    c = select(C, g, k_cal, n_cal, cost_cal, a, delta, cert)
                    if c is None:
                        continue
                    recs.append(dict(split=sp, regime=r, alpha=a, policy=g, cand=C.label[c],
                                     feasible=C.label[c] != "off", save=save_te[c], risk=risk_te[c],
                                     acc=acc_te[c], acc_full=P["F"][te].mean(), acc_nat=P["nat"][te].mean()))
                    if keep_choices:
                        choices[(sp, r, a, g)] = (c, te)
    return recs, choices


def summarise(recs, keys):
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    df["viol"] = (df["risk"] > df["alpha"]).astype(float)
    return df.groupby(keys + ["regime", "alpha", "policy"]).agg(
        save_mean=("save", "mean"), save_sd=("save", "std"), risk_mean=("risk", "mean"),
        risk_viol=("viol", "mean"), acc_mean=("acc", "mean"), acc_full=("acc_full", "mean"),
        acc_nat=("acc_nat", "mean"), feasible_rate=("feasible", "mean"),
        top_cand=("cand", lambda s: s.value_counts().index[0]), n_splits=("split", "nunique")).reset_index()


def paired_deltas(recs, keys, pairs):
    df = pd.DataFrame(recs)
    out = []
    if df.empty:
        return pd.DataFrame(out)
    piv = df.pivot_table(index=keys + ["regime", "alpha", "split"], columns="policy", values="save")
    for a_, b_ in pairs:
        if a_ not in piv or b_ not in piv:
            continue
        d = (piv[a_] - piv[b_]).dropna()
        for k, g in d.groupby(level=list(range(len(keys) + 2))):
            out.append(dict(zip(keys + ["regime", "alpha"], k), A=a_, B=b_, delta_mean=g.mean(),
                            delta_p025=g.quantile(0.025), delta_p975=g.quantile(0.975), win_rate=(g > 0).mean(),
                            n=len(g)))
    return pd.DataFrame(out)


def bootstrap_split0(P, C, choices, regime, alpha, a_, b_, w, disc, n_boot=2000, seed=0):
    if (0, regime, alpha, a_) not in choices or (0, regime, alpha, b_) not in choices:
        return None
    ca, te = choices[(0, regime, alpha, a_)]
    cb, _ = choices[(0, regime, alpha, b_)]
    cost = regime_cost(C.dec[[ca, cb]], C.ext[[ca, cb]], regime, w, disc)
    ref = P["ref_dec"]
    rng = np.random.default_rng(seed)
    bi = te[rng.integers(0, len(te), size=(n_boot, len(te)))]
    r = ref[bi].mean(1)
    d = 100 * ((1 - cost[0][bi].mean(1) / r) - (1 - cost[1][bi].mean(1) / r))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def trajectory(P, idx=None):
    idx = np.arange(P["N"]) if idx is None else idx
    cnt, settle = Counter(), []
    for i in idx:
        act = [j for j in range(P["m"]) if P["active"][i, j]]
        seq = [int(P["correct"][i, j]) for j in act] + [int(P["F"][i])]
        flips = sum(seq[k] != seq[k - 1] for k in range(1, len(seq)))
        if P["F"][i] == 1:
            t = "early" if seq[0] == 1 else ("benef" if flips <= 1 else "oscill")
            s = P["think"][i]
            for j in reversed(act):
                if not P["correct"][i, j]:
                    break
                s = P["budgets"][j]
            settle.append(float(s))
        else:
            t = "harmful" if any(seq) else "unsolved"
        cnt[t] += 1
    n = max(len(idx), 1)
    out = {k: 100.0 * cnt[k] / n for k in ("early", "benef", "harmful", "unsolved", "oscill")}
    out["settle_med"] = float(np.median(settle)) if settle else float("nan")
    out["settle_p90"] = float(np.percentile(settle, 90)) if settle else float("nan")
    return out


def setting_stats(P, run):
    act = P["active"]
    lines = run["len_line"][:, :P["m"]][act] if P["m"] else np.array([0])
    fulls = run["len_full"][:, :P["m"]][act] if P["m"] else np.array([0])
    probe_tot = P["L"][:, :P["m"]][act].sum() / P["N"] if P["m"] else 0.0
    return dict(N=P["N"], acc_natural=P["nat"].mean(), acc_full=P["F"].mean(), mean_think=P["think"].mean(),
                median_think=float(np.median(P["think"])), unclosed=1 - P["closed"].mean(),
                mean_active_ckpts=act.sum(1).mean() if P["m"] else 0.0,
                probe_len_line_mean=float(np.mean(lines)), probe_len_line_p95=float(np.percentile(lines, 95)),
                probe_len_full_mean=float(np.mean(fulls)),
                frac_probe_hit_cap=float(np.mean(fulls >= run["lp"].shape[2])),
                overhead_ratio=probe_tot / max(P["think"].mean(), 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/v2")
    ap.add_argument("--out", default="results/v2_analysis")
    ap.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--splits", type=int, default=50)
    ap.add_argument("--cal_frac", type=float, default=0.5)
    ap.add_argument("--cal_fracs", default="0.3,0.5,0.7")
    ap.add_argument("--accts", default="line,full,cap")
    ap.add_argument("--cert", choices=["ltt", "hoeffding"], default="ltt")
    ap.add_argument("--prefill_weight", type=float, default=0.25,
                    help="cost of one input token relative to one output token (API list-price ratio)")
    ap.add_argument("--cache_discount", type=float, default=0.1, help="cached-input price / uncached")
    ap.add_argument("--no_gbm", action="store_true")
    ap.add_argument("--ls_target", choices=["safe", "correct"], default="safe", help="LearnStop training label")
    ap.add_argument("--static_grid", choices=["fine", "ckpt"], default="fine", help="static-budget grid")
    ap.add_argument("--caps", default="512,1024,2048,4096,8192",
                    help="hard-cap budgets for LS+Cap (fewer caps = smaller Bonferroni penalty)")
    ap.add_argument("--ref", choices=["probe", "natural"], default="probe",
                    help="full-thinking reference: end probe after the trace (main) or the natural response")
    ap.add_argument("--no_policy", default="adaptthink",
                    help="comma-separated model substrings excluded from the early-exit policy analysis "
                         "(AdaptThink skips thinking on most questions; it enters only the fine-tuned comparison)")
    args = ap.parse_args()
    global LS_TARGET, STATIC_GRID
    LS_TARGET, STATIC_GRID = args.ls_target, args.static_grid
    t_start = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]
    accts = [a for a in args.accts.split(",") if a]
    runs = load_runs(Path(args.root))
    evals = sorted([k for k in runs if k[2] == "eval"])
    print(f"[v2] {len(runs)} runs; eval settings: {[f'{t}/{m}' for t, m, _ in evals]}")

    all_recs, cf_recs, cat_recs, stats_rows, traj_rows, boot_rows = [], [], [], [], [], []
    ft_rows = []
    prepared = {}
    for acct in accts:
        preps = {}
        for key, run in runs.items():
            ans_cap = int(run["meta"].get("ans_cap", 48))
            preps[key] = prepare(run, acct, ans_cap, ref=args.ref)
        prepared[acct] = preps
        for (task, model, _) in evals:
            P = preps[(task, model, "eval")]
            sid = f"{task}/{model}"
            if acct == MAIN["acct"]:
                stats_rows.append(dict(setting=sid, task=task, model=model,
                                       **setting_stats(P, runs[(task, model, "eval")])))
                traj_rows.append(dict(setting=sid, task=task, model=model, **trajectory(P)))
            if P["m"] == 0 or any(x and x in model.lower() for x in args.no_policy.lower().split(",")):
                continue
            Tr = preps.get((task, model, "train"))
            scores = {k: v for k, v in P.items() if k.startswith("S_")}
            grids = {}
            for key in SCALARS.values():
                src = Tr[f"S_{key}"] if Tr is not None and Tr["m"] else P[f"S_{key}"]
                grids[key] = grid_from(src)
            if Tr is not None and Tr["m"]:
                Xtr, ytr = train_rows([Tr])
                kinds = ["lr"] + ([] if args.no_gbm else ["gbm"])
                for kind in kinds:
                    fn = fit(kind, Xtr, ytr)
                    scores[f"LS_{kind}"] = score(fn, P); grids[f"LS_{kind}"] = grid_from(score(fn, Tr))
                same_model = [preps[k] for k in preps if k[1] == model and k[2] == "train" and preps[k]["m"]]
                other = [preps[k] for k in preps if k[1] == model and k[2] == "train" and k[0] != task
                         and preps[k]["m"]]
                if len(same_model) > 1:
                    fn = fit("lr", *train_rows(same_model))
                    scores["LS_pooled"] = score(fn, P); grids["LS_pooled"] = grid_from(score(fn, Tr))
                if other:
                    fn = fit("lr", *train_rows(other))
                    scores["LS_loto"] = score(fn, P); grids["LS_loto"] = grid_from(score(fn, Tr))
            C = build_cands(P, scores, grids, caps={int(x) for x in args.caps.split(",") if x})
            groups = sorted(set(C.group.tolist()))
            print(f"  [{acct}] {sid}: {len(C.label)} candidates, groups={groups}")
            main_grid = acct == MAIN["acct"]
            recs, choices = run_splits(P, C, np.arange(P["N"]), groups, alphas if main_grid else [MAIN["alpha"]],
                                       REGIMES if main_grid else ("kv_fork",), args.splits, args.cal_frac,
                                       args.delta, args.cert, args.prefill_weight, args.cache_discount,
                                       keep_choices=main_grid)
            for r in recs:
                r.update(setting=sid, task=task, model=model, acct=acct, cal_frac=args.cal_frac)
            all_recs += recs
            if main_grid:
                for a_, b_ in (("LearnStop", "Fixed"), ("LearnStop", "BestScalar"), ("LS+Cap", "Fixed"),
                               ("LearnStop", "AnswerConv")):
                    for reg in REGIMES:
                        ci = bootstrap_split0(P, C, choices, reg, MAIN["alpha"], a_, b_, args.prefill_weight,
                                              args.cache_discount)
                        if ci:
                            boot_rows.append(dict(setting=sid, regime=reg, alpha=MAIN["alpha"], A=a_, B=b_,
                                                  ci_lo=ci[0], ci_hi=ci[1]))
                for cf in [float(x) for x in args.cal_fracs.split(",")]:
                    rr, _ = run_splits(P, C, np.arange(P["N"]), ["Fixed", "BestScalar", "LearnStop", "LS+Cap"],
                                       [MAIN["alpha"]], ("kv_fork",), args.splits, cf, args.delta, args.cert,
                                       args.prefill_weight, args.cache_discount, seed0=1000)
                    for r in rr:
                        r.update(setting=sid, task=task, model=model, acct=acct, cal_frac=cf)
                    cf_recs += rr
                if task == "mmlu_pro":
                    cats = np.array([str(c).lower() for c in P["category"]])
                    for gname, sub in (("quant", np.where(np.isin(cats, list(QUANT_CATS)))[0]),
                                       ("non_quant", np.where(~np.isin(cats, list(QUANT_CATS)))[0])):
                        rr, _ = run_splits(P, C, sub, ["Fixed", "BestScalar", "LearnStop", "LS+Cap"],
                                           [MAIN["alpha"]], ("kv_fork",), args.splits, args.cal_frac,
                                           args.delta, args.cert, args.prefill_weight, args.cache_discount,
                                           seed0=2000)
                        for r in rr:
                            r.update(setting=sid, task=task, model=model, acct=acct, cal_frac=args.cal_frac,
                                     group=gname, n_group=len(sub))
                        cat_recs += rr
                        traj_rows.append(dict(setting=f"{sid}[{gname}]", task=task, model=model,
                                              **trajectory(P, sub)))

    Pm = prepared.get(MAIN["acct"], {})
    for (task, model, split) in evals:
        if "adaptthink" not in model.lower():
            continue
        base_key = next((k for k in Pm if k[0] == task and k[2] == "eval"
                         and k[1].lower().startswith("deepseek-r1-distill-qwen-7b")), None)
        if base_key is None:
            continue
        Pa, Pb = Pm[(task, model, split)], Pm[base_key]
        pos = {q: i for i, q in enumerate(Pb["qid"])}
        ia = np.array([i for i, q in enumerate(Pa["qid"]) if q in pos]); ib = np.array([pos[q] for q in Pa["qid"][ia]])
        n_cal = int(round(Pb["N"] * args.cal_frac))
        b2a = dict(zip(ib.tolist(), ia.tolist()))
        for sp in range(args.splits):
            perm = np.random.default_rng(sp).permutation(np.arange(Pb["N"]))
            te = np.array([t for t in perm[n_cal:] if t in b2a])
            ta = np.array([b2a[t] for t in te])
            ft_rows.append(dict(split=sp, task=task, model=model, base=base_key[1],
                                save=100 * (1 - Pa["gen_len"][ta].mean() / Pb["gen_len"][te].mean()),
                                risk=float(((Pb["nat"][te] == 1) & (Pa["nat"][ta] == 0)).mean()),
                                acc=Pa["nat"][ta].mean(), acc_base=Pb["nat"][te].mean(),
                                tokens=Pa["gen_len"][ta].mean(), tokens_base=Pb["gen_len"][te].mean(),
                                nothink=float((Pa["think"][ta] <= 5).mean())))

    keys = ["setting", "task", "model", "acct", "cal_frac"]
    res = summarise(all_recs, keys); res.to_csv(out / "results_all.csv", index=False)
    dl = paired_deltas(all_recs, keys, [("LearnStop", "Fixed"), ("LearnStop", "BestScalar"), ("LS+Cap", "Fixed"),
                                        ("LS+Cap", "BestScalar"), ("LearnStop", "AnswerConv"),
                                        ("LearnStop-Pooled", "LearnStop"), ("LearnStop-LOTO", "LearnStop"),
                                        ("LearnStop-GBM", "LearnStop"), ("BestScalar", "Fixed")])
    dl.to_csv(out / "deltas.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "bootstrap_split0.csv", index=False)
    cfs = summarise(cf_recs, keys); cfs.to_csv(out / "cal_frac_sensitivity.csv", index=False)
    cats = summarise(cat_recs, keys + ["group", "n_group"]); cats.to_csv(out / "mmlu_category.csv", index=False)
    st = pd.DataFrame(stats_rows); st.to_csv(out / "setting_stats.csv", index=False)
    tj = pd.DataFrame(traj_rows); tj.to_csv(out / "trajectory.csv", index=False)
    ft = pd.DataFrame(ft_rows)
    if not ft.empty:
        ft.groupby(["task", "model", "base"]).mean(numeric_only=True).drop(columns="split").reset_index() \
          .to_csv(out / "finetuned_baseline.csv", index=False)
    loo = regime_loo(res, st, tj)
    write_summary(out, args, res, dl, st, tj, cfs, cats, ft, loo, pd.DataFrame(boot_rows))
    print(f"[v2] done in {time.time() - t_start:.0f}s -> {out}")


def regime_loo(res, st, tj):
    m = res[(res.acct == MAIN["acct"]) & (res.regime == MAIN["regime"]) & (np.isclose(res.alpha, MAIN["alpha"]))]
    piv = m.pivot_table(index="setting", columns="policy", values="save_mean")
    cols = [c for c in ("Fixed", "BestScalar", "LearnStop") if c in piv]
    if len(cols) < 3:
        return None
    piv = piv[cols].dropna()
    lab = piv.idxmax(axis=1)
    d = tj.set_index("setting").join(st.set_index("setting")[["mean_think", "overhead_ratio"]], how="inner")
    d = d.loc[d.index.intersection(lab.index)]
    if len(d) < 4:
        return None
    feat = ["early", "benef", "harmful", "unsolved", "oscill", "mean_think", "overhead_ratio"]
    X = d[feat].values.astype(float); X[:, 5] = np.log(X[:, 5]); y = lab.loc[d.index].values
    from sklearn.tree import DecisionTreeClassifier, export_text
    acc_tree, acc_nn, acc_maj = [], [], []
    for k in range(len(y)):
        tr = np.arange(len(y)) != k
        t = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X[tr], y[tr])
        acc_tree.append(t.predict(X[k:k + 1])[0] == y[k])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        nn = np.argmin((((X[tr] - mu) / sd - (X[k] - mu) / sd) ** 2).sum(1))
        acc_nn.append(y[tr][nn] == y[k])
        acc_maj.append(Counter(y[tr]).most_common(1)[0][0] == y[k])
    tree = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    return dict(n=len(y), labels=dict(Counter(y)), loo_tree=float(np.mean(acc_tree)), loo_1nn=float(np.mean(acc_nn)),
                loo_majority=float(np.mean(acc_maj)), rules=export_text(tree, feature_names=feat),
                table=pd.DataFrame(X, index=d.index, columns=feat).assign(winner=y))


def write_summary(out, args, res, dl, st, tj, cfs, cats, ft, loo, boot):
    L = [f"# v2 analysis summary\n", f"ref={args.ref} cert={args.cert} delta={args.delta} splits={args.splits} "
         f"cal_frac={args.cal_frac} prefill_weight={args.prefill_weight} cache_discount={args.cache_discount}\n"]

    def tbl(df):
        if df.empty:
            return "(empty)"
        try:
            return df.to_markdown(index=False, floatfmt=".3g")
        except ImportError:
            return "```\n" + df.to_string(index=False, float_format=lambda x: f"{x:.3g}") + "\n```"

    L.append("## Settings (reference = natural full thinking + end probe)\n")
    L.append(tbl(st[["setting", "N", "acc_natural", "acc_full", "mean_think", "unclosed", "mean_active_ckpts",
                     "probe_len_line_mean", "probe_len_full_mean", "frac_probe_hit_cap", "overhead_ratio"]]))
    m = res[(res.acct == "line") & (res.regime == "kv_fork") & np.isclose(res.alpha, 0.15)]
    if not m.empty:
        L.append("\n## Main: total-token saving % at alpha=0.15, measured (line) probe cost, KV-fork\n")
        L.append(tbl(m.pivot_table(index="setting", columns="policy", values="save_mean").reset_index()))
        L.append("\n### test risk (mean) / risk-violation rate / accuracy\n")
        L.append(tbl(m[["setting", "policy", "risk_mean", "risk_viol", "acc_mean", "acc_full", "feasible_rate",
                        "top_cand"]]))
    if not dl.empty:
        d = dl[(dl.acct == "line") & (dl.regime == "kv_fork") & np.isclose(dl.alpha, 0.15)]
        L.append("\n## Paired differences over splits (pp, alpha=0.15, line, KV-fork)\n")
        L.append(tbl(d[["setting", "A", "B", "delta_mean", "delta_p025", "delta_p975", "win_rate"]]))
    if not boot.empty:
        L.append("\n## Question-level bootstrap 95% CI on split 0\n")
        L.append(tbl(boot))
    for acct in ("full", "cap"):
        a = res[(res.acct == acct) & (res.regime == "kv_fork") & np.isclose(res.alpha, 0.15)]
        if not a.empty:
            L.append(f"\n## Probe accounting = {acct} (alpha=0.15, KV-fork)\n")
            L.append(tbl(a.pivot_table(index="setting", columns="policy", values="save_mean").reset_index()))
    for reg in ("prefix_cache", "black_box"):
        a = res[(res.acct == "line") & (res.regime == reg) & np.isclose(res.alpha, 0.15)]
        if not a.empty:
            L.append(f"\n## Serving regime = {reg} (policy re-selected under this cost; alpha=0.15)\n")
            L.append(tbl(a.pivot_table(index="setting", columns="policy", values="save_mean").reset_index()))
    a = res[(res.acct == "line") & (res.regime == "kv_fork")]
    if not a.empty:
        L.append("\n## Alpha grid (LearnStop / LS+Cap / Fixed / BestScalar)\n")
        sub = a[a.policy.isin(["LearnStop", "LS+Cap", "Fixed", "BestScalar"])]
        L.append(tbl(sub.pivot_table(index=["setting", "alpha"], columns="policy", values="save_mean").reset_index()))
    if not cfs.empty:
        L.append("\n## Calibration-fraction sensitivity (alpha=0.15, line, KV-fork)\n")
        L.append(tbl(cfs.pivot_table(index=["setting", "cal_frac"], columns="policy", values="save_mean").reset_index()))
    L.append("\n## Trajectory decomposition (% of questions)\n")
    L.append(tbl(tj))
    if loo:
        L.append(f"\n## Regime prediction, leave-one-setting-out (n={loo['n']}, winners={loo['labels']})\n")
        L.append(f"depth-2 tree LOO acc = {loo['loo_tree']:.2f}; 1-NN = {loo['loo_1nn']:.2f}; "
                 f"majority = {loo['loo_majority']:.2f}\n\n```\n{loo['rules']}```\n")
        L.append(tbl(loo["table"].reset_index()))
    if not cats.empty:
        L.append("\n## MMLU-Pro by category group (alpha=0.15, line, KV-fork)\n")
        L.append(tbl(cats.pivot_table(index=["setting", "group", "n_group"], columns="policy",
                                      values="save_mean").reset_index()))
    if not ft.empty:
        L.append("\n## Fine-tuned adaptive CoT (AdaptThink) vs base model, same test halves\n")
        L.append(tbl(ft.groupby(["task", "model", "base"]).mean(numeric_only=True).drop(columns="split").reset_index()))
    (out / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
