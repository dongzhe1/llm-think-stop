"""Offline analysis of a saved probe run — NO GPU required.

Reads results/<...>/raw.npz (the per-question matrices dumped by the probes) and
produces the things reviewers will ask for, all on CPU:

  1. Strong baselines (accuracy–token frontiers + peak adapt gain over fixed budget):
       - fixed-budget        (= s1 budget forcing; the non-adaptive reference)
       - confidence exit     (DEER-style: stop when answer-token confidence ≥ c)
       - entropy exit        (stop when answer-token entropy ≤ e)
       - self-consistency    (stop when the answer has been stable for k checkpoints)
       - learned (ours)      (saved held-out p_stop, sweep threshold)
  2. Bootstrap 95% CI on the peak adapt gain at the operating point chosen on full data,
     for ours and every baseline — quantifies significance and ours-vs-baseline gaps.
  3. Feature ablation: re-train the logistic stop-classifier with grouped CV using only
     conf_lp vs all 10 features; compare peak gain.

Usage:
  python scripts/analyze_results.py results/learnstop/gsm8k_Qwen3-8B_n1000_1849040
  python scripts/analyze_results.py results/transfer/gsm8k-to-math500_Qwen3-32B_n200_xxx --npz target_raw.npz
"""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CONF_LP_IDX = 2   # column of conf_lp in the 10-dim feature matrix (see meta feature_names)
USED_K_IDX = 8    # column of used_k
ENDED_IDX = 9     # column of ended
DEFAULT_ANS_TOKENS = 48


# frontier utilities

def fixed_frontier(correct, think_lens, budgets):
    fix_tok = np.array([float(np.mean(np.minimum(think_lens, B))) for B in budgets])
    fix_acc = np.array([float(correct[:, j].mean()) for j in range(len(budgets))])
    order = np.argsort(fix_tok)
    return fix_tok, fix_acc, fix_tok[order], fix_acc[order]


def simulate(stop_idx, correct, think_lens, budgets):
    """Given a stop checkpoint index per question, return (accuracy, avg_tokens)."""
    N = correct.shape[0]
    accs = correct[np.arange(N), stop_idx]
    toks = np.minimum(think_lens, np.array(budgets)[stop_idx])
    return float(accs.mean()), float(toks.mean())


def first_ge(scores, thr):
    """First checkpoint index whose score ≥ thr, else last."""
    m = scores.shape[1]
    out = np.full(scores.shape[0], m - 1, dtype=int)
    for i in range(scores.shape[0]):
        hit = np.nonzero(scores[i] >= thr)[0]
        if hit.size:
            out[i] = hit[0]
    return out


def first_le(scores, thr):
    m = scores.shape[1]
    out = np.full(scores.shape[0], m - 1, dtype=int)
    for i in range(scores.shape[0]):
        hit = np.nonzero(scores[i] <= thr)[0]
        if hit.size:
            out[i] = hit[0]
    return out


def stability_run(ans):
    """run_len[i,j] = how many consecutive checkpoints (ending at j) share the same answer."""
    N, m = ans.shape
    run = np.zeros((N, m), dtype=int)
    for i in range(N):
        for j in range(m):
            a = ans[i, j]
            if a == "" or a == "nan":
                run[i, j] = 0
            elif j > 0 and ans[i, j] == ans[i, j - 1]:
                run[i, j] = run[i, j - 1] + 1
            else:
                run[i, j] = 1
    return run


def best_gain_over_sweep(stop_fn, sweep, correct, think_lens, budgets, fx_t, fx_a):
    """Over a threshold sweep, return (peak_gain, op_thr, frontier_rows)."""
    rows, best = [], (-9.9, None)
    for thr in sweep:
        idx = stop_fn(thr)
        a, t = simulate(idx, correct, think_lens, budgets)
        fa = float(np.interp(t, fx_t, fx_a))
        g = a - fa
        rows.append((float(thr), a, t, fa, g))
        if g > best[0]:
            best = (g, float(thr))
    return best[0], best[1], rows


# baselines

def run_baselines(d):
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    conf_lp, conf_ent = d["conf_lp"], d["conf_ent"]
    ans, p_stop = d["ans"], d["p_stop"]
    _, _, fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    conf = np.exp(conf_lp)                       # answer-token confidence in [0,1]
    run = stability_run(ans)

    out = {}
    # learned (ours): saved held-out p_stop
    out["learned"] = best_gain_over_sweep(
        lambda thr: first_ge(p_stop, thr), np.round(np.linspace(0.30, 0.95, 14), 3),
        correct, think_lens, budgets, fx_t, fx_a)
    # DEER-style confidence exit
    cgrid = np.round(np.linspace(conf.min(), conf.max(), 30), 4)
    out["confidence_exit"] = best_gain_over_sweep(
        lambda thr: first_ge(conf, thr), cgrid, correct, think_lens, budgets, fx_t, fx_a)
    # entropy exit (stop when entropy low)
    egrid = np.round(np.linspace(conf_ent.min(), conf_ent.max(), 30), 4)
    out["entropy_exit"] = best_gain_over_sweep(
        lambda thr: first_le(conf_ent, thr), egrid, correct, think_lens, budgets, fx_t, fx_a)
    # self-consistency / stability (stop when answer stable for k checkpoints)
    out["self_consistency"] = best_gain_over_sweep(
        lambda k: first_ge(run, k), list(range(1, len(budgets) + 1)),
        correct, think_lens, budgets, fx_t, fx_a)
    # confidence leaps: stop when delta_conf >= gamma AND conf >= tau
    N, m = correct.shape
    delta_conf = np.zeros_like(conf)
    delta_conf[:, 1:] = conf[:, 1:] - conf[:, :-1]
    def conf_leap_stop(params):
        gamma, tau = params
        idx = np.full(N, m - 1, dtype=int)
        for i in range(N):
            for j in range(m):
                if delta_conf[i, j] >= gamma and conf[i, j] >= tau:
                    idx[i] = j
                    break
        return idx
    best_leap = (-9.9, None)
    leap_rows = []
    for gamma in np.round(np.linspace(0.01, 0.5, 12), 3):
        for tau in np.round(np.linspace(0.3, 0.95, 8), 3):
            idx = conf_leap_stop((gamma, tau))
            a, t = simulate(idx, correct, think_lens, budgets)
            fa = float(np.interp(t, fx_t, fx_a))
            g = a - fa
            if g > best_leap[0]:
                best_leap = (g, (float(gamma), float(tau)))
    if best_leap[1] is not None:
        out["confidence_leap"] = (best_leap[0], best_leap[1], [])

    return out, (fx_t, fx_a)


# bootstrap CI

def bootstrap_gain(stop_idx_at_op, correct, think_lens, budgets, fx_t_full, fx_a_full,
                   B=1000, seed=0):
    """Bootstrap the adapt gain at a FIXED operating point (stop indices precomputed)."""
    rng = np.random.default_rng(seed)
    N = correct.shape[0]
    bud = np.array(budgets)
    gains = np.empty(B)
    for b in range(B):
        s = rng.integers(0, N, N)
        a = correct[s, stop_idx_at_op[s]].mean()
        t = np.minimum(think_lens[s], bud[stop_idx_at_op[s]]).mean()
        # recompute fixed frontier on the SAME resample for a paired comparison
        ft = np.array([np.minimum(think_lens[s], Bb).mean() for Bb in budgets])
        fa = np.array([correct[s, j].mean() for j in range(len(budgets))])
        order = np.argsort(ft)
        gains[b] = a - float(np.interp(t, ft[order], fa[order]))
    return float(np.median(gains)), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5))


# paired bootstrap: learned vs each baseline

def paired_bootstrap(stop_a, stop_b, correct, think_lens, budgets, B=10000, seed=42):
    """Bootstrap CI on (gain_A - gain_B) with paired resampling."""
    rng = np.random.default_rng(seed)
    N = correct.shape[0]
    bud = np.array(budgets)
    diffs = np.empty(B)
    for b in range(B):
        s = rng.integers(0, N, N)
        aa = correct[s, stop_a[s]].mean()
        ta = np.minimum(think_lens[s], bud[stop_a[s]]).mean()
        ab = correct[s, stop_b[s]].mean()
        tb = np.minimum(think_lens[s], bud[stop_b[s]]).mean()
        ft = np.array([np.minimum(think_lens[s], Bb).mean() for Bb in budgets])
        fa = np.array([correct[s, j].mean() for j in range(len(budgets))])
        order = np.argsort(ft)
        ga = aa - float(np.interp(ta, ft[order], fa[order]))
        gb = ab - float(np.interp(tb, ft[order], fa[order]))
        diffs[b] = ga - gb
    return float(np.mean(diffs)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# permutation test

def permutation_test(stop_a, stop_b, correct, think_lens, budgets, n_perm=10000, seed=42):
    """Two-sided permutation test on per-question accuracy difference."""
    rng = np.random.default_rng(seed)
    N = correct.shape[0]
    bud = np.array(budgets)
    acc_a = correct[np.arange(N), stop_a]
    acc_b = correct[np.arange(N), stop_b]
    d = acc_a - acc_b
    obs = float(np.abs(d.mean()))
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=N)
        if np.abs((d * signs).mean()) >= obs:
            count += 1
    return count / n_perm


# validation-selected operating point

def validation_selected_gain(d, seed=99):
    """Proper train/cal/test: threshold chosen on cal set, evaluated on test set."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    conf = np.exp(d["conf_lp"])
    run = stability_run(d["ans"])
    N, m = correct.shape

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_cal = int(round(N * 0.4))
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    def fixed_frontier_sub(idx):
        ft = np.array([float(np.mean(np.minimum(think_lens[idx], B))) for B in budgets])
        fa = np.array([float(correct[idx, j].mean()) for j in range(len(budgets))])
        order = np.argsort(ft)
        return ft[order], fa[order]

    def best_on_cal(stop_fn, sweep, idx):
        ft, fa = fixed_frontier_sub(idx)
        best_g, best_thr = -9.9, None
        for thr in sweep:
            sidx = stop_fn(thr)
            a = correct[idx, sidx[idx]].mean()
            t = np.minimum(think_lens[idx], np.array(budgets)[sidx[idx]]).mean()
            g = a - float(np.interp(t, ft, fa))
            if g > best_g:
                best_g, best_thr = g, thr
        return best_thr

    def eval_on_test(stop_fn, thr, idx):
        ft, fa = fixed_frontier_sub(idx)
        sidx = stop_fn(thr)
        a = correct[idx, sidx[idx]].mean()
        t = np.minimum(think_lens[idx], np.array(budgets)[sidx[idx]]).mean()
        g = a - float(np.interp(t, ft, fa))
        return float(g), float(a), float(t)

    thr_grid_learned = np.round(np.linspace(0.30, 0.95, 14), 3)
    cgrid = np.round(np.linspace(conf.min(), conf.max(), 30), 4)
    egrid = np.round(np.linspace(d["conf_ent"].min(), d["conf_ent"].max(), 30), 4)

    methods = {
        "learned": (lambda thr: first_ge(p_stop, thr), thr_grid_learned),
        "confidence_exit": (lambda thr: first_ge(conf, thr), cgrid),
        "entropy_exit": (lambda thr: first_le(d["conf_ent"], thr), egrid),
        "self_consistency": (lambda thr: first_ge(run, thr), list(range(1, len(budgets) + 1))),
    }

    results = {}
    for name, (fn, sweep) in methods.items():
        thr = best_on_cal(fn, sweep, cal_idx)
        if thr is None:
            continue
        g, a, t = eval_on_test(fn, thr, test_idx)
        results[name] = {"val_thr": round(float(thr), 4), "test_gain": round(g, 4),
                         "test_acc": round(a, 4), "test_tok": round(t, 1)}
    return results


# trajectory decomposition

def trajectory_decomposition(correct):
    """Classify each question by its correctness trajectory across checkpoints."""
    N, m = correct.shape
    cats = {"early_solved": 0, "beneficial_thinking": 0,
            "harmful_overthinking": 0, "unsolved": 0, "oscillating": 0}
    for i in range(N):
        seq = correct[i]
        first_correct = np.nonzero(seq)[0]
        if first_correct.size == 0:
            cats["unsolved"] += 1
        elif seq[-1] == 1 and seq[0] == 1:
            cats["early_solved"] += 1
        elif seq[-1] == 1 and seq[0] == 0:
            changes = np.sum(np.abs(np.diff(seq)))
            if changes <= 2:
                cats["beneficial_thinking"] += 1
            else:
                cats["oscillating"] += 1
        elif seq[-1] == 0 and first_correct.size > 0:
            cats["harmful_overthinking"] += 1
        else:
            cats["unsolved"] += 1
    total = sum(cats.values())
    return {k: {"count": v, "pct": round(100.0 * v / max(total, 1), 1)} for k, v in cats.items()}


# feature ablation

def ablation(d, folds=5, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y = d["X"], d["y"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    groups = np.repeat(np.arange(N), m)
    _, _, fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    def cv_pstop(cols):
        p = np.zeros((N, m))
        gkf = GroupKFold(n_splits=folds)
        Xc = X[:, cols]
        for tr, te in gkf.split(Xc, y, groups):
            sc = StandardScaler().fit(Xc[tr])
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(Xc[tr]), y[tr])
            pr = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
            for idx, pp in zip(te, pr):
                p[idx // m, idx % m] = pp
        g, thr, _ = best_gain_over_sweep(
            lambda t: first_ge(p, t), np.round(np.linspace(0.30, 0.95, 14), 3),
            correct, think_lens, budgets, fx_t, fx_a)
        return g, thr

    all_cols = list(range(X.shape[1]))
    no_position_cols = [c for c in all_cols if c not in (USED_K_IDX, ENDED_IDX)]

    g_all, thr_all = cv_pstop(all_cols)
    g_one, thr_one = cv_pstop([CONF_LP_IDX])
    g_nop, thr_nop = cv_pstop(no_position_cols)
    return {"all_features": {"peak_gain": round(g_all, 4), "op_thr": thr_all},
            "conf_lp_only": {"peak_gain": round(g_one, 4), "op_thr": thr_one},
            "no_ended_usedk": {"peak_gain": round(g_nop, 4), "op_thr": thr_nop},
            "delta": round(g_all - g_one, 4),
            "delta_no_ended_usedk": round(g_all - g_nop, 4)}


def conformal_recompute(d, ans_tokens=DEFAULT_ANS_TOKENS, delta=0.05):
    """Recompute conformal risk control with corrected risk definition and train/cal/test split.

    Risk: R_i(τ) = 1{full-thinking correct but stopped answer wrong}.
    This directly bounds accuracy loss: Acc_full - Acc_stop ≤ E[R(τ)].

    Protocol: split questions 60/20/20 into train/cal/test.
    - Train: used for classifier training (already done in grouped CV — we use p_stop from CV).
    - Cal: select threshold τ* satisfying R̂_cal(τ*) + Hoeffding ≤ α.
    - Test: evaluate accuracy and saving at τ*.
    """
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape
    full_correct = correct[:, -1]
    mean_full_tok = float(think_lens.mean())

    rng = np.random.default_rng(123)
    perm = rng.permutation(N)
    n_cal = int(round(N * 0.4))
    n_test = N - n_cal
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    def risk_on_subset(thr, idx):
        rv = np.zeros(len(idx))
        for ii, i in enumerate(idx):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            rv[ii] = float(full_correct[i] == 1 and correct[i, j] == 0)
        return rv

    def policy_on_subset(thr, idx):
        accs, toks, probes = [], [], []
        for i in idx:
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            accs.append(correct[i, j])
            toks.append(min(int(think_lens[i]), budgets[j]))
            probes.append((j + 1) * ans_tokens)
        return float(np.mean(accs)), float(np.mean(toks)), float(np.mean(probes))

    thr_grid = sorted(np.round(np.linspace(0.20, 0.99, 40), 3).tolist())
    K = len(thr_grid)
    hb = float(np.sqrt(np.log(K / delta) / (2.0 * n_cal)))  # union-bound over K thresholds

    results = []
    for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
        tau_star = None
        for thr in thr_grid:
            emp_risk = risk_on_subset(thr, cal_idx).mean()
            if emp_risk + hb <= alpha:
                tau_star = thr
                break
        if tau_star is None:
            results.append({"alpha": alpha, "tau": None, "fire": False})
        else:
            cal_risk = risk_on_subset(tau_star, cal_idx).mean()
            test_risk = risk_on_subset(tau_star, test_idx).mean()
            test_acc, test_tok, test_oh = policy_on_subset(tau_star, test_idx)
            full_total = mean_full_tok + ans_tokens
            think_saving = 100.0 * (1.0 - test_tok / max(mean_full_tok, 1))
            total_saving = 100.0 * (1.0 - (test_tok + test_oh) / max(full_total, 1))
            results.append({
                "alpha": alpha, "tau": round(float(tau_star), 3),
                "n_cal": n_cal, "n_test": n_test,
                "cal_risk": round(float(cal_risk), 4),
                "guaranteed": round(float(cal_risk + hb), 4),
                "test_risk": round(float(test_risk), 4),
                "test_acc": round(float(test_acc), 4),
                "test_tok": round(float(test_tok), 1),
                "probe_tok": round(float(test_oh), 1),
                "think_saving_pct": round(float(think_saving), 1),
                "total_saving_pct": round(float(total_saving), 1),
                "fire": True,
            })
    return results


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    # transfer dumps store conf_lp/conf_ent only inside X (cols 2,3); reconstruct if absent
    if "conf_lp" not in d and "X" in d and "correct" in d:
        N, m = d["correct"].shape
        d["conf_lp"] = d["X"][:, CONF_LP_IDX].reshape(N, m)
        d["conf_ent"] = d["X"][:, CONF_LP_IDX + 1].reshape(N, m)
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="a results/<...> run directory")
    parser.add_argument("--npz", default="raw.npz",
                    help="which npz inside the dir (raw.npz | source_raw.npz | target_raw.npz)")
    parser.add_argument("--boot", type=int, default=1000)
    parser.add_argument("--no_ablation", action="store_true",
                    help="skip the (slower) feature-ablation re-training")
    args = parser.parse_args()

    run = Path(args.run_dir)
    d = load_npz(run / args.npz)
    budgets = list(d["budgets"])
    correct, think_lens = d["correct"], d["think_lens"]

    base, (fx_t, fx_a) = run_baselines(d)

    logger.info(f"\n=== BASELINES — {run.name} / {args.npz} ===")
    logger.info(f"  {'method':>18}  {'peak gain':>10}  {'op thr':>12}")
    method_rows = []
    for name, (g, thr, rows) in base.items():
        thr_str = f"{thr}" if isinstance(thr, tuple) else f"{thr:.3f}"
        logger.info(f"  {name:>18}  {g:>+10.4f}  {thr_str:>12}")
        method_rows.append([name, round(g, 4), thr_str])

    # bootstrap CI at each method's operating point
    logger.info(f"\n=== BOOTSTRAP 95% CI on adapt gain (B={args.boot}) ===")
    conf = np.exp(d["conf_lp"]); run_stab = stability_run(d["ans"])
    N_q = correct.shape[0]
    delta_conf = np.zeros_like(conf)
    delta_conf[:, 1:] = conf[:, 1:] - conf[:, :-1]
    stopfns = {
        "learned":         lambda thr: first_ge(d["p_stop"], thr),
        "confidence_exit": lambda thr: first_ge(conf, thr),
        "entropy_exit":    lambda thr: first_le(d["conf_ent"], thr),
        "self_consistency":lambda thr: first_ge(run_stab, thr),
    }
    if "confidence_leap" in base:
        gamma_star, tau_star = base["confidence_leap"][1]
        def _leap_stop(params, _g=gamma_star, _t=tau_star):
            m = correct.shape[1]
            idx = np.full(N_q, m - 1, dtype=int)
            for i in range(N_q):
                for j in range(m):
                    if delta_conf[i, j] >= _g and conf[i, j] >= _t:
                        idx[i] = j; break
            return idx
        stopfns["confidence_leap"] = lambda thr, _f=_leap_stop: _f(thr)

    boot_rows = []
    for name, (g, thr, _rows) in base.items():
        if name == "confidence_leap":
            idx_op = stopfns[name](None)
        else:
            idx_op = stopfns[name](thr)
        med, lo, hi = bootstrap_gain(idx_op, correct, think_lens, budgets, fx_t, fx_a,
                                     B=args.boot)
        sig = "*" if lo > 0 else " "
        logger.info(f"  {name:>18}  median {med:>+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}] {sig}")
        boot_rows.append([name, round(med, 4), round(lo, 4), round(hi, 4), int(lo > 0)])

    # paired bootstrap: learned vs each baseline
    learned_thr = base["learned"][1]
    learned_idx = stopfns["learned"](learned_thr)
    logger.info(f"\n=== PAIRED BOOTSTRAP: learned − baseline (B=10000) ===")
    paired_rows = []
    for name in base:
        if name == "learned":
            continue
        if name == "confidence_leap":
            bl_idx = stopfns[name](None)
        else:
            bl_idx = stopfns[name](base[name][1])
        mean_d, lo_d, hi_d = paired_bootstrap(learned_idx, bl_idx, correct, think_lens, budgets)
        sig = "*" if lo_d > 0 else ("†" if hi_d < 0 else " ")
        logger.info(f"  learned − {name:<18}  mean {mean_d:>+.4f}  95% CI [{lo_d:+.4f}, {hi_d:+.4f}] {sig}")
        paired_rows.append([name, round(mean_d, 4), round(lo_d, 4), round(hi_d, 4)])

    # permutation test: learned vs strongest scalar baseline
    strongest_scalar = max(
        [(n, base[n][0]) for n in base if n != "learned"],
        key=lambda x: x[1]
    )
    sn = strongest_scalar[0]
    if sn == "confidence_leap":
        s_idx = stopfns[sn](None)
    else:
        s_idx = stopfns[sn](base[sn][1])
    perm_p = permutation_test(learned_idx, s_idx, correct, think_lens, budgets)
    logger.info(f"\n=== PERMUTATION TEST: learned vs {sn} ===")
    logger.info(f"  p-value = {perm_p:.4f}  ({'significant' if perm_p < 0.05 else 'not significant'} at α=0.05)")

    # validation-selected operating point
    val_sel = validation_selected_gain(d)
    logger.info(f"\n=== VALIDATION-SELECTED OPERATING POINT (threshold chosen on cal, evaluated on test) ===")
    for name, v in val_sel.items():
        logger.info(f"  {name:>18}  thr={v['val_thr']:.3f}  test_gain={v['test_gain']:+.4f}  "
              f"test_acc={v['test_acc']:.4f}  test_tok={v['test_tok']:.0f}")

    # trajectory decomposition
    traj = trajectory_decomposition(correct)
    logger.info(f"\n=== TRAJECTORY DECOMPOSITION ===")
    for cat, v in traj.items():
        logger.info(f"  {cat:<25}  {v['count']:>5}  ({v['pct']:>5.1f}%)")

    # feature ablation
    abl = None
    if not args.no_ablation and "X" in d and "y" in d:
        abl = ablation(d)
        logger.info(f"\n=== FEATURE ABLATION (grouped CV) ===")
        logger.info(f"  all 10 features       : peak gain {abl['all_features']['peak_gain']:+.4f}")
        logger.info(f"  no ended/used_k (8)   : peak gain {abl['no_ended_usedk']['peak_gain']:+.4f}  "
              f"(Δ = {abl['delta_no_ended_usedk']:+.4f})")
        logger.info(f"  conf_lp only (1)      : peak gain {abl['conf_lp_only']['peak_gain']:+.4f}  "
              f"(Δ = {abl['delta']:+.4f})")

    # conformal with corrected risk definition + cal/test split + union-bound
    conf_results = None
    if "p_stop" in d:
        meta_path = run / "meta.json"
        ans_tok = DEFAULT_ANS_TOKENS
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                ans_tok = meta.get("ans_tokens", DEFAULT_ANS_TOKENS)
        conf_results = conformal_recompute(d, ans_tokens=ans_tok)
        logger.info(f"\n=== CONFORMAL RISK CONTROL (union-bound over K thresholds, cal/test split) ===")
        logger.info(f"  {'α':>6}  {'τ*':>6}  {'cal risk':>9}  {'≤ bound':>9}  {'test risk':>10}  "
              f"{'test acc':>9}  {'think save':>11}  {'total save':>11}")
        for r in conf_results:
            if not r["fire"]:
                logger.info(f"  {r['alpha']:>6.2f}  {'—':>6}  {'n/a':>9}")
            else:
                logger.info(f"  {r['alpha']:>6.2f}  {r['tau']:>6.2f}  {r['cal_risk']:>9.4f}  "
                      f"≤{r['guaranteed']:>7.4f}  {r['test_risk']:>10.4f}  {r['test_acc']:>9.4f}  "
                      f"{r['think_saving_pct']:>10.1f}%  {r['total_saving_pct']:>10.1f}%")

    # write analysis artefacts
    adir = run / "analysis"
    adir.mkdir(exist_ok=True)
    import csv as _csv
    with open(adir / "baselines.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["method", "peak_gain", "op_thr"]); w.writerows(method_rows)
    with open(adir / "bootstrap_ci.csv", "w", newline="") as f:
        w = _csv.writer(f); w.writerow(["method", "median_gain", "ci_lo", "ci_hi", "sig_gt_0"])
        w.writerows(boot_rows)
    out = {"run": run.name, "npz": args.npz,
           "baselines": {k: {"peak_gain": round(v[0], 4),
                              "op_thr": round(v[1], 3) if isinstance(v[1], (int, float)) else v[1]}
                         for k, v in base.items()},
           "bootstrap": {r[0]: {"median": r[1], "ci": [r[2], r[3]], "sig": bool(r[4])}
                         for r in boot_rows},
           "paired_bootstrap": {r[0]: {"mean_diff": r[1], "ci": [r[2], r[3]],
                                        "sig": bool(r[2] > 0)}
                                 for r in paired_rows},
           "permutation_test": {"baseline": sn, "p_value": round(perm_p, 4)},
           "validation_selected": val_sel,
           "trajectory": traj,
           "ablation": abl,
           "conformal_caltest": conf_results}
    with open(adir / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    if conf_results:
        with open(adir / "conformal_caltest.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["alpha", "tau", "n_cal", "n_test", "cal_risk", "guaranteed",
                         "test_risk", "test_acc", "test_tok", "probe_tok",
                         "think_saving_pct", "total_saving_pct"])
            for r in conf_results:
                if r["fire"]:
                    w.writerow([r["alpha"], r["tau"], r["n_cal"], r["n_test"],
                                r["cal_risk"], r["guaranteed"], r["test_risk"],
                                r["test_acc"], r["test_tok"], r["probe_tok"],
                                r["think_saving_pct"], r["total_saving_pct"]])
    logger.info(f"\n[analyze] wrote {adir}/ analysis.json + CSVs")


if __name__ == "__main__":
    main()
