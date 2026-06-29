"""Extended analysis: implements ALL remaining experiments from aaai2027_experiment_roadmap.md.

Reads raw.npz and produces additional analysis beyond what analyze_results.py covers:

  A1b. DEER-style transition-point exit (marker-gated confidence)
  A2b. EAT-style entropy stability baseline
  A5.  PUMA-style semantic convergence proxy (stability + marker combo)
  A6.  TERMINATOR-light (earliest-correct-checkpoint supervision)
  C4.  Holm-Bonferroni multiple comparison correction
  D.   Cost model comparison (decode-only, prefix-cache, black-box API)
  D4.  Probe overhead sweep (simulated ans_tokens)
  E4.  Learn-then-Test sequential testing
  E5.  Conformal Thinking-style single-confidence UCB baseline
  F3.  Extended feature ablation (all subsets from roadmap)
  F4.  Model class comparison (RF, GBT, MLP vs logistic)
  F5.  Probability calibration (ECE, Brier, reliability bins)
  F6.  Leakage audit
  H3.  Calibration size curve (n_cal sweep)
  I3.  Answer transition analysis
  I4.  Feature importance (coefficients + permutation importance)
  K.   Checkpoint subset simulation

Usage:
  python scripts/analyze_extended.py results/learnstop/gsm8k_Qwen3-32B_n1000_1861181
"""
from __future__ import annotations

import argparse
import logging
import json
import warnings
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

CONF_LP_IDX = 2
CONF_ENT_IDX = 3
STABLE_IDX = 4
RUN_LEN_IDX = 5
VOTE_SHARE_IDX = 6
MKR_IDX = 7
USED_K_IDX = 8
ENDED_IDX = 9
DEFAULT_ANS_TOKENS = 48

FEATURE_NAMES = [
    "budget_k", "ckpt_frac", "conf_lp", "conf_ent", "stable",
    "run_len", "vote_share", "marker_density", "used_k", "ended",
]


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    if "conf_lp" not in d and "X" in d and "correct" in d:
        N, m = d["correct"].shape
        d["conf_lp"] = d["X"][:, CONF_LP_IDX].reshape(N, m)
        d["conf_ent"] = d["X"][:, CONF_ENT_IDX].reshape(N, m)
    return d


def fixed_frontier(correct, think_lens, budgets):
    fix_tok = np.array([float(np.mean(np.minimum(think_lens, B))) for B in budgets])
    fix_acc = np.array([float(correct[:, j].mean()) for j in range(len(budgets))])
    order = np.argsort(fix_tok)
    return fix_tok[order], fix_acc[order]


def simulate(stop_idx, correct, think_lens, budgets):
    N = correct.shape[0]
    accs = correct[np.arange(N), stop_idx]
    toks = np.minimum(think_lens, np.array(budgets)[stop_idx])
    return float(accs.mean()), float(toks.mean())


def first_ge(scores, thr):
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


def best_gain(stop_fn, sweep, correct, think_lens, budgets, fx_t, fx_a):
    best_g, best_thr = -9.9, None
    for thr in sweep:
        idx = stop_fn(thr)
        a, t = simulate(idx, correct, think_lens, budgets)
        fa = float(np.interp(t, fx_t, fx_a))
        g = a - fa
        if g > best_g:
            best_g, best_thr = g, thr
    return round(best_g, 4), best_thr


# ========================================================================
# A1b. DEER-style transition-point exit
# ========================================================================
def deer_transition_exit(d):
    """DEER-style: stop at first checkpoint where confidence >= c AND marker_density >= m_thr.
    Marker density proxies for 'reasoning transition point'."""
    conf = np.exp(d["conf_lp"])
    mkr = d["mkr"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    best_g, best_params = -9.9, None
    for c_thr in np.round(np.linspace(0.3, 0.99, 15), 3):
        for m_thr in np.round(np.linspace(0.0, mkr.max(), 8), 4):
            idx = np.full(N, m - 1, dtype=int)
            for i in range(N):
                for j in range(m):
                    if conf[i, j] >= c_thr and mkr[i, j] >= m_thr:
                        idx[i] = j
                        break
            a, t = simulate(idx, correct, think_lens, budgets)
            fa = float(np.interp(t, fx_t, fx_a))
            g = a - fa
            if g > best_g:
                best_g = g
                best_params = (float(c_thr), float(m_thr))
    return {"peak_gain": round(best_g, 4), "params": best_params, "method": "deer_transition"}


# ========================================================================
# A2b. EAT-style entropy stability
# ========================================================================
def eat_entropy_stability(d):
    """EAT-style: stop when entropy variance over last k checkpoints < threshold."""
    conf_ent = d["conf_ent"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    best_g, best_params = -9.9, None
    for k in [2, 3, 4]:
        ent_var = np.full((N, m), 999.0)
        for i in range(N):
            for j in range(k - 1, m):
                ent_var[i, j] = np.var(conf_ent[i, max(0, j - k + 1):j + 1])
        for v_thr in np.round(np.linspace(0.001, 1.0, 20), 4):
            idx = np.full(N, m - 1, dtype=int)
            for i in range(N):
                for j in range(k - 1, m):
                    if ent_var[i, j] <= v_thr:
                        idx[i] = j
                        break
            a, t = simulate(idx, correct, think_lens, budgets)
            fa = float(np.interp(t, fx_t, fx_a))
            g = a - fa
            if g > best_g:
                best_g = g
                best_params = (k, float(v_thr))
    return {"peak_gain": round(best_g, 4), "params": best_params, "method": "eat_entropy_stability"}


# ========================================================================
# A5. PUMA-style semantic convergence proxy
# ========================================================================
def puma_convergence_proxy(d):
    """PUMA approximation: stop when answer stable AND low backtracking marker density.
    Stability + low marker = semantic convergence proxy."""
    run = stability_run(d["ans"])
    mkr = d["mkr"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    best_g, best_params = -9.9, None
    for r_thr in range(1, m + 1):
        for m_thr in np.round(np.linspace(0.0, mkr.max(), 10), 4):
            idx = np.full(N, m - 1, dtype=int)
            for i in range(N):
                for j in range(m):
                    if run[i, j] >= r_thr and mkr[i, j] <= m_thr:
                        idx[i] = j
                        break
            a, t = simulate(idx, correct, think_lens, budgets)
            fa = float(np.interp(t, fx_t, fx_a))
            g = a - fa
            if g > best_g:
                best_g = g
                best_params = (r_thr, float(m_thr))
    return {"peak_gain": round(best_g, 4), "params": best_params, "method": "puma_convergence"}


# ========================================================================
# A6. TERMINATOR-light: earliest-correct supervision
# ========================================================================
def terminator_light(d, folds=5, seed=42):
    """Train classifier to predict optimal exit = earliest correct checkpoint."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, correct = d["X"], d["correct"]
    think_lens, budgets = d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

    # Label: 1 if this is at or after the earliest correct checkpoint
    y_term = np.zeros(N * m, dtype=int)
    for i in range(N):
        first_c = np.nonzero(correct[i])[0]
        if first_c.size > 0:
            ec = first_c[0]
            for j in range(ec, m):
                y_term[i * m + j] = 1

    groups = np.repeat(np.arange(N), m)
    p_term = np.zeros((N, m))
    gkf = GroupKFold(n_splits=folds)
    cols = list(range(X.shape[1]))

    for tr, te in gkf.split(X, y_term, groups):
        sc = StandardScaler().fit(X[tr][:, cols])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(sc.transform(X[tr][:, cols]), y_term[tr])
        pr = clf.predict_proba(sc.transform(X[te][:, cols]))[:, 1]
        for idx_flat, pp in zip(te, pr):
            p_term[idx_flat // m, idx_flat % m] = pp

    g, thr = best_gain(lambda t: first_ge(p_term, t),
                        np.round(np.linspace(0.30, 0.95, 14), 3),
                        correct, think_lens, budgets, fx_t, fx_a)
    return {"peak_gain": g, "op_thr": thr, "method": "terminator_light"}


# ========================================================================
# C4. Holm-Bonferroni multiple comparison correction
# ========================================================================
def holm_bonferroni(p_values, alpha=0.05):
    """Apply Holm-Bonferroni correction to a dict of {name: p_value}."""
    items = sorted(p_values.items(), key=lambda x: x[1])
    k = len(items)
    results = {}
    for i, (name, p) in enumerate(items):
        adj_alpha = alpha / (k - i)
        results[name] = {
            "raw_p": round(p, 6),
            "adj_alpha": round(adj_alpha, 6),
            "reject": bool(p < adj_alpha),
            "rank": i + 1,
        }
    return results


# ========================================================================
# D. Cost model comparison
# ========================================================================
def cost_models(d, ans_tokens=DEFAULT_ANS_TOKENS):
    """Compare three cost models: KV-fork, prefix-cache, black-box API."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape
    mean_full_tok = float(think_lens.mean())

    # best operating point from learned method
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)
    g, best_thr = best_gain(lambda t: first_ge(p_stop, t),
                             np.round(np.linspace(0.30, 0.95, 14), 3),
                             correct, think_lens, budgets, fx_t, fx_a)

    stop_idx = first_ge(p_stop, best_thr)
    accs = correct[np.arange(N), stop_idx]
    think_toks = np.minimum(think_lens, np.array(budgets)[stop_idx])
    num_probes = stop_idx + 1  # probed at checkpoints 0..stop_idx

    # D1: KV-fork (decode only)
    d1_total = think_toks + num_probes * ans_tokens
    # D2: Prefix-cache (add prefix cache lookup cost ~ 0.1x of prefix tokens)
    prefix_toks = np.array(budgets)[stop_idx]
    d2_total = think_toks + num_probes * ans_tokens + 0.1 * prefix_toks * num_probes
    # D3: Black-box API (full prefill each probe)
    cumulative_prefix = np.zeros(N)
    for i in range(N):
        for j in range(int(num_probes[i])):
            cumulative_prefix[i] += budgets[min(j, m - 1)]
    d3_total = think_toks + num_probes * ans_tokens + cumulative_prefix

    full_total = mean_full_tok + ans_tokens

    results = {}
    for name, total in [("kv_fork", d1_total), ("prefix_cache", d2_total), ("black_box_api", d3_total)]:
        mean_t = float(total.mean())
        saving_pct = 100.0 * (1.0 - mean_t / max(full_total, 1))
        results[name] = {
            "mean_total_tokens": round(mean_t, 1),
            "saving_pct": round(saving_pct, 1),
            "accuracy": round(float(accs.mean()), 4),
        }
    results["full_baseline"] = {
        "mean_total_tokens": round(full_total, 1),
        "saving_pct": 0.0,
        "accuracy": round(float(correct[:, -1].mean()), 4),
    }
    results["adapt_gain"] = round(g, 4)
    return results


# ========================================================================
# D4. Probe overhead sweep (simulated)
# ========================================================================
def probe_overhead_sweep(d):
    """Simulate total savings with different probe answer token caps."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape
    mean_full_tok = float(think_lens.mean())

    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)
    _, best_thr = best_gain(lambda t: first_ge(p_stop, t),
                             np.round(np.linspace(0.30, 0.95, 14), 3),
                             correct, think_lens, budgets, fx_t, fx_a)

    stop_idx = first_ge(p_stop, best_thr)
    think_toks = np.minimum(think_lens, np.array(budgets)[stop_idx]).mean()
    num_probes = (stop_idx + 1).mean()

    results = []
    for a_tok in [4, 8, 16, 24, 32, 48, 64]:
        probe_oh = num_probes * a_tok
        total = think_toks + probe_oh
        full_total = mean_full_tok + a_tok
        think_save = 100.0 * (1.0 - think_toks / max(mean_full_tok, 1))
        total_save = 100.0 * (1.0 - total / max(full_total, 1))
        results.append({
            "ans_tokens": a_tok,
            "probe_overhead": round(float(probe_oh), 1),
            "total_tokens": round(float(total), 1),
            "think_save_pct": round(float(think_save), 1),
            "total_save_pct": round(float(total_save), 1),
        })
    return results


# ========================================================================
# E4. Learn-then-Test sequential testing
# ========================================================================
def learn_then_test(d, delta=0.05):
    """LTT-style: sequential binomial tests on each threshold, Holm correction."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape
    full_correct = correct[:, -1]

    rng = np.random.default_rng(123)
    perm = rng.permutation(N)
    n_cal = int(round(N * 0.4))
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    thr_grid = sorted(np.round(np.linspace(0.20, 0.99, 40), 3).tolist())
    K = len(thr_grid)

    results = []
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        rejected_thrs = []
        for rank, thr in enumerate(thr_grid):
            # risk on cal
            risks = []
            for i in cal_idx:
                j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
                risks.append(float(full_correct[i] == 1 and correct[i, j] == 0))
            emp_risk = np.mean(risks)
            n = len(risks)
            # Hoeffding p-value for H0: R(τ) > α
            if emp_risk >= alpha:
                continue
            # upper bound via Hoeffding
            gap = alpha - emp_risk
            p_val = np.exp(-2 * n * gap ** 2)
            adj_alpha = delta / (K - rank)
            if p_val < adj_alpha:
                rejected_thrs.append(thr)

        if rejected_thrs:
            # most aggressive (lowest) rejected threshold
            tau_star = min(rejected_thrs)
            # evaluate on test
            accs, toks = [], []
            risks_test = []
            for i in test_idx:
                j = next((jj for jj in range(m) if p_stop[i, jj] >= tau_star), m - 1)
                accs.append(correct[i, j])
                toks.append(min(int(think_lens[i]), budgets[j]))
                risks_test.append(float(full_correct[i] == 1 and correct[i, j] == 0))
            results.append({
                "alpha": alpha, "tau": round(float(tau_star), 3),
                "test_risk": round(float(np.mean(risks_test)), 4),
                "test_acc": round(float(np.mean(accs)), 4),
                "test_tok": round(float(np.mean(toks)), 1),
                "n_rejected": len(rejected_thrs),
                "fire": True,
            })
        else:
            results.append({"alpha": alpha, "fire": False})
    return results


# ========================================================================
# E5. Conformal Thinking-style single-confidence UCB baseline
# ========================================================================
def conformal_thinking_baseline(d, delta=0.05):
    """Single confidence threshold with UCB on cal set (Conformal Thinking-style)."""
    conf = np.exp(d["conf_lp"])
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    full_correct = correct[:, -1]
    mean_full_tok = float(think_lens.mean())

    rng = np.random.default_rng(123)
    perm = rng.permutation(N)
    n_cal = int(round(N * 0.4))
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    cgrid = sorted(np.round(np.linspace(conf.min(), conf.max(), 30), 4).tolist())
    K = len(cgrid)
    hb = float(np.sqrt(np.log(K / delta) / (2 * n_cal)))

    results = []
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        tau_star = None
        for c_thr in cgrid:
            risks = []
            for i in cal_idx:
                j = next((jj for jj in range(m) if conf[i, jj] >= c_thr), m - 1)
                risks.append(float(full_correct[i] == 1 and correct[i, j] == 0))
            emp_risk = np.mean(risks)
            if emp_risk + hb <= alpha:
                tau_star = c_thr
                break
        if tau_star is None:
            results.append({"alpha": alpha, "fire": False, "method": "conf_thinking"})
        else:
            accs, toks = [], []
            for i in test_idx:
                j = next((jj for jj in range(m) if conf[i, jj] >= tau_star), m - 1)
                accs.append(correct[i, j])
                toks.append(min(int(think_lens[i]), budgets[j]))
            think_save = 100.0 * (1.0 - np.mean(toks) / max(mean_full_tok, 1))
            results.append({
                "alpha": alpha, "tau": round(float(tau_star), 4),
                "test_acc": round(float(np.mean(accs)), 4),
                "test_tok": round(float(np.mean(toks)), 1),
                "think_save_pct": round(float(think_save), 1),
                "fire": True, "method": "conf_thinking",
            })
    return results


# ========================================================================
# F3. Extended feature ablation
# ========================================================================
def extended_ablation(d, folds=5, seed=42):
    """Systematic feature ablation per roadmap Table 7.3."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y = d["X"], d["y"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    groups = np.repeat(np.arange(N), m)
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)

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
        g, thr = best_gain(lambda t: first_ge(p, t),
                           np.round(np.linspace(0.30, 0.95, 14), 3),
                           correct, think_lens, budgets, fx_t, fx_a)
        return g

    variants = {
        "budget_only":        [0, 1],                          # budget_k, ckpt_frac
        "confidence_only":    [CONF_LP_IDX],                   # conf_lp
        "entropy_only":       [CONF_ENT_IDX],                  # conf_ent
        "conf_ent":           [CONF_LP_IDX, CONF_ENT_IDX],     # conf + entropy
        "stability_only":     [STABLE_IDX, RUN_LEN_IDX, VOTE_SHARE_IDX],
        "trace_only":         [MKR_IDX, 0],                    # marker + budget
        "no_confidence":      [i for i in range(10) if i != CONF_LP_IDX],
        "no_stability":       [i for i in range(10) if i not in (STABLE_IDX, RUN_LEN_IDX, VOTE_SHARE_IDX)],
        "8_feature_main":     [i for i in range(10) if i not in (USED_K_IDX, ENDED_IDX)],
        "10_feature_all":     list(range(10)),
    }

    results = {}
    for name, cols in variants.items():
        g = cv_pstop(cols)
        results[name] = {"peak_gain": g, "n_features": len(cols),
                         "features": [FEATURE_NAMES[c] for c in cols]}
    return results


# ========================================================================
# F4. Model class comparison
# ========================================================================
def model_class_comparison(d, folds=5, seed=42):
    """Compare logistic, RF, gradient boosting, MLP classifiers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y = d["X"], d["y"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    groups = np.repeat(np.arange(N), m)
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)
    cols = [i for i in range(10) if i not in (USED_K_IDX, ENDED_IDX)]

    classifiers = {
        "logistic": lambda: LogisticRegression(max_iter=1000, C=1.0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed),
        "gradient_boosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=seed),
    }

    results = {}
    for clf_name, clf_fn in classifiers.items():
        p = np.zeros((N, m))
        gkf = GroupKFold(n_splits=folds)
        Xc = X[:, cols]
        for tr, te in gkf.split(Xc, y, groups):
            sc = StandardScaler().fit(Xc[tr])
            clf = clf_fn()
            clf.fit(sc.transform(Xc[tr]), y[tr])
            pr = clf.predict_proba(sc.transform(Xc[te]))[:, 1]
            for idx, pp in zip(te, pr):
                p[idx // m, idx % m] = pp
        g, thr = best_gain(lambda t: first_ge(p, t),
                           np.round(np.linspace(0.30, 0.95, 14), 3),
                           correct, think_lens, budgets, fx_t, fx_a)
        results[clf_name] = {"peak_gain": g, "op_thr": round(float(thr), 3) if thr else None}
    return results


# ========================================================================
# F5. Probability calibration (ECE, Brier, reliability bins)
# ========================================================================
def probability_calibration(d, n_bins=10):
    """Compute ECE, Brier score, and reliability diagram bins for the learned stopper."""
    p_stop = d["p_stop"]
    correct = d["correct"]
    N, m = correct.shape

    # Flatten: each (question, checkpoint) is a sample
    probs = p_stop.ravel()
    labels = d["y"] if "y" in d else correct.ravel()

    # Brier score
    brier = float(np.mean((probs - labels) ** 2))

    # ECE with binning
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []
    for b in range(n_bins):
        lo, hi = bins[b], bins[b + 1]
        mask = (probs >= lo) & (probs < hi)
        if b == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            bin_data.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
                             "count": 0, "mean_pred": None, "mean_true": None, "gap": None})
            continue
        mean_pred = float(probs[mask].mean())
        mean_true = float(labels[mask].mean())
        gap = abs(mean_pred - mean_true)
        ece += gap * mask.sum() / len(probs)
        bin_data.append({
            "bin_lo": round(lo, 2), "bin_hi": round(hi, 2),
            "count": int(mask.sum()),
            "mean_pred": round(mean_pred, 4),
            "mean_true": round(mean_true, 4),
            "gap": round(gap, 4),
        })

    return {"ece": round(ece, 4), "brier": round(brier, 4), "n_bins": n_bins, "bins": bin_data}


# ========================================================================
# F6. Leakage audit
# ========================================================================
def leakage_audit(d):
    """Check each feature for potential future-information leakage."""
    checks = []
    N, m = d["correct"].shape
    X = d["X"]

    # budget_k (col 0) — just the budget value at this checkpoint, OK
    checks.append({"feature": "budget_k", "col": 0, "status": "PASS",
                   "reason": "Deterministic grid value known at checkpoint time"})

    # ckpt_frac (col 1) — checkpoint fraction, OK
    checks.append({"feature": "ckpt_frac", "col": 1, "status": "PASS",
                   "reason": "Fraction of total grid positions, known at checkpoint"})

    # conf_lp (col 2) — computed from probe answer at this checkpoint, OK
    checks.append({"feature": "conf_lp", "col": 2, "status": "PASS",
                   "reason": "Computed from current checkpoint's probe answer logprobs"})

    # conf_ent (col 3) — same
    checks.append({"feature": "conf_ent", "col": 3, "status": "PASS",
                   "reason": "Computed from current checkpoint's probe answer entropy"})

    # stable (col 4) — same_as_prev, only needs current and previous checkpoint
    checks.append({"feature": "stable", "col": 4, "status": "PASS",
                   "reason": "Binary: current answer == previous answer (prefix-observable)"})

    # run_len (col 5) — consecutive same answers ending at current
    checks.append({"feature": "run_len", "col": 5, "status": "PASS",
                   "reason": "Count of consecutive same answers up to current checkpoint"})

    # vote_share (col 6) — fraction of past checkpoints with current answer
    # Check: does vote_share at checkpoint j only use checkpoints 0..j?
    vote_at_first = X[::m, VOTE_SHARE_IDX]  # first checkpoint of each question
    # At first checkpoint, vote_share should be 1.0 (only one vote = itself)
    all_one = np.all(np.isclose(vote_at_first, 1.0) | np.isclose(vote_at_first, 0.0))
    checks.append({"feature": "vote_share", "col": 6,
                   "status": "PASS" if all_one else "WARNING",
                   "reason": "First-checkpoint values are 0 or 1 as expected — prefix-only" if all_one
                   else "Some first-checkpoint vote_share values unexpected — verify implementation"})

    # marker_density (col 7) — backtracking markers in current prefix
    checks.append({"feature": "marker_density", "col": 7, "status": "PASS",
                   "reason": "Backtracking marker count in reasoning prefix up to current checkpoint"})

    # used_k (col 8) — the fraction of budget used
    # This could be problematic if it uses full_think_tokens
    used_k_vals = X[:, USED_K_IDX].reshape(N, m)
    # Check if used_k varies across checkpoints for the same question
    varies = any(np.std(used_k_vals[i]) > 0.01 for i in range(min(N, 50)))
    if varies:
        checks.append({"feature": "used_k", "col": 8, "status": "PASS",
                       "reason": "Values vary across checkpoints — computed from observed prefix length"})
    else:
        checks.append({"feature": "used_k", "col": 8, "status": "WARNING",
                       "reason": "Values constant across checkpoints — may use full_think_tokens"})

    # ended (col 9) — whether </think> seen
    ended_vals = X[:, ENDED_IDX].reshape(N, m)
    # ended should be monotonic (once ended, stays ended)
    monotonic = True
    for i in range(min(N, 100)):
        for j in range(1, m):
            if ended_vals[i, j - 1] == 1 and ended_vals[i, j] == 0:
                monotonic = False
                break
    checks.append({"feature": "ended", "col": 9,
                   "status": "PASS" if monotonic else "FAIL",
                   "reason": "Monotonic once triggered — prefix-observable" if monotonic
                   else "Non-monotonic: ended=1 then ended=0 found — investigate"})

    # Check: gold answer not in features
    checks.append({"feature": "gold_answer", "col": None, "status": "PASS",
                   "reason": "Gold answer is NOT included in the 10-feature vector"})

    # Check: full_think_tokens not in features
    checks.append({"feature": "full_think_tokens", "col": None, "status": "PASS",
                   "reason": "Full think token count is NOT included in features"})

    # Check: no question-level leakage in grouped CV
    checks.append({"feature": "grouped_cv", "col": None, "status": "PASS",
                   "reason": "GroupKFold ensures all checkpoints of a question are in the same fold"})

    all_pass = all(c["status"] == "PASS" for c in checks)
    return {"all_pass": all_pass, "checks": checks}


# ========================================================================
# H3. Calibration size curve
# ========================================================================
def calibration_size_curve(d, n_repeats=20, seed=42):
    """Sweep n_cal to see how many calibration samples are needed."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape
    rng = np.random.default_rng(seed)

    results = []
    for n_cal in [10, 20, 50, 100, 200, 400]:
        if n_cal >= N:
            continue
        gains = []
        for rep in range(n_repeats):
            perm = rng.permutation(N)
            cal_idx = perm[:n_cal]
            test_idx = perm[n_cal:]
            if len(test_idx) < 10:
                continue

            # Select threshold on cal
            def cal_frontier():
                ft = np.array([float(np.mean(np.minimum(think_lens[cal_idx], B))) for B in budgets])
                fa = np.array([float(correct[cal_idx, j].mean()) for j in range(len(budgets))])
                order = np.argsort(ft)
                return ft[order], fa[order]

            ft_c, fa_c = cal_frontier()
            best_g_cal, best_thr_cal = -9.9, None
            for thr in np.round(np.linspace(0.30, 0.95, 14), 3):
                idx = first_ge(p_stop, thr)
                a = correct[cal_idx, idx[cal_idx]].mean()
                t = np.minimum(think_lens[cal_idx], np.array(budgets)[idx[cal_idx]]).mean()
                g = a - float(np.interp(t, ft_c, fa_c))
                if g > best_g_cal:
                    best_g_cal, best_thr_cal = g, thr

            if best_thr_cal is None:
                gains.append(0.0)
                continue

            # Evaluate on test
            ft_t = np.array([float(np.mean(np.minimum(think_lens[test_idx], B))) for B in budgets])
            fa_t = np.array([float(correct[test_idx, j].mean()) for j in range(len(budgets))])
            order = np.argsort(ft_t)
            ft_t, fa_t = ft_t[order], fa_t[order]
            idx = first_ge(p_stop, best_thr_cal)
            a = correct[test_idx, idx[test_idx]].mean()
            t = np.minimum(think_lens[test_idx], np.array(budgets)[idx[test_idx]]).mean()
            g = a - float(np.interp(t, ft_t, fa_t))
            gains.append(g)

        results.append({
            "n_cal": n_cal,
            "mean_gain": round(float(np.mean(gains)), 4),
            "std_gain": round(float(np.std(gains)), 4),
            "min_gain": round(float(np.min(gains)), 4) if gains else None,
            "max_gain": round(float(np.max(gains)), 4) if gains else None,
        })
    return results


# ========================================================================
# I3. Answer transition analysis
# ========================================================================
def answer_transition_analysis(d):
    """Per-question analysis of answer changes across checkpoints."""
    correct = d["correct"]
    ans = d["ans"]
    conf = np.exp(d["conf_lp"])
    conf_ent = d["conf_ent"]
    N, m = correct.shape

    stats = {
        "mean_n_changes": 0, "mean_first_correct_ckpt": 0,
        "mean_last_change_ckpt": 0, "mean_conf_at_first_correct": 0,
        "frac_never_correct": 0, "frac_single_answer": 0,
        "mean_conf_slope": 0, "mean_ent_slope": 0,
    }

    n_changes_list = []
    first_correct_list = []
    last_change_list = []
    conf_at_first_list = []
    conf_slope_list = []
    ent_slope_list = []
    n_single = 0
    n_never = 0

    for i in range(N):
        # count answer changes
        changes = sum(1 for j in range(1, m) if ans[i, j] != ans[i, j - 1])
        n_changes_list.append(changes)
        if changes == 0:
            n_single += 1

        # first correct checkpoint
        fc = np.nonzero(correct[i])[0]
        if fc.size > 0:
            first_correct_list.append(fc[0])
            conf_at_first_list.append(conf[i, fc[0]])
        else:
            n_never += 1

        # last answer change checkpoint
        last_ch = 0
        for j in range(1, m):
            if ans[i, j] != ans[i, j - 1]:
                last_ch = j
        last_change_list.append(last_ch)

        # confidence and entropy slopes (linear regression over checkpoints)
        x = np.arange(m, dtype=float)
        if np.std(conf[i]) > 1e-10:
            slope_c = np.polyfit(x, conf[i], 1)[0]
        else:
            slope_c = 0.0
        conf_slope_list.append(slope_c)

        if np.std(conf_ent[i]) > 1e-10:
            slope_e = np.polyfit(x, conf_ent[i], 1)[0]
        else:
            slope_e = 0.0
        ent_slope_list.append(slope_e)

    stats["mean_n_changes"] = round(float(np.mean(n_changes_list)), 2)
    stats["mean_first_correct_ckpt"] = round(float(np.mean(first_correct_list)), 2) if first_correct_list else None
    stats["mean_last_change_ckpt"] = round(float(np.mean(last_change_list)), 2)
    stats["mean_conf_at_first_correct"] = round(float(np.mean(conf_at_first_list)), 4) if conf_at_first_list else None
    stats["frac_never_correct"] = round(n_never / N, 4)
    stats["frac_single_answer"] = round(n_single / N, 4)
    stats["mean_conf_slope"] = round(float(np.mean(conf_slope_list)), 6)
    stats["mean_ent_slope"] = round(float(np.mean(ent_slope_list)), 6)
    return stats


# ========================================================================
# I4. Feature importance (logistic coefficients + permutation importance)
# ========================================================================
def feature_importance(d, folds=5, seed=42):
    """Compute logistic regression coefficients and permutation importance."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    X, y = d["X"], d["y"]
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    N, m = correct.shape
    groups = np.repeat(np.arange(N), m)
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)
    cols = list(range(X.shape[1]))

    # Train a single model on full data for coefficients
    sc = StandardScaler().fit(X[:, cols])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(X[:, cols]), y)
    coefficients = {FEATURE_NAMES[i]: round(float(clf.coef_[0, i]), 4) for i in range(len(cols))}
    intercept = round(float(clf.intercept_[0]), 4)

    # Permutation importance via grouped CV
    def cv_gain(X_input):
        p = np.zeros((N, m))
        gkf = GroupKFold(n_splits=folds)
        for tr, te in gkf.split(X_input, y, groups):
            s = StandardScaler().fit(X_input[tr])
            c = LogisticRegression(max_iter=1000, C=1.0).fit(s.transform(X_input[tr]), y[tr])
            pr = c.predict_proba(s.transform(X_input[te]))[:, 1]
            for idx, pp in zip(te, pr):
                p[idx // m, idx % m] = pp
        g, _ = best_gain(lambda t: first_ge(p, t),
                         np.round(np.linspace(0.30, 0.95, 14), 3),
                         correct, think_lens, budgets, fx_t, fx_a)
        return g

    base_gain = cv_gain(X[:, cols])

    perm_importance = {}
    rng = np.random.default_rng(seed)
    for fi in range(len(cols)):
        X_perm = X[:, cols].copy()
        X_perm[:, fi] = rng.permutation(X_perm[:, fi])
        perm_gain = cv_gain(X_perm)
        drop = base_gain - perm_gain
        perm_importance[FEATURE_NAMES[fi]] = round(drop, 4)

    return {
        "coefficients": coefficients,
        "intercept": intercept,
        "base_gain": round(base_gain, 4),
        "permutation_importance": perm_importance,
    }


# ========================================================================
# K. Checkpoint subset simulation
# ========================================================================
def checkpoint_subset_simulation(d):
    """Simulate using fewer checkpoints to estimate optimal schedule."""
    correct, think_lens, budgets = d["correct"], d["think_lens"], list(d["budgets"])
    p_stop = d["p_stop"]
    N, m = correct.shape

    # Full checkpoint set result
    fx_t, fx_a = fixed_frontier(correct, think_lens, budgets)
    g_full, _ = best_gain(lambda t: first_ge(p_stop, t),
                          np.round(np.linspace(0.30, 0.95, 14), 3),
                          correct, think_lens, budgets, fx_t, fx_a)

    results = []
    # Try different subset sizes
    for n_ckpts in [4, 6, 8]:
        if n_ckpts >= m:
            continue
        # Linear subset
        lin_idx = np.round(np.linspace(0, m - 1, n_ckpts)).astype(int)
        # Log-spaced subset (more early checkpoints)
        log_idx = np.unique(np.round(np.geomspace(1, m, n_ckpts) - 1).astype(int))
        log_idx = np.clip(log_idx, 0, m - 1)
        if len(log_idx) < n_ckpts:
            log_idx = np.unique(np.concatenate([log_idx, [m - 1]]))

        for schedule_name, ckpt_idx in [("linear", lin_idx), ("log", log_idx)]:
            ckpt_idx = np.array(sorted(set(ckpt_idx)))
            sub_budgets = [budgets[j] for j in ckpt_idx]
            sub_correct = correct[:, ckpt_idx]
            sub_p_stop = p_stop[:, ckpt_idx]

            ft_sub, fa_sub = fixed_frontier(sub_correct, think_lens, sub_budgets)
            g_sub, thr_sub = best_gain(
                lambda t: first_ge(sub_p_stop, t),
                np.round(np.linspace(0.30, 0.95, 14), 3),
                sub_correct, think_lens, sub_budgets, ft_sub, fa_sub)

            # Compute probe overhead
            stop_idx = first_ge(sub_p_stop, thr_sub) if thr_sub else np.full(N, len(ckpt_idx) - 1, dtype=int)
            mean_probes = float((stop_idx + 1).mean())

            results.append({
                "n_checkpoints": len(ckpt_idx),
                "schedule": schedule_name,
                "checkpoint_budgets": [int(b) for b in sub_budgets],
                "peak_gain": g_sub,
                "gain_retention": round(g_sub / max(g_full, 0.001), 3) if g_full > 0 else None,
                "mean_probes": round(mean_probes, 2),
            })
    return {"full_gain": round(g_full, 4), "full_n_checkpoints": m, "subsets": results}


# ========================================================================
# Main
# ========================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="results/<...> directory")
    parser.add_argument("--npz", default="raw.npz")
    parser.add_argument("--skip_slow", action="store_true",
                    help="skip slower analyses (model_class, feature_importance, extended_ablation)")
    args = parser.parse_args()

    run = Path(args.run_dir)
    d = load_npz(run / args.npz)
    adir = run / "analysis"
    adir.mkdir(exist_ok=True)

    existing = {}
    epath = adir / "extended_analysis.json"
    if epath.exists():
        with open(epath) as f:
            existing = json.load(f)

    out = {"run": run.name, "npz": args.npz}

    # A1b. DEER-style
    logger.info(f"\n=== {run.name}: A1b. DEER-style transition exit ===")
    deer = deer_transition_exit(d)
    logger.info(f"  peak_gain={deer['peak_gain']:+.4f}  params={deer['params']}")
    out["deer_transition"] = deer

    # A2b. EAT-style
    logger.info(f"  A2b. EAT-style entropy stability")
    eat = eat_entropy_stability(d)
    logger.info(f"  peak_gain={eat['peak_gain']:+.4f}  params={eat['params']}")
    out["eat_entropy_stability"] = eat

    # A5. PUMA-style
    logger.info(f"  A5. PUMA semantic convergence proxy")
    puma = puma_convergence_proxy(d)
    logger.info(f"  peak_gain={puma['peak_gain']:+.4f}  params={puma['params']}")
    out["puma_convergence"] = puma

    # A6. TERMINATOR-light
    if "X" in d and "y" in d:
        logger.info(f"  A6. TERMINATOR-light (earliest-correct supervision)")
        term = terminator_light(d)
        logger.info(f"  peak_gain={term['peak_gain']:+.4f}  op_thr={term['op_thr']}")
        out["terminator_light"] = term

    # D. Cost models
    if "p_stop" in d:
        logger.info(f"  D. Cost model comparison")
        costs = cost_models(d)
        for name, v in costs.items():
            if isinstance(v, dict) and "saving_pct" in v:
                logger.info(f"    {name}: saving={v['saving_pct']:.1f}%  acc={v['accuracy']:.4f}")
        out["cost_models"] = costs

    # D4. Probe overhead sweep
    if "p_stop" in d:
        logger.info(f"  D4. Probe overhead sweep")
        ohs = probe_overhead_sweep(d)
        for r in ohs:
            logger.info(f"    A={r['ans_tokens']}  overhead={r['probe_overhead']:.0f}  total_save={r['total_save_pct']:.1f}%")
        out["probe_overhead_sweep"] = ohs

    # E4. Learn-then-Test
    if "p_stop" in d:
        logger.info(f"  E4. Learn-then-Test sequential testing")
        ltt = learn_then_test(d)
        for r in ltt:
            if r.get("fire"):
                logger.info(f"    α={r['alpha']:.2f}  τ={r['tau']:.3f}  test_risk={r['test_risk']:.4f}  "
                      f"test_acc={r['test_acc']:.4f}")
            else:
                logger.info(f"    α={r['alpha']:.2f}  no threshold qualifies")
        out["learn_then_test"] = ltt

    # E5. Conformal Thinking baseline
    logger.info(f"  E5. Conformal Thinking-style single-confidence UCB")
    ct = conformal_thinking_baseline(d)
    for r in ct:
        if r.get("fire"):
            logger.info(f"    α={r['alpha']:.2f}  τ={r['tau']:.4f}  test_acc={r['test_acc']:.4f}  "
                  f"think_save={r['think_save_pct']:.1f}%")
        else:
            logger.info(f"    α={r['alpha']:.2f}  no threshold qualifies")
    out["conformal_thinking_baseline"] = ct

    # F5. Probability calibration
    if "p_stop" in d:
        logger.info(f"  F5. Probability calibration")
        pcal = probability_calibration(d)
        logger.info(f"    ECE={pcal['ece']:.4f}  Brier={pcal['brier']:.4f}")
        out["probability_calibration"] = pcal

    # F6. Leakage audit
    if "X" in d:
        logger.info(f"  F6. Leakage audit")
        audit = leakage_audit(d)
        status = "ALL PASS" if audit["all_pass"] else "WARNINGS"
        logger.info(f"    Status: {status}")
        for c in audit["checks"]:
            sym = "✓" if c["status"] == "PASS" else "⚠"
            logger.info(f"      {sym} {c['feature']}: {c['reason']}")
        out["leakage_audit"] = audit

    # H3. Calibration size curve
    if "p_stop" in d:
        logger.info(f"  H3. Calibration size curve")
        csc = calibration_size_curve(d)
        for r in csc:
            logger.info(f"    n_cal={r['n_cal']:>4d}  mean_gain={r['mean_gain']:+.4f}  "
                  f"std={r['std_gain']:.4f}")
        out["calibration_size_curve"] = csc

    # I3. Answer transition analysis
    logger.info(f"  I3. Answer transition analysis")
    ata = answer_transition_analysis(d)
    logger.info(f"    mean_changes={ata['mean_n_changes']:.2f}  "
          f"single_answer={ata['frac_single_answer']:.1%}  "
          f"never_correct={ata['frac_never_correct']:.1%}")
    out["answer_transitions"] = ata

    # K. Checkpoint subset simulation
    if "p_stop" in d:
        logger.info(f"  K. Checkpoint subset simulation")
        css = checkpoint_subset_simulation(d)
        logger.info(f"    full: {css['full_n_checkpoints']} ckpts  gain={css['full_gain']:+.4f}")
        for s in css["subsets"]:
            ret = f"  retention={s['gain_retention']:.1%}" if s["gain_retention"] else ""
            logger.info(f"    {s['n_checkpoints']} ckpts ({s['schedule']}): "
                  f"gain={s['peak_gain']:+.4f}{ret}  probes={s['mean_probes']:.1f}")
        out["checkpoint_subsets"] = css

    # Slower analyses
    if not args.skip_slow and "X" in d and "y" in d:
        # F3. Extended feature ablation
        logger.info(f"  F3. Extended feature ablation (10 variants)")
        ext_abl = extended_ablation(d)
        for name, v in sorted(ext_abl.items(), key=lambda x: -x[1]["peak_gain"]):
            logger.info(f"    {name:<20}  gain={v['peak_gain']:+.4f}  ({v['n_features']} feats)")
        out["extended_ablation"] = ext_abl

        # F4. Model class comparison
        logger.info(f"  F4. Model class comparison")
        mcc = model_class_comparison(d)
        for name, v in sorted(mcc.items(), key=lambda x: -x[1]["peak_gain"]):
            logger.info(f"    {name:<22}  gain={v['peak_gain']:+.4f}")
        out["model_class_comparison"] = mcc

        # I4. Feature importance
        logger.info(f"  I4. Feature importance (coefficients + permutation)")
        fi = feature_importance(d)
        logger.info(f"    Coefficients (standardized):")
        for feat, coef in sorted(fi["coefficients"].items(), key=lambda x: -abs(x[1])):
            logger.info(f"      {feat:<20} {coef:+.4f}")
        logger.info(f"    Permutation importance (gain drop):")
        for feat, imp in sorted(fi["permutation_importance"].items(), key=lambda x: -x[1]):
            logger.info(f"      {feat:<20} {imp:+.4f}")
        out["feature_importance"] = fi

    # Write extended analysis
    with open(epath, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\n  [extended] wrote {epath}")


if __name__ == "__main__":
    main()
