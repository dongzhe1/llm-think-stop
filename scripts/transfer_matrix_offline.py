"""Offline cross-task AND cross-model transfer matrix from existing raw.npz files.

Trains a stop-classifier on SOURCE raw.npz and evaluates on TARGET raw.npz,
WITHOUT any GPU re-run. Supports:
  - Cross-task transfer (same model): e.g. GSM8K → MMLU-Pro
  - Cross-model transfer (same task): e.g. Qwen3-8B → Qwen3-32B on GSM8K
  - Three protocols per pair:
      1. Zero-shot: source threshold applied directly to target (no calibration)
      2. Target-calibrated: source classifier, threshold selected on target cal set
      3. Target-trained upper bound: classifier trained + threshold selected on target

Usage:
  python scripts/transfer_matrix_offline.py \
      --source results/learnstop/gsm8k_Qwen3-8B_n1000_1861180 \
      --target results/learnstop/math500_Qwen3-8B_n500_1861183 \
      --label "GSM8K_8B -> MATH500_8B"

  python scripts/transfer_matrix_offline.py --all   # run full matrix
"""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

USED_K_IDX = 8
ENDED_IDX = 9


def load_npz(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def fixed_frontier(correct, think_lens, budgets):
    ft = np.array([float(np.mean(np.minimum(think_lens, B))) for B in budgets])
    fa = np.array([float(correct[:, j].mean()) for j in range(len(budgets))])
    order = np.argsort(ft)
    return ft[order], fa[order]


def first_ge(scores, thr):
    m = scores.shape[1]
    out = np.full(scores.shape[0], m - 1, dtype=int)
    for i in range(scores.shape[0]):
        hit = np.nonzero(scores[i] >= thr)[0]
        if hit.size:
            out[i] = hit[0]
    return out


def simulate(stop_idx, correct, think_lens, budgets):
    N = correct.shape[0]
    accs = correct[np.arange(N), stop_idx]
    toks = np.minimum(think_lens, np.array(budgets)[stop_idx])
    return float(accs.mean()), float(toks.mean())


def best_gain(p_stop, correct, think_lens, budgets, fx_t, fx_a):
    sweep = np.round(np.linspace(0.30, 0.95, 14), 3)
    best_g, best_thr = -9.9, None
    for thr in sweep:
        idx = first_ge(p_stop, thr)
        a, t = simulate(idx, correct, think_lens, budgets)
        fa = float(np.interp(t, fx_t, fx_a))
        g = a - fa
        if g > best_g:
            best_g, best_thr = g, thr
    return round(best_g, 4), best_thr


def run_transfer(source_path, target_path, label="", folds=5, seed=42):
    """Run three transfer protocols from source to target."""
    src = load_npz(source_path / "raw.npz")
    tgt = load_npz(target_path / "raw.npz")

    cols = [i for i in range(src["X"].shape[1]) if i not in (USED_K_IDX, ENDED_IDX)]

    N_s, m = src["correct"].shape
    N_t = tgt["correct"].shape[0]
    budgets_s = list(src["budgets"])
    budgets_t = list(tgt["budgets"])

    sc = StandardScaler().fit(src["X"][:, cols])
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(
        sc.transform(src["X"][:, cols]), src["y"])

    # Source p_stop (from grouped CV for fair eval)
    groups_s = np.repeat(np.arange(N_s), m)
    p_stop_source = np.zeros((N_s, m))
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(src["X"][:, cols], src["y"], groups_s):
        s = StandardScaler().fit(src["X"][tr][:, cols])
        c = LogisticRegression(max_iter=1000, C=1.0).fit(
            s.transform(src["X"][tr][:, cols]), src["y"][tr])
        pr = c.predict_proba(s.transform(src["X"][te][:, cols]))[:, 1]
        for idx, pp in zip(te, pr):
            p_stop_source[idx // m, idx % m] = pp

    # Target p_stop (apply source classifier)
    p_stop_transfer = clf.predict_proba(sc.transform(tgt["X"][:, cols]))[:, 1].reshape(N_t, m)

    # Target p_stop (train on target for upper bound)
    groups_t = np.repeat(np.arange(N_t), m)
    p_stop_target_trained = np.zeros((N_t, m))
    gkf_t = GroupKFold(n_splits=min(folds, N_t))
    for tr, te in gkf_t.split(tgt["X"][:, cols], tgt["y"], groups_t):
        s = StandardScaler().fit(tgt["X"][tr][:, cols])
        c = LogisticRegression(max_iter=1000, C=1.0).fit(
            s.transform(tgt["X"][tr][:, cols]), tgt["y"][tr])
        pr = c.predict_proba(s.transform(tgt["X"][te][:, cols]))[:, 1]
        for idx, pp in zip(te, pr):
            p_stop_target_trained[idx // m, idx % m] = pp

    fx_t_s, fx_a_s = fixed_frontier(src["correct"], src["think_lens"], budgets_s)
    g_indist, thr_indist = best_gain(p_stop_source, src["correct"],
                                      src["think_lens"], budgets_s, fx_t_s, fx_a_s)

    fx_t_t, fx_a_t = fixed_frontier(tgt["correct"], tgt["think_lens"], budgets_t)

    # Protocol 1: Zero-shot (source threshold, no target calibration)
    g_zeroshot, _ = None, None
    if thr_indist is not None:
        idx = first_ge(p_stop_transfer, thr_indist)
        a, t = simulate(idx, tgt["correct"], tgt["think_lens"], budgets_t)
        fa = float(np.interp(t, fx_t_t, fx_a_t))
        g_zeroshot = round(a - fa, 4)

    # Protocol 2: Target-calibrated (source classifier, threshold on target cal set)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N_t)
    n_cal = int(round(N_t * 0.4))
    cal_idx = perm[:n_cal]
    test_idx = perm[n_cal:]

    def best_on_subset(p_stop_mat, idx):
        ft, fa = fixed_frontier(tgt["correct"][idx], tgt["think_lens"][idx], budgets_t)
        best_g, best_thr = -9.9, None
        for thr in np.round(np.linspace(0.30, 0.95, 14), 3):
            sidx = first_ge(p_stop_mat, thr)
            a = tgt["correct"][idx, sidx[idx]].mean()
            t = np.minimum(tgt["think_lens"][idx], np.array(budgets_t)[sidx[idx]]).mean()
            g = a - float(np.interp(t, ft, fa))
            if g > best_g:
                best_g, best_thr = g, thr
        return best_g, best_thr

    def eval_on_test(p_stop_mat, thr, idx):
        ft, fa = fixed_frontier(tgt["correct"][idx], tgt["think_lens"][idx], budgets_t)
        sidx = first_ge(p_stop_mat, thr)
        a = tgt["correct"][idx, sidx[idx]].mean()
        t = np.minimum(tgt["think_lens"][idx], np.array(budgets_t)[sidx[idx]]).mean()
        g = a - float(np.interp(t, ft, fa))
        return round(g, 4), round(float(a), 4), round(float(t), 1)

    _, thr_cal = best_on_subset(p_stop_transfer, cal_idx)
    if thr_cal is not None:
        g_cal, acc_cal, tok_cal = eval_on_test(p_stop_transfer, thr_cal, test_idx)
    else:
        g_cal, acc_cal, tok_cal = None, None, None

    # Protocol 3: Target-trained upper bound
    _, thr_target = best_on_subset(p_stop_target_trained, cal_idx)
    if thr_target is not None:
        g_target, acc_target, tok_target = eval_on_test(p_stop_target_trained, thr_target, test_idx)
    else:
        g_target, acc_target, tok_target = None, None, None

    # Peak gains (post-hoc) for reference
    g_transfer_peak, _ = best_gain(p_stop_transfer, tgt["correct"], tgt["think_lens"],
                                    budgets_t, fx_t_t, fx_a_t)
    g_target_peak, _ = best_gain(p_stop_target_trained, tgt["correct"], tgt["think_lens"],
                                  budgets_t, fx_t_t, fx_a_t)

    result = {
        "label": label,
        "source": str(source_path.name),
        "target": str(target_path.name),
        "n_source": N_s, "n_target": N_t,
        "source_indist_gain": g_indist,
        "zero_shot_gain": g_zeroshot,
        "target_calibrated_gain": g_cal,
        "target_calibrated_acc": acc_cal,
        "target_calibrated_tok": tok_cal,
        "target_trained_gain": g_target,
        "target_trained_acc": acc_target,
        "target_trained_tok": tok_target,
        "transfer_peak_gain": g_transfer_peak,
        "target_peak_gain": g_target_peak,
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"Transfer: {label}")
    logger.info(f"  Source in-dist gain:     {g_indist:+.4f}")
    logger.info(f"  Zero-shot transfer:     {g_zeroshot:+.4f}" if g_zeroshot is not None else "  Zero-shot: n/a")
    logger.info(f"  Target-calibrated:      {g_cal:+.4f}" if g_cal is not None else "  Target-calibrated: n/a")
    logger.info(f"  Target-trained (UB):    {g_target:+.4f}" if g_target is not None else "  Target-trained: n/a")
    logger.info(f"  Transfer peak (posthoc):{g_transfer_peak:+.4f}")
    logger.info(f"{'='*60}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", help="source run directory")
    parser.add_argument("--target", help="target run directory")
    parser.add_argument("--label", default="")
    parser.add_argument("--all", action="store_true", help="run full transfer matrix")
    parser.add_argument("--out", default="results/transfer_matrix.json")
    args = parser.parse_args()

    if args.all:
        base = Path("results/learnstop")
        runs = {
            "gsm8k_8B": base / "gsm8k_Qwen3-8B_n1000_1861180",
            "gsm8k_32B": base / "gsm8k_Qwen3-32B_n1000_1861181",
            "math500_8B": base / "math500_Qwen3-8B_n500_1861183",
            "math500_32B": base / "math500_Qwen3-32B_n500_1861184",
            "mmlu_8B": base / "mmlu_pro_Qwen3-8B_n800_1861185",
            "mmlu_32B": base / "mmlu_pro_Qwen3-32B_n800_1861186",
            "aime_8B": base / "aime_Qwen3-8B_n90_1893901",
            "aime_32B": base / "aime_Qwen3-32B_n90_1884209",
            "gsm8k_dsr1_7b": base / "gsm8k_DeepSeek-R1-Distill-Qwen-7B_n500_1884043",
            "gsm8k_dsr1_llama": base / "gsm8k_DeepSeek-R1-Distill-Llama-8B_n500_1884044",
        }

        # Cross-task transfer pairs (same model)
        cross_task = [
            ("gsm8k_8B", "math500_8B", "GSM8K→MATH500 8B"),
            ("gsm8k_32B", "math500_32B", "GSM8K→MATH500 32B"),
            ("gsm8k_8B", "mmlu_8B", "GSM8K→MMLU-Pro 8B"),
            ("gsm8k_32B", "mmlu_32B", "GSM8K→MMLU-Pro 32B"),
            ("math500_8B", "gsm8k_8B", "MATH500→GSM8K 8B"),
            ("math500_32B", "gsm8k_32B", "MATH500→GSM8K 32B"),
            ("mmlu_8B", "gsm8k_8B", "MMLU-Pro→GSM8K 8B"),
            ("mmlu_32B", "gsm8k_32B", "MMLU-Pro→GSM8K 32B"),
            ("math500_8B", "mmlu_8B", "MATH500→MMLU-Pro 8B"),
            ("math500_32B", "mmlu_32B", "MATH500→MMLU-Pro 32B"),
            ("mmlu_8B", "math500_8B", "MMLU-Pro→MATH500 8B"),
            ("mmlu_32B", "math500_32B", "MMLU-Pro→MATH500 32B"),
            ("aime_8B", "gsm8k_8B", "AIME→GSM8K 8B"),
            ("aime_32B", "gsm8k_32B", "AIME→GSM8K 32B"),
            ("gsm8k_8B", "aime_8B", "GSM8K→AIME 8B"),
            ("gsm8k_32B", "aime_32B", "GSM8K→AIME 32B"),
        ]

        # Cross-model transfer (same task)
        cross_model = [
            ("gsm8k_8B", "gsm8k_32B", "Qwen3-8B→32B GSM8K"),
            ("gsm8k_32B", "gsm8k_8B", "Qwen3-32B→8B GSM8K"),
            ("gsm8k_8B", "gsm8k_dsr1_7b", "Qwen3-8B→DSR1-7B GSM8K"),
            ("gsm8k_8B", "gsm8k_dsr1_llama", "Qwen3-8B→DSR1-Llama GSM8K"),
            ("gsm8k_32B", "gsm8k_dsr1_7b", "Qwen3-32B→DSR1-7B GSM8K"),
        ]

        all_results = []
        for src_key, tgt_key, label in cross_task + cross_model:
            src_path = runs.get(src_key)
            tgt_path = runs.get(tgt_key)
            if src_path is None or tgt_path is None:
                logger.info(f"SKIP {label}: missing data")
                continue
            if not (src_path / "raw.npz").exists() or not (tgt_path / "raw.npz").exists():
                logger.info(f"SKIP {label}: raw.npz not found")
                continue
            r = run_transfer(src_path, tgt_path, label=label)
            r["type"] = "cross_task" if (src_key, tgt_key, label) in cross_task else "cross_model"
            all_results.append(r)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nWrote {len(all_results)} transfer results to {out_path}")

        # Also write CSV for paper/data
        import csv
        csv_path = Path("paper/data/table_transfer_matrix.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["label", "type", "source", "target",
                        "source_indist_gain", "zero_shot_gain",
                        "target_calibrated_gain", "target_trained_gain",
                        "transfer_peak_gain"])
            for r in all_results:
                w.writerow([r["label"], r["type"], r["source"], r["target"],
                            r["source_indist_gain"], r["zero_shot_gain"],
                            r["target_calibrated_gain"], r["target_trained_gain"],
                            r["transfer_peak_gain"]])
        logger.info(f"Wrote {csv_path}")

    else:
        if not args.source or not args.target:
            logger.info("Provide --source and --target, or use --all")
            return
        run_transfer(Path(args.source), Path(args.target), label=args.label)


if __name__ == "__main__":
    main()
