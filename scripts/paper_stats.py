"""Paper-ready statistics from all completed experiment runs.

Reads results/*/analysis/analysis.json and produces consolidated tables.
Output: results/paper_stats/ (tables as .csv + paper_tables.txt)
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
OUT = RESULTS / "paper_stats"
OUT.mkdir(parents=True, exist_ok=True)

# name parsing

_MODEL_PATTERNS = ["Qwen3.5-9B", "Qwen3-32B", "Qwen3-8B"]

def parse_run_name(name):
    """Parse a run directory name into (task, model, n, tag).
    Pattern: {task}_{model}_n{N}_{tag}
    """
    m = re.search(r'_n(\d+)_(\d+)$', name)
    if not m:
        raise ValueError(f"Cannot parse run name: {name}")
    n = int(m.group(1))
    tag = m.group(2)
    prefix = name[:m.start()]

    model = None
    for mp in _MODEL_PATTERNS:
        if prefix.endswith("_" + mp):
            model = mp
            task = prefix[:-(len(mp) + 1)]
            break
        elif prefix.endswith(mp):
            model = mp
            task = prefix[:-len(mp)].rstrip("_")
            break
    if model is None:
        raise ValueError(f"Cannot find model in: {prefix}")
    return task, model, n, tag


# data loading

def load_learnstop():
    """Yield (task, model, n, run_dir, analysis, summary) for each learnstop run."""
    for d in sorted((RESULTS / "learnstop").iterdir()):
        if not d.is_dir():
            continue
        af = d / "analysis" / "analysis.json"
        sf = d / "summary.json"
        if not af.exists() or not sf.exists():
            continue
        with open(af) as f:
            analysis = json.load(f)
        with open(sf) as f:
            summary = json.load(f)
        task, model, n, tag = parse_run_name(d.name)
        yield task, model, n, d, analysis, summary


def load_transfer():
    """Yield (source, target, model, n, run_dir, analysis, summary)."""
    for d in sorted((RESULTS / "transfer").iterdir()):
        if not d.is_dir():
            continue
        sf = d / "summary.json"
        if not sf.exists():
            continue
        with open(sf) as f:
            summary = json.load(f)
        task, model, n, tag = parse_run_name(d.name)
        # task looks like "gsm8k-to-math500" or "gsm8k-to-mmlu_pro"
        if "-to-" not in task:
            continue
        source, target = task.split("-to-", 1)
        target_analysis = None
        taf = d / "analysis" / "analysis.json"
        if taf.exists():
            with open(taf) as f:
                target_analysis = json.load(f)
        yield source, target, model, n, d, target_analysis, summary


# paper tables

def table_main_results(ls_rows, tr_rows):
    """Main results table — in-distribution."""
    lines = []
    lines.append("=" * 110)
    lines.append("TABLE 1: In-Distribution Learnstop Results (grouped 5-fold CV)")
    lines.append("=" * 110)
    header = f"{'Task':<14} {'Model':<12} {'n':<6} {'Full Acc':<10} {'Full Tok':<10} {'Peak Gain':<11} {'CI lo':<9} {'CI hi':<9} {'Sig':<5} {'Gain Thr':<10} {'ΔAblation':<11} {'Tok@Full':<10}"
    lines.append(header)
    lines.append("-" * 110)

    # sort: task order, then model
    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
    sorted_rows = sorted(ls_rows, key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d, analysis, summary in sorted_rows:
        b = analysis.get("baselines", {})
        boot = analysis.get("bootstrap", {})
        abl = analysis.get("ablation", {})
        learned = b.get("learned", {})
        lb = boot.get("learned", {})
        full_acc = summary.get("full_acc", float("nan"))
        full_tok = summary.get("mean_full_think_tok", float("nan"))
        tok_at_full = summary.get("tok_at_full_acc", float("nan"))
        peak_gain = learned.get("peak_gain", float("nan"))
        op_thr = learned.get("op_thr", float("nan"))
        ci = lb.get("ci", [float("nan"), float("nan")])
        sig = "✓" if lb.get("sig", False) else ""
        delta_abl = abl.get("delta", float("nan")) if abl else float("nan")
        lines.append(f"{task:<14} {model:<12} {n:<6} {full_acc:<10.4f} {full_tok:<10.1f} {peak_gain:<11.4f} {ci[0]:<9.4f} {ci[1]:<9.4f} {sig:<5} {op_thr:<10.3f} {delta_abl:<+11.4f} {tok_at_full:<10.1f}")
    lines.append("")

    # Transfer
    lines.append("=" * 120)
    lines.append("TABLE 2: Cross-Task Transfer (GSM8K -> target, clf trained on source only)")
    lines.append("=" * 120)
    header2 = f"{'Target':<14} {'Model':<12} {'n':<6} {'In-dist Gain':<13} {'Transf Gain':<13} {'Gap':<10} {'CI lo':<9} {'CI hi':<9} {'Sig':<5}"
    lines.append(header2)
    lines.append("-" * 120)
    sorted_tr = sorted(tr_rows, key=lambda x: (task_order.get(x[1], 99), model_order.get(x[2], 99)))
    for source, target, model, n, d, target_analysis, summary in sorted_tr:
        if target_analysis is None:
            continue
        indis_gain = summary.get("indist_peak_gain", float("nan"))
        transf_gain = summary.get("transfer_test_peak_gain", float("nan"))
        gap = summary.get("gap", float("nan"))
        boot = target_analysis.get("bootstrap", {}).get("learned", {})
        ci = boot.get("ci", [float("nan"), float("nan")])
        sig = "✓" if boot.get("sig", False) else ""
        lines.append(f"{target:<14} {model:<12} {n:<6} {indis_gain:<13.4f} {transf_gain:<13.4f} {gap:<10.4f} {ci[0]:<9.4f} {ci[1]:<9.4f} {sig:<5}")
    lines.append("")
    return "\n".join(lines)


def table_baseline_comparison(ls_rows):
    """Compare learned vs all baselines on each grid."""
    lines = []
    lines.append("=" * 130)
    lines.append("TABLE 3: Method Comparison — All Baselines vs Learned (peak adapt gain)")
    lines.append("=" * 130)
    methods = ["learned", "self_consistency", "entropy_exit", "confidence_exit"]
    header = f"{'Task':<14} {'Model':<12} " + " ".join(f"{m:>16}" for m in methods) + f"  {'Best':<16} {'Learned wins?':<15}"
    lines.append(header)
    lines.append("-" * 130)

    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
    sorted_rows = sorted(ls_rows, key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d, analysis, summary in sorted_rows:
        b = analysis.get("baselines", {})
        boot = analysis.get("bootstrap", {})
        vals = []
        best_val, best_name = -9.9, ""
        for m in methods:
            v = b.get(m, {}).get("peak_gain", float("nan"))
            sig = "✓" if boot.get(m, {}).get("sig", False) else ""
            vals.append(f"{v:+.4f}{sig}")
            if v > best_val:
                best_val, best_name = v, m
        learned_v = b.get("learned", {}).get("peak_gain", float("nan"))
        winner = "✓" if best_name == "learned" else f"({best_name})"
        lines.append(f"{task:<14} {model:<12} " + " ".join(f"{v:>16}" for v in vals) + f"  {best_name:>16}  {winner:<15}")
    lines.append("")
    return "\n".join(lines)


def table_ablation(ls_rows):
    """Feature ablation summary."""
    lines = []
    lines.append("=" * 90)
    lines.append("TABLE 4: Feature Ablation — 10 Causal Features vs Single conf_lp")
    lines.append("=" * 90)
    header = f"{'Task':<14} {'Model':<12} {'All 10 feat':<13} {'conf_lp only':<14} {'Δ':<10} {'Interpretation':<25}"
    lines.append(header)
    lines.append("-" * 90)

    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
    sorted_rows = sorted(ls_rows, key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d, analysis, summary in sorted_rows:
        abl = analysis.get("ablation")
        if abl is None:
            continue
        all_f = abl.get("all_features", {}).get("peak_gain", float("nan"))
        one_f = abl.get("conf_lp_only", {}).get("peak_gain", float("nan"))
        delta = abl.get("delta", float("nan"))
        if delta > 0.02:
            interp = "features clearly help"
        elif delta > 0.005:
            interp = "features moderately help"
        elif delta > -0.005:
            interp = "features neutral"
        else:
            interp = "conf_lp alone sufficient"
        lines.append(f"{task:<14} {model:<12} {all_f:<13.4f} {one_f:<14.4f} {delta:<+10.4f} {interp:<25}")
    lines.append("")
    return "\n".join(lines)


def table_conformal(ls_rows):
    """Conformal risk control summary from conformal.csv files."""
    lines = []
    lines.append("=" * 110)
    lines.append("TABLE 5: Conformal Risk Control (δ=0.05, Hoeffding bound)")
    lines.append("=" * 110)
    header = f"{'Task':<14} {'Model':<12} {'α=0.10':<22} {'α=0.15':<22} {'α=0.20':<22}"
    lines.append(header)
    lines.append(f"{'':<14} {'':<12} {'τ/saving/acc':<22} {'τ/saving/acc':<22} {'τ/saving/acc':<22}")
    lines.append("-" * 110)

    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
    # sort by task then model
    entries = []
    for task, model, n, d, analysis, summary in ls_rows:
        entries.append((task, model, n, d))
    entries.sort(key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d in entries:
        cf = d / "conformal.csv"
        row_data = {0.10: "n/a", 0.15: "n/a", 0.20: "n/a"}
        if cf.exists():
            with open(cf) as f:
                reader = csv.DictReader(f)
                for r in reader:
                    alpha_str = r.get("alpha", "").strip()
                    if not alpha_str:
                        continue
                    alpha = float(alpha_str)
                    tau_str = r.get("tau", "").strip()
                    if not tau_str or tau_str == "nan":
                        continue
                    if alpha in row_data:
                        tau = float(tau_str)
                        saving = float(r.get("saving_pct", 0))
                        acc = float(r.get("acc", 0))
                        row_data[alpha] = f"τ={tau:.2f} {saving:.0f}% {acc:.3f}"
        line = f"{task:<14} {model:<12} {row_data[0.10]:<22} {row_data[0.15]:<22} {row_data[0.20]:<22}"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def table_concise(ls_rows):
    """Concise baseline vs full vs learned."""
    lines = []
    lines.append("=" * 125)
    lines.append("TABLE 6: Concise Baseline ('think briefly' prompt) vs Full vs Learned")
    lines.append("=" * 125)
    header = f"{'Task':<14} {'Model':<12} {'Full Acc':<10} {'Full Tok':<10} {'Concise Acc':<12} {'Concise Tok':<12} {'ΔAcc':<9} {'Learned Gain':<13} {'Concise better?':<16}"
    lines.append(header)
    lines.append("-" * 125)

    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
    sorted_rows = sorted(ls_rows, key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d, analysis, summary in sorted_rows:
        full_acc = summary.get("full_acc", float("nan"))
        full_tok = summary.get("mean_full_think_tok", float("nan"))
        concise = summary.get("concise_baseline", {})
        if concise is None:
            concise = {}
        c_acc = concise.get("acc", float("nan"))
        c_tok = concise.get("mean_think_tok", float("nan"))
        delta_acc = c_acc - full_acc
        learned_gain = analysis.get("baselines", {}).get("learned", {}).get("peak_gain", float("nan"))
        # concise is "better" if it improves acc AND saves tokens vs full
        if not np.isnan(delta_acc) and not np.isnan(c_tok) and not np.isnan(full_tok):
            if delta_acc > 0 and c_tok < full_tok:
                verdict = "✓ (acc+ & tok-)"
            elif delta_acc > 0:
                verdict = "acc+ only"
            elif c_tok < full_tok:
                verdict = "tok- only"
            elif delta_acc < -0.05:
                verdict = "❌ concise hurts"
            else:
                verdict = "≈ tie"
        else:
            verdict = "n/a"
        lines.append(f"{task:<14} {model:<12} {full_acc:<10.4f} {full_tok:<10.0f} {c_acc:<12.4f} {c_tok:<12.0f} {delta_acc:<+9.4f} {learned_gain:<13.4f} {verdict:<16}")
    lines.append("")
    return "\n".join(lines)


def table_paired_bootstrap(ls_rows):
    """Paired bootstrap: learned minus baseline gain, with CI.

    This re-runs bootstrap on the DIFFERENCE in per-resample gain so each
    baseline gets a within-resample paired comparison against learned.
    """
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 7: Paired Bootstrap — Learned minus Baseline (B=1000, within-resample)")
    lines.append("=" * 120)
    header = f"{'Task':<14} {'Model':<12} {'vs Self-Consist':<24} {'vs Entropy':<24} {'vs Conf-Exit':<24}"
    lines.append(header)
    lines.append(f"{'':<14} {'':<12} {'Δgain CI':<24} {'Δgain CI':<24} {'Δgain CI':<24}")
    lines.append("-" * 120)

    task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
    model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}

    entries = []
    for task, model, n, d, analysis, summary in ls_rows:
        entries.append((task, model, n, d, analysis))
    entries.sort(key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99)))

    for task, model, n, d, analysis in entries:
        npz_path = d / "raw.npz"
        if not npz_path.exists():
            continue
        z = np.load(npz_path, allow_pickle=False)
        correct = z["correct"]
        think_lens = z["think_lens"]
        budgets = list(z["budgets"])
        p_stop = z["p_stop"]
        conf_lp = z["conf_lp"]
        conf_ent = z["conf_ent"]
        ans = z["ans"]
        bud = np.array(budgets)
        N = correct.shape[0]

        # stability
        m = ans.shape[1]
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

        # Get learned op_thr
        learned_thr = analysis["baselines"]["learned"]["op_thr"]

        # precompute stop indices at learned's op point
        def first_ge(scores, thr):
            out = np.full(scores.shape[0], scores.shape[1] - 1, dtype=int)
            for i in range(scores.shape[0]):
                hit = np.nonzero(scores[i] >= thr)[0]
                if hit.size:
                    out[i] = hit[0]
            return out
        def first_le(scores, thr):
            out = np.full(scores.shape[0], scores.shape[1] - 1, dtype=int)
            for i in range(scores.shape[0]):
                hit = np.nonzero(scores[i] <= thr)[0]
                if hit.size:
                    out[i] = hit[0]
            return out

        learned_idx = first_ge(p_stop, learned_thr)

        # function to compute gain for a given stop_idx on a resample
        def compute_gain(stop_idx, correct_s, think_s, bud):
            a = correct_s[np.arange(len(correct_s)), stop_idx].mean()
            t = np.minimum(think_s, bud[stop_idx]).mean()
            ft = np.array([np.minimum(think_s, Bb).mean() for Bb in budgets])
            fa = np.array([correct_s[:, j].mean() for j in range(len(budgets))])
            order = np.argsort(ft)
            fa_fixed = float(np.interp(t, ft[order], fa[order]))
            return a - fa_fixed

        rng = np.random.default_rng(42)
        B = 1000

        baselines_info = {
            "self_consistency": ("op_thr", lambda thr: first_ge(run, thr)),
            "entropy_exit": ("op_thr", lambda thr: first_le(conf_ent, thr)),
            "confidence_exit": ("op_thr", lambda thr: first_ge(np.exp(conf_lp), thr)),
        }

        row_parts = []
        for bname, (thr_key, stop_fn) in baselines_info.items():
            b_thr = analysis["baselines"][bname]["op_thr"]
            base_idx = stop_fn(b_thr)

            diffs = np.empty(B)
            for b in range(B):
                s = rng.integers(0, N, N)
                lg = compute_gain(learned_idx[s], correct[s], think_lens[s], bud)
                bg = compute_gain(base_idx[s], correct[s], think_lens[s], bud)
                diffs[b] = lg - bg

            med = float(np.median(diffs))
            lo = float(np.percentile(diffs, 2.5))
            hi = float(np.percentile(diffs, 97.5))
            sig = "✓" if lo > 0 else ("✗" if hi < 0 else "≈")
            row_parts.append(f"{med:+.4f} [{lo:+.4f},{hi:+.4f}] {sig}")

        lines.append(f"{task:<14} {model:<12} {row_parts[0]:<24} {row_parts[1]:<24} {row_parts[2]:<24}")

    lines.append("")
    return "\n".join(lines)


# main

def main():
    ls_rows = list(load_learnstop())
    tr_rows = list(load_transfer())

    logger.info(f"Found {len(ls_rows)} learnstop runs, {len(tr_rows)} transfer runs")

    sections = [
        table_main_results(ls_rows, tr_rows),
        table_baseline_comparison(ls_rows),
        table_ablation(ls_rows),
        table_conformal(ls_rows),
        table_concise(ls_rows),
        table_paired_bootstrap(ls_rows),
    ]

    full = "\n".join(sections)
    logger.info(full)

    out_txt = OUT / "paper_tables.txt"
    out_txt.write_text(full)
    logger.info(f"\n[paper_stats] wrote {out_txt}")

    # also dump as structured CSV for each table
    # Table 1: main in-dist results
    with open(OUT / "table1_indist.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "model", "n", "full_acc", "full_tok", "peak_gain", "ci_lo", "ci_hi", "sig", "op_thr", "delta_ablation", "tok_at_full_acc"])
        task_order = {"gsm8k": 0, "math500": 1, "mmlu_pro": 2}
        model_order = {"Qwen3-8B": 0, "Qwen3-32B": 1}
        for task, model, n, d, analysis, summary in sorted(ls_rows, key=lambda x: (task_order.get(x[0], 99), model_order.get(x[1], 99))):
            b = analysis.get("baselines", {}).get("learned", {})
            boot = analysis.get("bootstrap", {}).get("learned", {})
            abl = analysis.get("ablation", {})
            if abl is None:
                abl = {}
            w.writerow([
                task, model, n,
                summary.get("full_acc"), summary.get("mean_full_think_tok"),
                b.get("peak_gain"),
                boot.get("ci", [None, None])[0], boot.get("ci", [None, None])[1],
                int(boot.get("sig", False)),
                b.get("op_thr"),
                abl.get("delta"),
                summary.get("tok_at_full_acc"),
            ])

    # Table 2: transfer
    with open(OUT / "table2_transfer.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "model", "n", "indist_gain", "transfer_gain", "gap", "ci_lo", "ci_hi", "sig"])
        for source, target, model, n, d, ta, summary in sorted(tr_rows, key=lambda x: (task_order.get(x[1], 99), model_order.get(x[2], 99))):
            if ta is None:
                continue
            boot = ta.get("bootstrap", {}).get("learned", {})
            w.writerow([
                target, model, n,
                summary.get("indist_peak_gain"), summary.get("transfer_test_peak_gain"),
                summary.get("gap"),
                boot.get("ci", [None, None])[0], boot.get("ci", [None, None])[1],
                int(boot.get("sig", False)),
            ])

    logger.info(f"[paper_stats] wrote {OUT}/table1_indist.csv, table2_transfer.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
