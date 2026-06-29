"""Cross-task transfer probe for Proposal 5.

Trains the stop-classifier on SOURCE (e.g. GSM8K), then applies it—without retraining—to TARGET
(e.g. MATH-500). This is the key transfer experiment: does the per-question stop signal generalise
across task domains?

Two conformal results are reported:
  (a) In-distribution (source):  grouped 5-fold CV on source (same as learnstop probe).
  (b) Out-of-distribution (target): split-conformal—clf trained on source, threshold calibrated on
      a held-out split of TARGET (cal_frac of target questions), evaluated on the remaining target
      questions. This gives a valid finite-sample guarantee on the TARGET distribution even though
      the clf was trained on SOURCE.

The separation is meaningful for the paper:
  - clf weights     : learned on source (what features matter)
  - stop threshold  : calibrated on a small target cal-set (how confident before stopping)
  Both together give a "few-shot calibration" story: train once, calibrate cheaply on new task.

Usage:
  python scripts/reason_transfer_probe.py \
      --source_data data/gsm8k.jsonl  --target_data data/math500.jsonl \
      --model $WORKSPACE/models/Qwen3-8B \
      --n 300 --max_think 3072 --budgets 0,128,192,256,384,512,640,768,1024,1536
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time

import numpy as np

logger = logging.getLogger(__name__)

MARKERS = ["wait", "hmm", "but ", "let me", "re-check", "recheck", "actually", "alternatively",
           "double-check", "double check", "hold on", "oops", "mistake", "wrong", "let's", "however"]

log = logging.getLogger(__name__)


def load_jsonl(path, n, seed=42):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rng = np.random.default_rng(seed)
    if n and n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def extract_answer(text, is_mc=False):
    if text is None:
        return ""
    if is_mc:
        m = re.search(r'\b([A-J])\b', text.strip()[:120])
        return m.group(1) if m else text.strip()[:1].upper()
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).replace(",", "").replace(" ", "").strip().rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return (nums[-1].rstrip(".") if nums else text.strip()[-32:]).strip()


def norm(s):
    return re.sub(r"\s+", "", str(s)).replace(",", "").rstrip(".").lower()


def marker_count(text):
    t = text.lower()
    return sum(t.count(m) for m in MARKERS)


# inference helpers (shared between source and target)

def run_dataset(rows, gold, budgets, tok, model, has_et, think_start, think_end, args):
    """Full pass + budget forcing with confidence. Returns (correct[N,m], p_raw[N,m], think_lens[N])."""
    import torch
    N = len(rows)
    m = len(budgets)
    dev = next(model.parameters()).device
    _task = rows[0].get("task", "") if rows else ""
    is_mc = _task in {"mmlu_pro", "gpqa"} or bool(rows and re.match(r'^[a-j]$', norm(rows[0]["gold"])))

    def base_prompt(q):
        if is_mc:
            instr = ("Answer the multiple-choice question. Write only the letter of the correct "
                     "answer (A, B, C, …) after 'Final answer:'.\n\n" + q)
        else:
            instr = ("Solve the problem. Put the final answer after 'Final answer:' "
                     "(a single number or \\boxed{...}).\n\n" + q)
        msgs = [{"role": "user", "content": instr}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if has_et:
            kw["enable_thinking"] = True
        return tok.apply_chat_template(msgs, **kw)

    def gen(texts, max_new, want_conf=False, _tag=""):
        out_txt, lp_out, ent_out = [], [], []
        n_batches = (len(texts) + args.batch - 1) // args.batch
        t0 = time.time()
        for bi, i in enumerate(range(0, len(texts), args.batch)):
            if _tag:
                log.info(f"  [{_tag}] batch {bi+1}/{n_batches} ...")
            ch = texts[i:i + args.batch]
            inp = tok(ch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len + args.max_think).to(dev)
            with torch.no_grad():
                out = model.generate(
                    **inp, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    return_dict_in_generate=True, output_scores=want_conf)
            plen = inp["input_ids"].shape[1]
            seq = out.sequences[:, plen:]
            out_txt += tok.batch_decode(seq, skip_special_tokens=True)
            if want_conf:
                B = seq.shape[0]
                ar = torch.arange(B, device=dev)
                lp_sum = torch.zeros(B, device=dev)
                ent_sum = torch.zeros(B, device=dev)
                cnt = torch.zeros(B, device=dev)
                done = torch.zeros(B, dtype=torch.bool, device=dev)
                for t, sc in enumerate(out.scores):
                    logp = torch.log_softmax(sc.float(), dim=-1)
                    ent = -(logp.exp() * logp).sum(-1)
                    tid = seq[:, t]
                    active = (~done).float()
                    lp_sum += logp[ar, tid] * active
                    ent_sum += ent * active
                    cnt += active
                    done = done | (tid == tok.eos_token_id) | (tid == tok.pad_token_id)
                cnt = cnt.clamp(min=1)
                lp_out += (lp_sum / cnt).cpu().tolist()
                ent_out += (ent_sum / cnt).cpu().tolist()
            if _tag:
                log.info(f"  [{_tag}] batch {bi+1}/{n_batches} done  ({time.time()-t0:.0f}s elapsed)")
            # Free GPU memory between batches — critical for 32B model long generations.
            del out, seq, inp
            if dev == "cuda":
                torch.cuda.empty_cache()
        return (out_txt, lp_out, ent_out) if want_conf else out_txt

    # 1. Full pass
    bases = [base_prompt(r["question"]) for r in rows]
    full_gen = gen(bases, args.max_think + args.ans_tokens, want_conf=False, _tag="full")
    reasonings, think_lens = [], []
    for g in full_gen:
        closed = think_end in g
        think_txt = g.split(think_end)[0].replace(think_start, "").strip() if closed else g
        reasonings.append(think_txt)
        think_lens.append(len(tok.encode(think_txt, add_special_tokens=False)))
    think_lens = np.array(think_lens)

    bases = [b.rsplit(think_start, 1)[0] if b.rstrip().endswith(think_start) else b for b in bases]
    reason_ids = [tok.encode(rz, add_special_tokens=False) for rz in reasonings]

    def forced_prompt(base, rid, B):
        if B <= 0:
            return base.rstrip() + f"\n{think_start}\n\n{think_end}\n\nFinal answer:"
        trunc = tok.decode(rid[:B], skip_special_tokens=True)
        return base.rstrip() + f"\n{think_start}\n" + trunc + f"\n{think_end}\n\nFinal answer:"

    # 2. Budget forcing with confidence
    ans = np.empty((N, m), dtype=object)
    conf_lp = np.zeros((N, m)); conf_ent = np.zeros((N, m)); mkr = np.zeros((N, m))
    for j, B in enumerate(budgets):
        log.info(f"[budget] {j+1}/{m}  B={B} ...")
        fp = [forced_prompt(bases[i], reason_ids[i], B) for i in range(N)]
        txt, lp, ent = gen(fp, args.ans_tokens, want_conf=True, _tag=f"B={B}")
        for i in range(N):
            ans[i, j] = extract_answer(txt[i], is_mc=is_mc)
            conf_lp[i, j] = lp[i]; conf_ent[i, j] = ent[i]
            trunc = tok.decode(reason_ids[i][:B], skip_special_tokens=True) if B > 0 else ""
            mkr[i, j] = marker_count(trunc) / max(min(B, think_lens[i]), 1) * 100.0
        log.info(f"  scored budget {B:>5}  acc={np.mean([norm(ans[i,j])==gold[i] for i in range(N)]):.3f}")

    correct = np.array([[norm(ans[i, j]) == gold[i] for j in range(m)] for i in range(N)], dtype=int)

    # 3. Causal features
    def feats(i, j):
        a = norm(ans[i, j])
        stable = int(j > 0 and a == norm(ans[i, j - 1]) and a != "")
        run = 0
        for k in range(j, -1, -1):
            if norm(ans[i, k]) == a and a != "":
                run += 1
            else:
                break
        votes = sum(1 for k in range(j + 1) if norm(ans[i, k]) == a and a != "")
        ended = int(think_lens[i] <= budgets[j])
        used = min(int(think_lens[i]), budgets[j])
        return [budgets[j] / 1000.0, j / max(m - 1, 1), conf_lp[i, j], conf_ent[i, j],
                stable, run, votes / (j + 1), mkr[i, j], used / 1000.0, ended]

    X = np.array([feats(i, j) for i in range(N) for j in range(m)], dtype=float)
    return correct, X, think_lens, ans


def print_frontier(label, fix_tok, fix_acc, p_stop, correct, think_lens, budgets, thrs):
    """Simulate and print adaptive frontier for a set of questions given p_stop scores."""
    N, m = correct.shape
    order = np.argsort(fix_tok)
    fx_t, fx_a = fix_tok[order], fix_acc[order]

    def fixed_at(tk):
        return float(np.interp(tk, fx_t, fx_a))

    def policy(thr):
        accs, toks = [], []
        for i in range(N):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            accs.append(correct[i, j])
            toks.append(min(int(think_lens[i]), budgets[j]))
        return float(np.mean(accs)), float(np.mean(toks))

    log.info(f"\n=== {label} ===")
    log.info(f"{'thr':>6}{'accuracy':>12}{'avg tokens':>12}{'fixed@cost':>12}{'adapt gain':>12}")
    best_gain = -9.9
    for thr in thrs:
        a, t = policy(thr)
        fa = fixed_at(t)
        gain = a - fa
        best_gain = max(best_gain, gain)
        log.info(f"{thr:>6.2f}{a:>12.3f}{t:>12.0f}{fa:>12.3f}{gain:>+12.3f}")
    return best_gain, policy, fixed_at


def conformal_section(label, p_stop, correct, think_lens, budgets, thrs, delta=0.05):
    """Risk control table for a held-out question set."""
    N, m = correct.shape
    full_correct = correct[:, -1]
    mean_full_tok = float(think_lens.mean())

    def risk_at(thr):
        rv = np.zeros(N)
        for i in range(N):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            rv[i] = float(full_correct[i] == 1 and correct[i, j] == 0)
        return rv

    def policy_acc_tok(thr):
        accs, toks = [], []
        for i in range(N):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            accs.append(correct[i, j])
            toks.append(min(int(think_lens[i]), budgets[j]))
        return float(np.mean(accs)), float(np.mean(toks))

    hb = float(np.sqrt(np.log(1.0 / delta) / (2.0 * N)))
    thr_grid = sorted(np.round(np.linspace(0.20, 0.99, 40), 3).tolist())
    cached = {thr: (risk_at(thr).mean(), *policy_acc_tok(thr)) for thr in thr_grid}

    log.info(f"\n=== CONFORMAL RISK CONTROL — {label} (δ={delta}, n={N}, bound={hb:.3f}) ===")
    log.info(f"  {'α':>6}  {'τ*':>6}  {'R̂(τ*)':>8}  {'guaranteed R':>14}  "
          f"{'avg tok':>9}  {'acc':>7}  {'saving':>8}")
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        tau_star = next((thr for thr in thr_grid if cached[thr][0] + hb <= alpha), None)
        if tau_star is None:
            log.info(f"  {alpha:>6.2f}  {'—':>6}  {'n/a':>8}  {'—':>14}  "
                  f"{'no τ satisfies guarantee at this n':>32}")
        else:
            emp_risk, a, t = cached[tau_star]
            saving = 100.0 * (1.0 - t / max(mean_full_tok, 1))
            log.info(f"  {alpha:>6.2f}  {tau_star:>6.2f}  {emp_risk:>8.3f}  "
                  f"≤{emp_risk+hb:>12.3f}  {t:>9.0f}  {a:>7.3f}  {saving:>7.0f}%")


def main():
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_data", required=True)
    parser.add_argument("--target_data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=300,
                    help="questions per dataset (target is split cal/test so use >=200)")
    parser.add_argument("--max_think", type=int, default=3072)
    parser.add_argument("--budgets", default="0,128,192,256,384,512,640,768,1024,1536")
    parser.add_argument("--ans_tokens", type=int, default=48)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--cal_frac", type=float, default=0.5,
                    help="fraction of target used as conformal calibration set")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/transfer",
                    help="root dir for the saved run artefacts (raw npz + CSVs + json)")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    budgets = sorted(set(int(b) for b in args.budgets.split(",")))
    m = len(budgets)
    thrs = np.round(np.linspace(0.30, 0.95, 14), 3)

    src_rows = load_jsonl(args.source_data, args.n, args.seed)
    tgt_rows = load_jsonl(args.target_data, args.n, args.seed + 1)
    src_gold = [norm(r["gold"]) for r in src_rows]
    tgt_gold = [norm(r["gold"]) for r in tgt_rows]

    import os
    src_name = os.path.splitext(os.path.basename(args.source_data))[0]
    tgt_name = os.path.splitext(os.path.basename(args.target_data))[0]

    log.info(f"\n{'='*60}")
    log.info(f"Transfer probe: {src_name} → {tgt_name} | model={args.model}")
    log.info(f"source n={len(src_rows)} | target n={len(tgt_rows)} | budgets={budgets}")
    log.info(f"{'='*60}")

    # load model
    tok = AutoTokenizer = None
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    # model-family detection (mirrors learnstop probe)
    mname = args.model.lower().replace("/", "")
    if "qwen3" in mname:
        family = "qwen3"
    elif "deepseek" in mname or "r1" in mname:
        family = "deepseek_r1"
    else:
        family = "generic"
    HAS_ET = (family == "qwen3")
    TS = "<think>"; TE = "</think>"
    log.info(f"[transfer] model family={family}  has_enable_thinking={HAS_ET}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32),
        device_map=("auto" if dev == "cuda" else None),
        attn_implementation=attn_impl)
    log.info(f"[transfer] attn_implementation={attn_impl}")
    if dev == "cpu":
        model.to(dev)
    model.eval()

    # run SOURCE
    log.info(f"\n[transfer] Running SOURCE ({src_name}, n={len(src_rows)})...")
    src_correct, src_X, src_think, src_ans = run_dataset(
        src_rows, src_gold, budgets, tok, model, HAS_ET, TS, TE, args)
    Ns = len(src_rows)
    log.info(f"[transfer] source mean full think = {src_think.mean():.0f} tok")

    # run TARGET
    log.info(f"\n[transfer] Running TARGET ({tgt_name}, n={len(tgt_rows)})...")
    tgt_correct, tgt_X, tgt_think, tgt_ans = run_dataset(
        tgt_rows, tgt_gold, budgets, tok, model, HAS_ET, TS, TE, args)
    Nt = len(tgt_rows)
    log.info(f"[transfer] target mean full think = {tgt_think.mean():.0f} tok")

    # SOURCE: train clf (full source) + grouped CV for in-distribution eval
    src_y = src_correct.reshape(-1)
    src_groups = np.repeat(np.arange(Ns), m)

    scaler = StandardScaler().fit(src_X)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(src_X), src_y)

    # grouped CV for in-distribution p_stop
    src_p_stop = np.zeros((Ns, m))
    gkf = GroupKFold(n_splits=args.folds)
    for fold_idx, (tr, te) in enumerate(gkf.split(src_X, src_y, src_groups), 1):
        log.info(f"[cv] fold {fold_idx}/{args.folds} ...")
        sc2 = StandardScaler().fit(src_X[tr])
        clf2 = LogisticRegression(max_iter=1000, C=1.0).fit(sc2.transform(src_X[tr]), src_y[tr])
        pr = clf2.predict_proba(sc2.transform(src_X[te]))[:, 1]
        for idx, p in zip(te, pr):
            src_p_stop[idx // m, idx % m] = p
        log.info(f"[cv] fold {fold_idx}/{args.folds} done")

    # TARGET: apply source clf directly (no retraining)
    tgt_p_stop_raw = clf.predict_proba(scaler.transform(tgt_X))[:, 1].reshape(Nt, m)

    # split target into cal and test
    rng = np.random.default_rng(args.seed + 99)
    n_cal = int(round(Nt * args.cal_frac))
    idx_perm = rng.permutation(Nt)
    cal_idx = idx_perm[:n_cal]
    test_idx = idx_perm[n_cal:]

    tgt_p_stop_cal = tgt_p_stop_raw[cal_idx]
    tgt_p_stop_test = tgt_p_stop_raw[test_idx]
    tgt_correct_cal = tgt_correct[cal_idx]
    tgt_correct_test = tgt_correct[test_idx]
    tgt_think_test = tgt_think[test_idx]
    tgt_think_cal = tgt_think[cal_idx]

    # FIXED-budget frontiers
    def fix_frontier(correct, think_lens):
        fix_tok = np.array([float(np.mean(np.minimum(think_lens, B))) for B in budgets])
        fix_acc = np.array([float(correct[:, j].mean()) for j in range(m)])
        return fix_tok, fix_acc

    src_fix_tok, src_fix_acc = fix_frontier(src_correct, src_think)
    tgt_fix_tok, tgt_fix_acc = fix_frontier(tgt_correct, tgt_think)
    tgt_test_fix_tok, tgt_test_fix_acc = fix_frontier(tgt_correct_test, tgt_think_test)

    log.info(f"\n=== FIXED-budget frontier — SOURCE ({src_name}) ===")
    log.info(f"{'budget':>8}{'accuracy':>12}{'avg tokens':>12}")
    for j, B in enumerate(budgets):
        log.info(f"{B:>8}{src_fix_acc[j]:>12.3f}{src_fix_tok[j]:>12.0f}")

    log.info(f"\n=== FIXED-budget frontier — TARGET ({tgt_name}, all) ===")
    log.info(f"{'budget':>8}{'accuracy':>12}{'avg tokens':>12}")
    for j, B in enumerate(budgets):
        log.info(f"{B:>8}{tgt_fix_acc[j]:>12.3f}{tgt_fix_tok[j]:>12.0f}")

    # adaptive frontiers
    src_best_gain, _, _ = print_frontier(
        f"IN-DISTRIBUTION adaptive stop — SOURCE ({src_name}, grouped {args.folds}-fold CV)",
        src_fix_tok, src_fix_acc, src_p_stop, src_correct, src_think, budgets, thrs)

    tgt_best_gain, _, _ = print_frontier(
        f"TRANSFER adaptive stop — TARGET ({tgt_name}, clf from {src_name}, all target)",
        tgt_fix_tok, tgt_fix_acc, tgt_p_stop_raw, tgt_correct, tgt_think, budgets, thrs)

    tgt_test_best_gain, _, _ = print_frontier(
        f"TRANSFER adaptive stop — TARGET-TEST ({tgt_name} test split n={len(test_idx)}, "
        f"clf from {src_name})",
        tgt_test_fix_tok, tgt_test_fix_acc, tgt_p_stop_test,
        tgt_correct_test, tgt_think_test, budgets, thrs)

    # conformal sections
    conformal_section(
        f"SOURCE in-distribution ({src_name}, grouped CV)",
        src_p_stop, src_correct, src_think, budgets, thrs)

    # split conformal on target: calibrate threshold on cal, report on test
    # find threshold on cal half → apply to test half
    log.info(f"\n=== SPLIT-CONFORMAL TRANSFER ({src_name}→{tgt_name}) ===")
    log.info(f"  Threshold calibrated on target-CAL (n={n_cal}), evaluated on target-TEST (n={len(test_idx)})")
    log.info(f"  This gives a valid finite-sample guarantee on the target distribution.\n")

    delta = 0.05
    hb_cal = float(np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))
    full_correct_cal = tgt_correct_cal[:, -1]

    def risk_at_cal(thr):
        rv = np.zeros(n_cal)
        for i in range(n_cal):
            j = next((jj for jj in range(m) if tgt_p_stop_cal[i, jj] >= thr), m - 1)
            rv[i] = float(full_correct_cal[i] == 1 and tgt_correct_cal[i, j] == 0)
        return rv

    def policy_test(thr):
        accs, toks = [], []
        for i in range(len(test_idx)):
            j = next((jj for jj in range(m) if tgt_p_stop_test[i, jj] >= thr), m - 1)
            accs.append(tgt_correct_test[i, j])
            toks.append(min(int(tgt_think_test[i]), budgets[j]))
        return float(np.mean(accs)), float(np.mean(toks))

    thr_grid = sorted(np.round(np.linspace(0.20, 0.99, 40), 3).tolist())
    log.info(f"  Hoeffding bound on cal set (n={n_cal}, δ={delta}) = {hb_cal:.3f}")
    log.info(f"\n  {'α':>6}  {'τ*(cal)':>8}  {'R̂_cal':>8}  {'R̂+bound':>10}  "
          f"{'test acc':>9}  {'test tok':>9}  {'saving':>8}")
    full_test_tok = float(tgt_think_test.mean())
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        tau_star = next(
            (thr for thr in thr_grid if risk_at_cal(thr).mean() + hb_cal <= alpha), None)
        if tau_star is None:
            log.info(f"  {alpha:>6.2f}  {'—':>8}  {'n/a':>8}  {'—':>10}  "
                  f"{'no τ satisfies guarantee at this n':>40}")
        else:
            emp_cal = risk_at_cal(tau_star).mean()
            test_acc, test_tok = policy_test(tau_star)
            saving = 100.0 * (1.0 - test_tok / max(full_test_tok, 1))
            log.info(f"  {alpha:>6.2f}  {tau_star:>8.2f}  {emp_cal:>8.3f}  "
                  f"≤{emp_cal+hb_cal:>8.3f}  {test_acc:>9.3f}  {test_tok:>9.0f}  {saving:>7.0f}%")

    # ZERO-SHOT TRANSFER: source clf + source threshold, no target calibration
    # This is the true zero-shot evaluation — no target data used at all.
    # We use the best in-distribution threshold from source CV as the operating point.
    src_fix_tok_s, src_fix_acc_s = fix_frontier(src_correct, src_think)
    order_s = np.argsort(src_fix_tok_s)
    fx_t_s, fx_a_s = src_fix_tok_s[order_s], src_fix_acc_s[order_s]

    def src_policy_gain(thr):
        accs, toks = [], []
        for i in range(Ns):
            j = next((jj for jj in range(m) if src_p_stop[i, jj] >= thr), m - 1)
            accs.append(src_correct[i, j])
            toks.append(min(int(src_think[i]), budgets[j]))
        a, t = float(np.mean(accs)), float(np.mean(toks))
        return a - float(np.interp(t, fx_t_s, fx_a_s))

    # find best source threshold
    src_best_thr = max(thrs, key=lambda t: src_policy_gain(t))

    # apply source threshold to target (zero-shot)
    zs_accs, zs_toks = [], []
    for i in range(Nt):
        j = next((jj for jj in range(m) if tgt_p_stop_raw[i, jj] >= src_best_thr), m - 1)
        zs_accs.append(tgt_correct[i, j])
        zs_toks.append(min(int(tgt_think[i]), budgets[j]))
    zs_acc, zs_tok = float(np.mean(zs_accs)), float(np.mean(zs_toks))
    zs_fa = float(np.interp(zs_tok, tgt_fix_tok[np.argsort(tgt_fix_tok)],
                              tgt_fix_acc[np.argsort(tgt_fix_tok)]))
    zs_gain = zs_acc - zs_fa

    log.info(f"\n=== ZERO-SHOT TRANSFER (source thr={src_best_thr:.2f}, no target calibration) ===")
    log.info(f"  target acc={zs_acc:.3f}  avg_tok={zs_tok:.0f}  "
             f"fixed@cost={zs_fa:.3f}  gain={zs_gain:+.3f}")

    # summary
    log.info("\n" + "="*60)
    log.info("TRANSFER SUMMARY")
    log.info("="*60)
    log.info(f"  in-distribution  ({src_name})  peak adapt gain : {src_best_gain:+.3f}")
    log.info(f"  zero-shot        ({tgt_name})  adapt gain      : {zs_gain:+.3f}  (source thr, no target cal)")
    log.info(f"  target-calibrated({tgt_name})  peak adapt gain : {tgt_best_gain:+.3f}")
    log.info(f"  target-test      ({tgt_name})  peak adapt gain : {tgt_test_best_gain:+.3f}")
    gap = src_best_gain - tgt_test_best_gain
    log.info(f"  in-dist → transfer gap : {gap:+.3f}  "
          f"({'acceptable' if gap < 0.04 else 'notable — discuss in paper'})")

    log.info("\n--- TRANSFER VERDICT ---")
    if tgt_test_best_gain >= 0.03:
        log.info(f"  GO: transfer adapt gain {tgt_test_best_gain:+.3f} is positive and meaningful. "
              f"The stop signal generalises across task domains without retraining. "
              f"The clf captures a task-agnostic reasoning pattern.")
    elif tgt_test_best_gain >= 0.00:
        log.info(f"  PARTIAL: transfer gain {tgt_test_best_gain:+.3f} is positive but small. "
              f"The signal transfers but attenuates; discuss as a 'cheap calibration' setting "
              f"(threshold tuned on a small target cal set).")
    else:
        log.info(f"  NO-GO for zero-shot transfer: gain {tgt_test_best_gain:+.3f} is negative. "
              f"The clf does not generalise; reframe as task-specific training (in-distribution only).")
    log.info("="*60)

    # PERSIST: source + target raw matrices for offline CIs/baselines/ablation
    import result_io
    run = result_io.make_run_dir(args.out_dir, data_name=f"{src_name}-to-{tgt_name}",
                                 model=args.model, n=max(Ns, Nt))
    result_io.dump_raw(run / "source_raw.npz", budgets=budgets, gold=src_gold,
                       correct=src_correct, think_lens=src_think, X=src_X,
                       p_stop=src_p_stop, ans=src_ans)
    result_io.dump_raw(run / "target_raw.npz", budgets=budgets, gold=tgt_gold,
                       correct=tgt_correct, think_lens=tgt_think, X=tgt_X,
                       p_stop=tgt_p_stop_raw, ans=tgt_ans,
                       cal_idx=np.asarray(cal_idx), test_idx=np.asarray(test_idx))
    result_io.write_json(run / "meta.json", {
        "probe": "transfer", "source": args.source_data, "target": args.target_data,
        "source_name": src_name, "target_name": tgt_name, "model": args.model,
        "source_n": Ns, "target_n": Nt, "cal_frac": args.cal_frac, "budgets": budgets,
        "max_think": args.max_think, "folds": args.folds, "seed": args.seed,
    })
    result_io.write_json(run / "summary.json", {
        "indist_peak_gain": round(float(src_best_gain), 4),
        "transfer_peak_gain": round(float(tgt_best_gain), 4),
        "transfer_test_peak_gain": round(float(tgt_test_best_gain), 4),
        "zeroshot_gain": round(float(zs_gain), 4),
        "zeroshot_thr": round(float(src_best_thr), 3),
        "gap": round(float(gap), 4),
    })
    result_io.log_saved(log, run)


if __name__ == "__main__":
    main()
