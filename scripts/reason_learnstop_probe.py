"""Decisive probe for Proposal 5 — can a LEARNED per-question stop capture the budget frontier?

The budget probe (reason_budget_probe.py) proved the OPPORTUNITY: a fixed think-token budget far below
the model's natural length matches/beats full accuracy, and there is a runaway tail. What it did NOT prove
is the crux of the paper: that a *realizable, causal, per-question* signal can decide WHEN to stop and
recover that frontier (the naive "answer repeats" rule was far too eager).

This probe tests exactly that. For each question we run the model once, then force an answer at every
budget checkpoint (capturing the answer-token CONFIDENCE), build causal per-checkpoint features, and train
a lightweight logistic stop-classifier with GROUPED (by question) 5-fold CV. We then simulate the stop
POLICY on held-out questions across a sweep of thresholds and compare its accuracy-vs-token frontier to:
  (a) the NAIVE stability stop (one point),
  (b) the FIXED-budget frontier (the non-adaptive baseline = the budget probe's table).

GO if the learned adaptive frontier clearly beats the naive stop AND matches-or-exceeds the fixed-budget
frontier at matched average cost (adaptivity should win by spending few tokens on easy items, more on hard).

Features per checkpoint j (all causal — only info available by budget B_j):
  budget B_j, j index, answer-token mean log-prob & mean entropy, answer stable vs previous checkpoint,
  run-length of a stable answer, vote-share of the current answer so far, backtrack-marker density in the
  reasoning so far ("wait/hmm/but/let me/actually/..."), think tokens used so far, whether the trace ended.
Label: the forced answer at this checkpoint is correct (gold; training-only).

Usage:
  python scripts/reason_learnstop_probe.py --data data/gsm8k.jsonl \
      --model models/Qwen3-8B --n 200 --max_think 3072 \
      --budgets 0,128,192,256,384,512,640,768,1024,1536
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

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
    # GPT-2/Llama BPE tokenizers decode space as Ġ (U+0120); normalize before regex
    text = text.replace('Ġ', ' ').replace('Ċ', '\n')
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


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_think", type=int, default=3072)
    parser.add_argument("--budgets", default="0,128,192,256,384,512,640,768,1024,1536")
    parser.add_argument("--ans_tokens", type=int, default=48)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", default="results/learnstop",
                    help="root dir for the saved run artefacts (raw.npz + CSVs + json)")
    parser.add_argument("--concise", action="store_true",
                    help="also run a 'think briefly' prompt baseline (extra GPU full pass)")
    parser.add_argument("--probe_template", default="terse", choices=["terse", "no_reasoning", "the_answer_is"],
                    help="probe answer prompt template for robustness sweep (J1)")
    parser.add_argument("--temperature", type=float, default=0.0,
                    help="decoding temperature for robustness sweep (J3); 0 = greedy")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    budgets = sorted(set(int(b) for b in args.budgets.split(",")))
    m = len(budgets)
    rows = load_jsonl(args.data, args.n, args.seed)
    gold = [norm(r["gold"]) for r in rows]
    # Detect MCQ by task field (authoritative) or fall back to checking if gold is a single letter.
    # Checking gold[0] alone is unreliable: GPQA gold was previously stored as full answer text,
    # causing is_mc=False and treating the task as free-form (bug fix).
    _task = rows[0].get("task", "") if rows else ""
    is_mc = _task in {"mmlu_pro", "gpqa"} or bool(rows and re.match(r'^[a-j]$', gold[0]))
    N = len(rows)
    log.info(f"[learnstop] n={N} | model={args.model} | budgets={budgets} | folds={args.folds} | mc={is_mc}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    # model-family detection
    mname = args.model.lower().replace("/", "")
    if "qwen3" in mname:
        family = "qwen3"        # Qwen3 thinking models: enable_thinking kwarg, <think>/</think>
    elif "deepseek" in mname or "r1" in mname:
        family = "deepseek_r1"  # DeepSeek-R1 distill: no enable_thinking, uses <think>/</think> or <!-- think -->
    else:
        family = "generic"      # fallback: try <think>/</think>, no enable_thinking

    # per-family config
    THINK_START = "<think>"     # marker that opens a thinking section
    THINK_END = "</think>"      # marker that closes a thinking section
    HAS_ENABLE_THINKING = (family == "qwen3")
    log.info(f"[learnstop] model family={family}  has_enable_thinking={HAS_ENABLE_THINKING}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Try flash_attention_2 first (O(N) memory vs O(N²) for sdpa); fall back to sdpa.
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401 — verify flash-attn is importable
    except ImportError:
        attn_impl = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32),
        device_map=("auto" if dev == "cuda" else None),
        attn_implementation=attn_impl)
    log.info(f"[learnstop] attn_implementation={attn_impl}")
    if dev == "cpu":
        model.to(dev)
    model.eval()

    def base_prompt(q, thinking):
        if is_mc:
            instr = ("Answer the multiple-choice question. Write only the letter of the correct "
                     "answer (A, B, C, …) after 'Final answer:'.\n\n" + q)
        else:
            instr = ("Solve the problem. Put the final answer after 'Final answer:' "
                     "(a single number or \\boxed{...}).\n\n" + q)
        msgs = [{"role": "user", "content": instr}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if HAS_ENABLE_THINKING:
            kw["enable_thinking"] = thinking
        return tok.apply_chat_template(msgs, **kw)

    def gen(texts, max_new, want_conf=False, _tag=""):
        txt, lp_out, ent_out = [], [], []
        n_batches = (len(texts) + args.batch - 1) // args.batch
        t0 = time.time()
        for bi, i in enumerate(range(0, len(texts), args.batch)):
            if _tag:
                log.info(f"  [{_tag}] batch {bi+1}/{n_batches} ...")
            ch = texts[i:i + args.batch]
            inp = tok(ch, return_tensors="pt", padding=True, truncation=True,
                      max_length=args.max_len + args.max_think).to(model.device)
            gen_kwargs = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id,
                              return_dict_in_generate=True, output_scores=want_conf)
            if args.temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=args.temperature, top_p=1.0)
            else:
                gen_kwargs["do_sample"] = False
            with torch.no_grad():
                out = model.generate(**inp, **gen_kwargs)
            plen = inp["input_ids"].shape[1]
            seq = out.sequences[:, plen:]
            txt += tok.batch_decode(seq, skip_special_tokens=True)
            if _tag:
                log.info(f"  [{_tag}] batch {bi+1}/{n_batches} done  ({time.time()-t0:.0f}s elapsed)")
            if want_conf:
                B = seq.shape[0]
                ar = torch.arange(B, device=model.device)
                lp_sum = torch.zeros(B, device=model.device)
                ent_sum = torch.zeros(B, device=model.device)
                cnt = torch.zeros(B, device=model.device)
                done = torch.zeros(B, dtype=torch.bool, device=model.device)
                for t, sc in enumerate(out.scores):
                    logp = torch.log_softmax(sc.float(), dim=-1)
                    # 0 * -inf = NaN in IEEE 754; replace those terms with 0 before summing
                    ent = -(logp.exp() * logp).nan_to_num(0.0).sum(-1)
                    tid = seq[:, t]
                    chosen = logp[ar, tid]
                    active = (~done).float()
                    lp_sum += chosen * active
                    ent_sum += ent * active
                    cnt += active
                    done = done | (tid == tok.eos_token_id) | (tid == tok.pad_token_id)
                cnt = cnt.clamp(min=1)
                lp_out += (lp_sum / cnt).cpu().tolist()
                ent_out += (ent_sum / cnt).cpu().tolist()
            # Free GPU memory between batches — critical for 32B model long generations
            # where KV-cache + attention matrices can fragment the 80 GiB H100.
            del out, seq, inp
            if dev == "cuda":
                torch.cuda.empty_cache()
        return (txt, lp_out, ent_out) if want_conf else txt

    # 1. FULL pass: capture reasoning trace + length
    log.info(f"[learnstop] FULL pass  ({N} questions, batch={args.batch}) ...")
    bases = [base_prompt(r["question"], thinking=True) for r in rows]
    full_gen = gen(bases, args.max_think + args.ans_tokens, want_conf=False, _tag="full")
    reasonings, think_lens = [], []
    for g in full_gen:
        closed = THINK_END in g
        think_txt = g.split(THINK_END)[0].replace(THINK_START, "").strip() if closed else g
        reasonings.append(think_txt)
        think_lens.append(len(tok.encode(think_txt, add_special_tokens=False)))
    think_lens = np.array(think_lens)
    log.info(f"[learnstop] mean full think length = {think_lens.mean():.0f} tokens")

    # optional CONCISE-PROMPT baseline: instruct brevity, measure acc + think tokens
    concise = None
    if args.concise:
        log.info("[learnstop] running concise-prompt baseline ('think briefly') ...")
        brief = "Think briefly and concisely; keep your reasoning short.\n\n"
        if is_mc:
            c_instr = (brief + "Answer the multiple-choice question. Write only the letter of the "
                       "correct answer (A, B, C, …) after 'Final answer:'.\n\n")
        else:
            c_instr = (brief + "Solve the problem. Put the final answer after 'Final answer:' "
                       "(a single number or \\boxed{...}).\n\n")
        c_prompts = []
        for r in rows:
            kw = dict(tokenize=False, add_generation_prompt=True)
            if HAS_ENABLE_THINKING:
                kw["enable_thinking"] = True
            c_prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": c_instr + r["question"]}], **kw))
        c_gen = gen(c_prompts, args.max_think + args.ans_tokens, want_conf=False, _tag="concise")
        c_correct, c_think = [], []
        for gi, g in enumerate(c_gen):
            closed = THINK_END in g
            ctxt = g.split(THINK_END)[0].replace(THINK_START, "").strip() if closed else g
            c_think.append(len(tok.encode(ctxt, add_special_tokens=False)))
            tail = g.split(THINK_END)[-1] if closed else g
            c_correct.append(int(norm(extract_answer(tail, is_mc=is_mc)) == gold[gi]))
        concise = {"acc": round(float(np.mean(c_correct)), 4),
                   "mean_think_tok": round(float(np.mean(c_think)), 1)}
        log.info(f"[learnstop] concise baseline: acc={concise['acc']:.3f} "
                 f"@ {concise['mean_think_tok']:.0f} think tok "
                 f"(vs full {think_lens.mean():.0f} tok)")

    bases = [b.rsplit(THINK_START, 1)[0] if b.rstrip().endswith(THINK_START) else b for b in bases]

    # pre-tokenize reasonings once (for truncation + marker density)
    reason_ids = [tok.encode(rz, add_special_tokens=False) for rz in reasonings]

    # probe answer prompt tail varies by --probe_template (J1 robustness)
    PROBE_TAILS = {
        "terse": "Final answer:",
        "no_reasoning": "Give the answer directly with no additional reasoning.\nFinal answer:",
        "the_answer_is": "The answer is",
    }
    probe_tail = PROBE_TAILS[args.probe_template]

    def forced_prompt(base, rid, B):
        if B <= 0:
            return base.rstrip() + f"\n{THINK_START}\n\n{THINK_END}\n\n{probe_tail}"
        trunc = tok.decode(rid[:B], skip_special_tokens=True)
        return base.rstrip() + f"\n{THINK_START}\n" + trunc + f"\n{THINK_END}\n\n{probe_tail}"

    # 2. BUDGET FORCING with confidence at every checkpoint
    ans = np.empty((N, m), dtype=object)     # forced answer text
    conf_lp = np.zeros((N, m)); conf_ent = np.zeros((N, m))
    mkr = np.zeros((N, m))
    for j, B in enumerate(budgets):
        log.info(f"[learnstop] budget {j+1}/{m}  B={B} ...")
        fp = [forced_prompt(bases[i], reason_ids[i], B) for i in range(N)]
        txt, lp, ent = gen(fp, args.ans_tokens, want_conf=True)
        for i in range(N):
            ans[i, j] = extract_answer(txt[i], is_mc=is_mc)
            conf_lp[i, j] = lp[i]; conf_ent[i, j] = ent[i]
            trunc = tok.decode(reason_ids[i][:B], skip_special_tokens=True) if B > 0 else ""
            mkr[i, j] = marker_count(trunc) / max(min(B, think_lens[i]), 1) * 100.0
        log.info(f"[learnstop] scored budget {B:>5} "
              f"(acc {np.mean([norm(ans[i,j])==gold[i] for i in range(N)]):.3f})")

    # Safety: replace any NaN/inf that slipped through from edge-case logprob computation
    conf_lp  = np.nan_to_num(conf_lp,  nan=0.0, posinf=0.0, neginf=-20.0)
    conf_ent = np.nan_to_num(conf_ent, nan=0.0, posinf=0.0, neginf=0.0)

    correct = np.array([[norm(ans[i, j]) == gold[i] for j in range(m)] for i in range(N)], dtype=int)
    full_acc = float(correct[:, -1].mean())   # ~ "let it finish at the largest budget"

    # 3. causal features per (question, checkpoint)
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
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)   # guard before sklearn
    y = correct.reshape(-1)
    groups = np.repeat(np.arange(N), m)

    # 4. grouped 5-fold CV: predict P(stop here is correct) on held-out questions
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    p_stop = np.zeros((N, m))
    gkf = GroupKFold(n_splits=args.folds)
    for fold_idx, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        log.info(f"[cv] fold {fold_idx}/{args.folds} ...")
        sc = StandardScaler().fit(X[tr])
        # nan_to_num after transform: zero-variance columns become NaN when std=0
        X_tr = np.nan_to_num(sc.transform(X[tr]), nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(sc.transform(X[te]), nan=0.0, posinf=0.0, neginf=0.0)
        clf = LogisticRegression(max_iter=1000, C=1.0).fit(X_tr, y[tr])
        pr = clf.predict_proba(X_te)[:, 1]
        for idx, p in zip(te, pr):
            p_stop[idx // m, idx % m] = p
        log.info(f"[cv] fold {fold_idx}/{args.folds} done")

    # 5. policy frontier: stop at first checkpoint with P >= thr
    def policy(thr):
        accs, toks, probes = [], [], []
        for i in range(N):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            accs.append(correct[i, j])
            toks.append(min(int(think_lens[i]), budgets[j]))
            probes.append((j + 1) * args.ans_tokens)
        return float(np.mean(accs)), float(np.mean(toks)), float(np.mean(probes))

    # fixed-budget frontier (non-adaptive baseline)
    fix_tok = np.array([float(np.mean(np.minimum(think_lens, B))) for B in budgets])
    fix_acc = np.array([float(correct[:, j].mean()) for j in range(m)])
    order = np.argsort(fix_tok)
    fx_t, fx_a = fix_tok[order], fix_acc[order]

    def fixed_at(tk):  # interpolate the fixed-budget accuracy at a given avg-token cost
        return float(np.interp(tk, fx_t, fx_a))

    # naive stability stop (first checkpoint whose answer repeats the previous one)
    ns_acc, ns_tok = [], []
    for i in range(N):
        j = next((jj for jj in range(1, m)
                  if norm(ans[i, jj]) == norm(ans[i, jj - 1]) and ans[i, jj].strip()), m - 1)
        ns_acc.append(correct[i, j]); ns_tok.append(min(int(think_lens[i]), budgets[j]))
    naive_acc, naive_tok = float(np.mean(ns_acc)), float(np.mean(ns_tok))

    log.info("\n=== FIXED-budget frontier (non-adaptive baseline) ===")
    log.info(f"{'budget':>8}{'accuracy':>12}{'avg tokens':>12}")
    fixed_rows = []
    for j, B in enumerate(budgets):
        log.info(f"{B:>8}{fix_acc[j]:>12.3f}{fix_tok[j]:>12.0f}")
        fixed_rows.append([int(B), round(float(fix_acc[j]), 4), round(float(fix_tok[j]), 1)])

    log.info(f"\n=== NAIVE stability-stop: acc {naive_acc:.3f} @ {naive_tok:.0f} tok "
          f"(fixed-budget acc at same cost = {fixed_at(naive_tok):.3f}) ===")

    log.info("\n=== LEARNED adaptive stop frontier (grouped 5-fold CV, held-out) ===")
    log.info(f"{'thr':>6}{'accuracy':>12}{'avg tokens':>12}{'probe tok':>12}{'total tok':>12}{'fixed@cost':>12}{'adapt gain':>12}")
    thrs = np.round(np.linspace(0.30, 0.95, 14), 3)
    best_gain, best_match, best_gain_thr = -9.9, None, None
    adaptive_rows = []
    for thr in thrs:
        a, t, oh = policy(thr)
        fa = fixed_at(t)
        gain = a - fa
        if gain > best_gain:
            best_gain, best_gain_thr = gain, float(thr)
        if a >= full_acc - 0.001 and (best_match is None or t < best_match[1]):
            best_match = (a, t, thr)
        log.info(f"{thr:>6.2f}{a:>12.3f}{t:>12.0f}{oh:>12.0f}{t+oh:>12.0f}{fa:>12.3f}{gain:>+12.3f}")
        adaptive_rows.append([round(float(thr), 3), round(float(a), 4),
                              round(float(t), 1), round(float(fa), 4), round(float(gain), 4),
                              round(float(oh), 1), round(float(t + oh), 1)])

    # VERDICT
    log.info("\n--- VERDICT ---")
    beats_naive = any(policy(thr)[0] >= naive_acc and policy(thr)[1] <= naive_tok for thr in thrs) \
        or max(policy(t)[0] for t in thrs) >= naive_acc + 0.05
    match_str = (f"reaches full-budget acc ({full_acc:.3f}) at {best_match[1]:.0f} tok "
                 f"(fixed budget needs {[B for j,B in enumerate(budgets) if fix_acc[j]>=full_acc-0.005][:1]})"
                 if best_match else f"never reaches full-budget acc ({full_acc:.3f})")
    if best_gain >= 0.02 and beats_naive:
        log.info(f"  GO: the learned adaptive stop BEATS the fixed-budget frontier by up to {best_gain:+.3f} at "
              f"matched cost and clears the naive stop; it {match_str}. The per-question signal is real and "
              f"realizable -> build the controller (add conformal risk control for the guarantee).")
    elif best_gain >= -0.01 and beats_naive:
        log.info(f"  PARTIAL: the learned stop MATCHES the fixed-budget frontier (max adapt gain {best_gain:+.3f}) "
              f"and beats the naive rule; {match_str}. A working efficiency method, but adaptivity isn't clearly "
              f"winning yet -> add richer features (entropy trajectory, multi-sample agreement) / conformal stop.")
    else:
        log.info(f"  NO-GO: the learned stop does not beat fixed-budget (max gain {best_gain:+.3f}) "
              f"{'nor the naive rule ' if not beats_naive else ''}-> the realizable per-question signal is too "
              f"weak; reconsider features or the framing (e.g. lead with the runaway-tail capping result).")
    log.info("=================================================")

    # 6. CONFORMAL RISK CONTROL
    # We already have held-out p_stop[i,j] from grouped CV (no leakage).
    # Risk_i(τ) = 1 if full-thinking was correct but the stopped answer is wrong.
    # This directly bounds accuracy loss: Acc_full - Acc_stop ≤ E[R_i(τ)].
    # R̂(τ) = empirical mean risk on the n held-out questions.
    # By Hoeffding: P(R(τ) > R̂(τ) + sqrt(log(1/δ)/(2n))) ≤ δ
    # For each α, find the SMALLEST τ (most aggressive / earliest stopping) such that
    # R̂(τ) + bound ≤ α — that is the conformal guarantee threshold.

    full_correct = correct[:, -1]   # 1 if correct at full thinking budget

    def risk_at(thr):
        rv = np.zeros(N)
        for i in range(N):
            j = next((jj for jj in range(m) if p_stop[i, jj] >= thr), m - 1)
            rv[i] = float(full_correct[i] == 1 and correct[i, j] == 0)
        return rv

    delta = 0.05
    n_cal = N   # all questions have a held-out prediction (one fold each)
    hb = float(np.sqrt(np.log(1.0 / delta) / (2.0 * n_cal)))

    log.info(f"\n=== CONFORMAL RISK CONTROL (δ={delta}, n_cal={n_cal}, Hoeffding bound={hb:.3f}) ===")
    log.info(f"  Risk_i(τ): 1 if full-thinking correct but stopped answer wrong (bounds acc loss)")
    log.info(f"  Guarantee: Acc_full - Acc_stop ≤ α with probability ≥ {1-delta:.0%}\n")
    log.info(f"  {'α':>6}  {'τ*':>6}  {'R̂(τ*)':>8}  {'guaranteed R':>14}  {'avg tok':>9}  "
          f"{'total tok':>10}  {'acc':>7}  {'think save':>11}  {'total save':>11}")

    thr_grid = np.round(np.linspace(0.20, 0.99, 40), 3)  # finer grid for conformal search
    # pre-compute risk and policy outcome for each threshold (ascending order = most aggressive first)
    thr_sorted = sorted(thr_grid)
    cached = {}
    for thr in thr_sorted:
        rv = risk_at(thr)
        a, t, oh = policy(thr)
        cached[thr] = (rv.mean(), a, t, oh)

    mean_full_tok = float(think_lens.mean())
    full_total = mean_full_tok + args.ans_tokens   # full thinking + one answer decode
    conformal_rows = []
    for alpha in [0.02, 0.05, 0.10, 0.15, 0.20]:
        tau_star = None
        for thr in thr_sorted:   # most aggressive → safest; take first that satisfies
            emp_risk = cached[thr][0]
            if emp_risk + hb <= alpha:
                tau_star = thr
                break
        if tau_star is None:
            log.info(f"  {alpha:>6.2f}  {'—':>6}  {'—':>8}  {'—':>14}  "
                  f"{'n/a — no τ in grid satisfies guarantee':>32}")
            conformal_rows.append([alpha, "", "", "", "", "", "", "", ""])
        else:
            emp_risk, a, t, oh = cached[tau_star]
            saving = 100.0 * (1.0 - t / max(mean_full_tok, 1))
            total_t = t + oh
            saving_total = 100.0 * (1.0 - total_t / max(full_total, 1))
            log.info(f"  {alpha:>6.2f}  {tau_star:>6.2f}  {emp_risk:>8.3f}  "
                  f"≤{emp_risk+hb:>12.3f}  {t:>9.0f}  {total_t:>10.0f}  "
                  f"{a:>7.3f}  {saving:>10.0f}%  {saving_total:>10.0f}%")
            conformal_rows.append([alpha, round(float(tau_star), 3), round(float(emp_risk), 4),
                                   round(float(emp_risk + hb), 4), round(float(t), 1),
                                   round(float(a), 4), round(float(saving), 1),
                                   round(float(total_t), 1), round(float(saving_total), 1)])

    log.info("\n  Interpretation: at each α, stopping at τ* saves tokens vs full thinking,")
    log.info("  with a finite-sample guarantee that Acc_full - Acc_stop ≤ α (prob ≥ 95%).")
    log.info("  'think save' counts only reasoning tokens; 'total save' includes probe answer overhead.")
    log.info("=================================================")

    # 7. PERSIST: raw matrices (for offline CIs/baselines/ablation) + headline CSVs
    import result_io
    run = result_io.make_run_dir(args.out_dir, data_name=Path(args.data).stem,
                                 model=args.model, n=N)
    result_io.dump_raw(run / "raw.npz", budgets=budgets, gold=gold,
                       correct=correct, think_lens=think_lens,
                       conf_lp=conf_lp, conf_ent=conf_ent, mkr=mkr,
                       p_stop=p_stop, X=X, y=y, ans=ans)
    result_io.write_csv(run / "frontier_fixed.csv",
                        ["budget", "accuracy", "avg_tokens"], fixed_rows)
    result_io.write_csv(run / "frontier_adaptive.csv",
                        ["thr", "accuracy", "avg_tokens", "fixed_at_cost", "adapt_gain",
                         "probe_tokens", "total_tokens"],
                        adaptive_rows)
    result_io.write_csv(run / "conformal.csv",
                        ["alpha", "tau", "emp_risk", "guaranteed", "avg_tok", "acc", "saving_pct",
                         "total_tok", "saving_total_pct"],
                        conformal_rows)
    result_io.write_json(run / "meta.json", {
        "probe": "learnstop", "data": args.data, "data_name": Path(args.data).stem,
        "model": args.model, "n": N, "is_mc": bool(is_mc), "budgets": budgets,
        "max_think": args.max_think, "ans_tokens": args.ans_tokens, "folds": args.folds,
        "seed": args.seed,
        "feature_names": ["budget_k", "ckpt_frac", "conf_lp", "conf_ent", "stable",
                          "run_len", "vote_share", "marker_density", "used_k", "ended"],
        "probe_template": args.probe_template,
        "temperature": args.temperature,
    })
    result_io.write_json(run / "summary.json", {
        "full_acc": round(full_acc, 4), "mean_full_think_tok": round(mean_full_tok, 1),
        "peak_adapt_gain": round(float(best_gain), 4), "peak_gain_thr": best_gain_thr,
        "naive_acc": round(naive_acc, 4), "naive_tok": round(naive_tok, 1),
        "tok_at_full_acc": (round(float(best_match[1]), 1) if best_match else None),
        "acc_at_match": (round(float(best_match[0]), 4) if best_match else None),
        "ans_tokens": args.ans_tokens,
        "concise_baseline": concise,
    })
    result_io.log_saved(log, run)


if __name__ == "__main__":
    main()
