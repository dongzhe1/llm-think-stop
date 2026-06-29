"""Budget-forcing probe: does the model overthink?

Generates the full reasoning trace once per question, then force-answers at
every budget checkpoint. Measures accuracy-vs-tokens to answer three questions:

(a) Does thinking help?  (full vs zero-think accuracy)
(b) Is the model overthinking?  (matched accuracy at budget << full length)
(c) Is there a realizable per-question stop signal?  (stability-stop rule)
"""
from __future__ import annotations

import argparse
import json
import logging
import re

import numpy as np

logger = logging.getLogger(__name__)


def load_jsonl(path, n, seed=42):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rng = np.random.default_rng(seed)
    if n and n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def extract_answer(text: str) -> str:
    if text is None:
        return ""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).replace(",", "").replace(" ", "").strip().rstrip(".")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return (nums[-1].rstrip(".") if nums else text.strip()[-32:]).strip()


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).replace(",", "").rstrip(".").lower()


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max_think", type=int, default=2048)
    parser.add_argument("--budgets", default="0,64,128,256,512,1024")
    parser.add_argument("--ans_tokens", type=int, default=48)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    budgets = sorted(set(int(b) for b in args.budgets.split(",")))
    rows = load_jsonl(args.data, args.n, args.seed)
    gold = [norm(r["gold"]) for r in rows]
    logger.info("[reason] n=%d model=%s budgets=%s max_think=%d",
                len(rows), args.model, budgets, args.max_think)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    is_qwen3 = "qwen3" in args.model.lower().replace("/", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        device_map=("auto" if device == "cuda" else None))
    if device == "cpu":
        model.to(device)
    model.eval()

    def base_prompt(q, thinking):
        instr = (
            "Solve the problem. Put the final answer after 'Final answer:' "
            "(a single number or \\boxed{...}).\n\n" + q
        )
        msgs = [{"role": "user", "content": instr}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if is_qwen3:
            kw["enable_thinking"] = thinking
        return tokenizer.apply_chat_template(msgs, **kw)

    def gen(texts, max_new, sample=False):
        outs = []
        for i in range(0, len(texts), args.batch):
            ch = texts[i:i + args.batch]
            inp = tokenizer(
                ch, return_tensors="pt", padding=True, truncation=True,
                max_length=args.max_len + args.max_think,
            ).to(model.device)
            with torch.no_grad():
                o = model.generate(
                    **inp, max_new_tokens=max_new, do_sample=sample,
                    temperature=(0.7 if sample else None),
                    top_p=(0.95 if sample else None),
                    pad_token_id=tokenizer.pad_token_id,
                )
            plen = inp["input_ids"].shape[1]
            outs += tokenizer.batch_decode(o[:, plen:], skip_special_tokens=True)
        return outs

    # Full pass: capture the reasoning trace
    logger.info("[reason] full pass...")
    full_prompts = [base_prompt(r["question"], thinking=True) for r in rows]
    full_gen = gen(full_prompts, args.max_think + args.ans_tokens, sample=False)

    reasonings, full_ans, think_lens, full_closed = [], [], [], []
    for g in full_gen:
        closed = "</think>" in g
        think_txt = (g.split("</think>")[0].replace("<think>", "").strip()
                     if closed else g)
        after = g.split("</think>")[-1] if closed else ""
        reasonings.append(think_txt)
        full_ans.append(extract_answer(after if after.strip() else g))
        think_lens.append(len(tokenizer.encode(think_txt,
                                                add_special_tokens=False)))
        full_closed.append(int(closed and after.strip() != ""))

    think_lens = np.array(think_lens)
    logger.info("[reason] mean full think = %.0f tokens (median %.0f, max %d)",
                think_lens.mean(), np.median(think_lens), think_lens.max())

    # Budget forcing: answer at each truncated budget
    bases = [base_prompt(r["question"], thinking=True) for r in rows]
    bases = [b.rsplit("<think>", 1)[0] if b.rstrip().endswith("<think>") else b
             for b in bases]

    def forced_prompt(base, reasoning, budget):
        if budget <= 0:
            return base.rstrip() + "\n<think>\n\n</think>\n\nFinal answer:"
        ids = tokenizer.encode(reasoning, add_special_tokens=False)[:budget]
        trunc = tokenizer.decode(ids, skip_special_tokens=True)
        return (base.rstrip() + "\n<think>\n" + trunc
                + "\n</think>\n\nFinal answer:")

    ans_by_budget = {}
    for budget in budgets:
        fp = [forced_prompt(bases[i], reasonings[i], budget)
              for i in range(len(rows))]
        a = gen(fp, args.ans_tokens, sample=False)
        ans_by_budget[budget] = [extract_answer(x) for x in a]

    def acc(preds):
        return float(np.mean([norm(p) == g for p, g in zip(preds, gold)]))

    full_acc = acc(full_ans)
    logger.info("\n=== accuracy vs think-token budget ===")
    logger.info("%8s  %12s  %12s", "budget", "accuracy", "avg tokens")
    accs = {}
    for budget in budgets:
        a = acc(ans_by_budget[budget])
        accs[budget] = a
        used = float(np.mean(np.minimum(think_lens, budget)))
        logger.info("%8d  %12.3f  %12.0f", budget, a, used)
    logger.info("%8s  %12.3f  %12.0f", "FULL", full_acc, think_lens.mean())

    zero_acc = accs.get(0, full_acc)

    # Ceiling: best accuracy over budget grid AND full
    cand = dict(accs)
    cand["FULL"] = full_acc
    best_key = max(cand, key=cand.get)
    best_acc = cand[best_key]
    target = best_acc - 0.01

    fixed_budget = next((b for b in budgets if accs[b] >= target), None)
    fixed_used = (float(np.mean(np.minimum(think_lens, fixed_budget)))
                  if fixed_budget is not None else float("nan"))
    logger.info("[reason] ceiling = %.3f @ %s (full=%.3f); "
                "min budget within 1%% of ceiling = %s (avg %.0f tok)",
                best_acc, best_key, full_acc, fixed_budget, fixed_used)

    # Stability stop: stop when answer matches previous checkpoint
    grid = budgets
    stop_pred, stop_tok, settled = [], [], []
    for i in range(len(rows)):
        chosen, prev = None, None
        for j, b in enumerate(grid):
            cur = ans_by_budget[b][i]
            if j > 0 and norm(cur) == norm(prev) and cur.strip():
                chosen = (b, cur)
                break
            prev = cur
        if chosen is None:
            chosen = (grid[-1], ans_by_budget[grid[-1]][i])
        stop_pred.append(chosen[1])
        stop_tok.append(min(think_lens[i], chosen[0]))
        settled.append(int(norm(chosen[1]) == norm(full_ans[i])))

    stab_acc = acc(stop_pred)
    stab_tok = float(np.mean(stop_tok))
    frac_settled = float(np.mean(settled))
    savings = 100 * (1 - stab_tok / max(think_lens.mean(), 1))

    logger.info("\n=== stability stop (answer repeats) ===")
    logger.info("  accuracy  = %.3f  (full %.3f)", stab_acc, full_acc)
    logger.info("  avg think = %.0f  (full %.0f, saving %.0f%%)",
                stab_tok, think_lens.mean(), savings)
    logger.info("  P(stop==full) = %.3f", frac_settled)

    # Overthinking diagnostic
    cap = best_key if isinstance(best_key, int) else budgets[-1]
    closed_rate = float(np.mean(full_closed))
    closed_idx = [i for i in range(len(rows)) if full_closed[i]]
    full_acc_closed = (float(np.mean([norm(full_ans[i]) == gold[i]
                                      for i in closed_idx]))
                       if closed_idx else float("nan"))
    long_mask = think_lens > cap
    n_long = int(long_mask.sum())
    if n_long > 0:
        cap_ans = ans_by_budget[cap]
        acc_cap_long = float(np.mean([
            norm(cap_ans[i]) == gold[i]
            for i in range(len(rows)) if long_mask[i]]))
        acc_full_long = float(np.mean([
            norm(full_ans[i]) == gold[i]
            for i in range(len(rows)) if long_mask[i]]))
    else:
        acc_cap_long = acc_full_long = float("nan")

    logger.info("\n=== overthinking diagnostic ===")
    logger.info("  traces that closed </think>: %.3f", closed_rate)
    logger.info("  full acc on closed traces: %.3f  (all-trace %.3f)",
                full_acc_closed, full_acc)
    logger.info("  questions wanting >%d tok: %d", cap, n_long)
    if n_long > 0:
        logger.info("    capped@%d acc = %.3f  vs  full acc = %.3f  "
                    "(capping nets %+.3f, >0 = true overthinking)",
                    cap, acc_cap_long, acc_full_long,
                    acc_cap_long - acc_full_long)

    # Verdict
    logger.info("\n--- VERDICT ---")
    helps = (best_acc - zero_acc) >= 0.05
    cheap_match = (fixed_budget is not None
                   and fixed_used <= 0.6 * think_lens.mean())
    caps_beat_full = (best_key != "FULL" and (best_acc - full_acc) >= 0.02)
    long_loss = (n_long > 0) and (acc_cap_long - acc_full_long) >= 0.03
    overthinks = cheap_match or caps_beat_full or long_loss
    realizable = (stab_acc >= target
                  and stab_tok <= 0.6 * think_lens.mean())

    if not helps:
        logger.info("  NO-GO: thinking barely helps (ceiling %.3f vs "
                    "zero-think %.3f)", best_acc, zero_acc)
    elif not overthinks:
        logger.info("  NO-GO: matched accuracy needs ~full budget")
    elif realizable:
        logger.info("  GO: ceiling %.3f @ %s (full %.3f); "
                    "stability-stop hits %.3f at %.0f tok (%.0f%% saved).",
                    best_acc, best_key, full_acc, stab_acc, stab_tok, savings)
    else:
        why = []
        if caps_beat_full:
            why.append(f"capping beats full by {best_acc - full_acc:+.3f}")
        if long_loss:
            why.append(f"on long traces capping nets "
                       f"{acc_cap_long - acc_full_long:+.3f}")
        if cheap_match:
            why.append(f"matched ceiling at avg {fixed_used:.0f} tok")
        logger.info("  PARTIAL: overthinking confirmed — %s. "
                    "But stability-stop only reaches %.3f at %.0f tok — "
                    "a learned stop is needed.",
                    "; ".join(why), stab_acc, stab_tok)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
