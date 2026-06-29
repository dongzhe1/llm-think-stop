"""Serving cost profiling — wall-clock measurement of probe overhead.

Measures actual latency of the checkpoint probing cycle: pre-generate full
reasoning traces, then time only the probe loop for various (batch_size,
ans_cap, n_checkpoints) configs. Outputs wall-clock latency, tokens/s, and
peak GPU memory.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def load_jsonl(path, n, seed=42):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rng = np.random.default_rng(seed)
    if n and n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def _make_budget_grid(n_checkpoints, max_budget=1536):
    if n_checkpoints <= 1:
        return [0]
    step = max_budget // (n_checkpoints - 1)
    return [min(i * step, max_budget) for i in range(n_checkpoints)]


def profile_hf(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    model.eval()

    rows = load_jsonl(args.data, args.n, args.seed)
    has_et = "qwen3" in args.model.lower()
    THINK_START, THINK_END = "<think>", "</think>"

    def make_prompt(q, thinking=True):
        instr = f"Solve the problem. Put the final answer after 'Final answer:'.\n\n{q}"
        msgs = [{"role": "user", "content": instr}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if has_et:
            kw["enable_thinking"] = thinking
        return tokenizer.apply_chat_template(msgs, **kw)

    # Phase 1: pre-generate full reasoning
    logger.info("Phase 1: pre-generating full reasoning for %d questions", len(rows))
    prompts_all = [make_prompt(r["question"]) for r in rows]
    full_texts = []
    t0 = time.perf_counter()
    for qi in range(len(rows)):
        inp = tokenizer([prompts_all[qi]], return_tensors="pt", padding=True,
                        truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=1536, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        plen = inp["input_ids"].shape[1]
        full_texts.append(tokenizer.decode(out[0, plen:], skip_special_tokens=True))
        del out, inp
        if (qi + 1) % 10 == 0:
            logger.info("  generated %d/%d", qi + 1, len(rows))
    torch.cuda.empty_cache()
    t_gen = time.perf_counter() - t0
    logger.info("Phase 1 done: %d questions in %.1fs", len(rows), t_gen)

    # Pre-tokenize truncated reasoning at all budget levels
    max_budget = 1536
    all_n_ckpts = [4, 7, 10]
    all_budget_grids = {nc: _make_budget_grid(nc, max_budget) for nc in all_n_ckpts}
    all_budgets_needed = sorted(set(
        b for grid in all_budget_grids.values() for b in grid))

    logger.info("Pre-tokenizing truncated reasoning...")
    trunc_cache = {}
    for qi, (prompt, ftxt) in enumerate(zip(prompts_all, full_texts)):
        base = (prompt.rsplit(THINK_START, 1)[0]
                if prompt.rstrip().endswith(THINK_START) else prompt)
        think_part = ftxt.split(THINK_END)[0].replace(THINK_START, "")
        reason_ids = tokenizer.encode(think_part, add_special_tokens=False)
        for b in all_budgets_needed:
            if b <= 0:
                fp = base.rstrip() + f"\n{THINK_START}\n\n{THINK_END}\n\nFinal answer:"
            else:
                trunc = tokenizer.decode(reason_ids[:b], skip_special_tokens=True)
                fp = base.rstrip() + f"\n{THINK_START}\n{trunc}\n{THINK_END}\n\nFinal answer:"
            trunc_cache[(qi, b)] = fp
    logger.info("  cached %d truncated prompts", len(trunc_cache))

    # Phase 2: time probing only
    results = []
    ans_caps = [4, 8, 16, 32, 48, 64]

    for batch_size in [1, 4]:
        for ans_cap in ans_caps:
            for n_checkpoints in all_n_ckpts:
                ckpt_budgets = all_budget_grids[n_checkpoints]
                timings = []
                peak_mems = []
                total_probe_tokens = 0

                for qi in range(0, len(rows), batch_size):
                    batch_indices = list(range(qi, min(qi + batch_size, len(rows))))
                    if not batch_indices:
                        continue
                    torch.cuda.reset_peak_memory_stats()
                    t_start = time.perf_counter()

                    for b in ckpt_budgets:
                        forced = [trunc_cache[(bi, b)] for bi in batch_indices]
                        finp = tokenizer(forced, return_tensors="pt", padding=True,
                                         truncation=True, max_length=4096).to(device)
                        with torch.no_grad():
                            pout = model.generate(
                                **finp, max_new_tokens=ans_cap, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
                        total_probe_tokens += ((pout.shape[1] - finp["input_ids"].shape[1])
                                               * pout.shape[0])
                        del finp, pout

                    t_end = time.perf_counter()
                    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    timings.append(t_end - t_start)
                    peak_mems.append(peak_mem)
                    torch.cuda.empty_cache()

                if timings:
                    total_time = sum(timings)
                    entry = {
                        "backend": "hf_transformers",
                        "batch_size": batch_size,
                        "ans_cap": ans_cap,
                        "n_checkpoints": n_checkpoints,
                        "checkpoint_budgets": ckpt_budgets,
                        "mean_latency_s": round(float(np.mean(timings)), 3),
                        "std_latency_s": round(float(np.std(timings)), 3),
                        "total_time_s": round(total_time, 2),
                        "peak_memory_gb": round(float(np.max(peak_mems)), 2),
                        "total_probe_tokens": int(total_probe_tokens),
                        "probe_tokens_per_s": (
                            round(total_probe_tokens / total_time, 1)
                            if total_time > 0 else 0),
                        "n_samples": len(timings),
                    }
                    results.append(entry)
                    logger.info("  bs=%d A=%d ckpts=%d: %.2fs ±%.2f peak=%.1fGB tok/s=%.0f",
                                batch_size, ans_cap, n_checkpoints,
                                entry["mean_latency_s"], entry["std_latency_s"],
                                entry["peak_memory_gb"], entry["probe_tokens_per_s"])

    return results, t_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="results/serving_profile.json")
    args = parser.parse_args()

    logger.info("Serving profile: model=%s backend=%s n=%d",
                args.model, args.backend, args.n)

    if args.backend == "hf":
        results, gen_time = profile_hf(args)
    else:
        logger.warning("vLLM backend not available; falling back to HF")
        results, gen_time = profile_hf(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "backend": args.backend,
            "n": args.n,
            "generation_time_s": round(gen_time, 1),
            "results": results,
        }, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
