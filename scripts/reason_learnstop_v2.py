from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import grading_v2 as G
from gen_backend import make_backend

MARKERS = ["wait", "hmm", "but ", "let me", "re-check", "recheck", "actually", "alternatively",
           "double-check", "double check", "hold on", "oops", "mistake", "wrong", "let's", "however"]
DEFAULT_BUDGETS = "0,128,256,384,512,768,1024,1536,2048,3072,4096,6144,8192,12288"
PROBE_TAIL = "Final answer:"
log = logging.getLogger("ls_v2")


def instruction(mode: str) -> str:
    if mode == "mc":
        return ("Answer the multiple-choice question. Write only the letter of the correct answer "
                "after 'Final answer:'.\n\n")
    if mode == "math":
        return "Solve the problem. Write 'Final answer:' followed by the final answer in \\boxed{}.\n\n"
    if mode == "numeric":
        return "Solve the problem. Write 'Final answer:' followed by the final answer as a single number.\n\n"
    return "Solve the task. Write 'Final answer:' followed by only the final answer.\n\n"


def marker_density(text: str, n_tok: int) -> float:
    t = text.lower()
    return sum(t.count(m) for m in MARKERS) / max(n_tok, 1) * 100.0


def qseed(base: int, qid: str) -> int:
    return (base * 1_000_003 + int(hashlib.md5(qid.encode()).hexdigest()[:8], 16)) % (2**31 - 1)


def load_tokenizer(path: str):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    probe = "Hello world\n\nFinal answer: 42"
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.decode(tok.encode(probe, add_special_tokens=False)) == probe:
        return tok
    log.warning(f"[v2] {type(tok).__name__} fails the round-trip check; using PreTrainedTokenizerFast")
    tok = PreTrainedTokenizerFast.from_pretrained(path)
    assert tok.decode(tok.encode(probe, add_special_tokens=False)) == probe, "tokenizer round-trip failed"
    return tok


def eos_ids_of(tok, model_path: str) -> list[int]:
    ids = set()
    gc = Path(model_path) / "generation_config.json"
    if gc.exists():
        e = json.load(open(gc)).get("eos_token_id")
        ids.update(e if isinstance(e, list) else ([e] if e is not None else []))
    for t in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")):
        if isinstance(t, int) and t != tok.unk_token_id:
            ids.add(t)
    return sorted(ids)


def load_rows(path: str, n: int):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return rows[:n] if n else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated jsonl files from prepare_reason_v2.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", default="")
    ap.add_argument("--out_root", default="results/v2")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--quantization", default="", help="e.g. fp8 to fit Qwen3-32B on one H100")
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--max_think", type=int, default=16384)
    ap.add_argument("--resp_tokens", type=int, default=1024, help="extra tokens for the natural response")
    ap.add_argument("--ans_cap", type=int, default=48, help="max tokens decoded per probe")
    ap.add_argument("--logprobs", type=int, default=20, help="top-K for probe confidence/entropy")
    ap.add_argument("--budgets", default=DEFAULT_BUDGETS)
    ap.add_argument("--end_only", action="store_true", help="natural pass + end probe only (no checkpoints)")
    ap.add_argument("--temperature", type=float, default=-1, help="-1 = family default")
    ap.add_argument("--top_p", type=float, default=-1)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 = all rows (use small n for a smoke test)")
    ap.add_argument("--probe_chunk", type=int, default=128, help="questions per probe chunk (resume unit)")
    ap.add_argument("--natural_chunk", type=int, default=512, help="questions per natural-pass chunk (resume unit)")
    ap.add_argument("--hf_batch_tokens", type=int, default=262144)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", force=True)

    tok = load_tokenizer(args.model)
    mlow = args.model.lower()
    family = "qwen3" if "qwen3" in mlow else ("deepseek_r1" if ("deepseek" in mlow or "r1" in mlow
                                                                 or "adaptthink" in mlow) else "generic")
    T = args.temperature if args.temperature >= 0 else 0.6
    top_p = args.top_p if args.top_p > 0 else 0.95
    top_k = args.top_k if args.top_k >= 0 else (20 if family == "qwen3" else 0)
    think_end_id = tok.convert_tokens_to_ids("</think>")
    if think_end_id is None or think_end_id == tok.unk_token_id:
        think_end_id = None
    model_tag = args.model_tag or os.path.basename(args.model.rstrip("/"))
    budgets = [] if args.end_only else sorted({int(b) for b in args.budgets.split(",") if b.strip()})
    log.info(f"[v2] model={args.model} family={family} T={T} top_p={top_p} top_k={top_k} "
             f"think_end_id={think_end_id} budgets={budgets}")

    def ids_of(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    think_start_id = tok.convert_tokens_to_ids("<think>")
    if think_start_id is None or think_start_id == tok.unk_token_id:
        think_start_id = None

    def build_prompt(row) -> list[int]:
        msgs = [{"role": "user", "content": instruction(row["mode"]) + row["question"]}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if family == "qwen3":
            kw["enable_thinking"] = True
        text = tok.apply_chat_template(msgs, **kw)
        base = text.rstrip()[:-len("<think>")] if text.rstrip().endswith("<think>") else text
        if think_start_id is not None:
            return ids_of(base) + [think_start_id] + ids_of("\n")
        return ids_of(base + "<think>\n")

    if think_end_id is not None:
        suffix_ids = ids_of("\n") + [think_end_id] + ids_of("\n\n" + PROBE_TAIL)
    else:
        suffix_ids = ids_of("\n</think>\n\n" + PROBE_TAIL)
    stop_ids = eos_ids_of(tok, args.model)
    eos_ids = set(stop_ids) | ({tok.pad_token_id} if isinstance(tok.pad_token_id, int) else set())
    log.info(f"[v2] stop/eos ids={stop_ids} suffix={tok.decode(suffix_ids)!r}")

    datasets = []
    for path in [p for p in args.data.split(",") if p.strip()]:
        if (Path(args.out_root) / f"{Path(path).stem}__{model_tag}" / "raw.npz").exists():
            log.info(f"[v2] {Path(path).stem}__{model_tag}/raw.npz exists — skipping")
            continue
        rows = load_rows(path, args.n)
        prompts = [build_prompt(r) for r in rows]
        datasets.append((path, rows, prompts))
    if not datasets:
        log.info("[v2] nothing to do")
        return
    max_prompt = max(len(p) for _, _, ps in datasets for p in ps)
    max_model_len = max_prompt + args.max_think + max(args.resp_tokens, len(suffix_ids) + args.ans_cap) + 16
    log.info(f"[v2] max prompt={max_prompt} tok -> max_model_len={max_model_len}")

    backend = make_backend(args.backend, args.model, tp=args.tp, gpu_mem=args.gpu_mem,
                           max_model_len=max_model_len, seed=args.seed,
                           quantization=(args.quantization or None), enforce_eager=args.enforce_eager,
                           max_logprobs=args.logprobs, batch_tokens=args.hf_batch_tokens)

    for path, rows, prompts in datasets:
        run_one(args, tok, backend, path, rows, prompts, budgets, suffix_ids, eos_ids, stop_ids, think_end_id,
                family, (T, top_p, top_k), model_tag)


def split_trace(tok, ids, think_end_id, max_think):
    if think_end_id is not None:
        pos = ids.index(think_end_id) if think_end_id in ids else None
        if pos is not None and pos <= max_think:
            return ids[:pos], True, ids[pos + 1:]
        return ids[:max_think], False, []
    text = tok.decode(ids, skip_special_tokens=False)
    if "</think>" in text:
        before, after = text.split("</think>", 1)
        R = tok.encode(before, add_special_tokens=False)
        if len(R) <= max_think:
            return R, True, tok.encode(after, add_special_tokens=False)
    return ids[:max_think], False, []


def line_stop(tok, ids, eos_ids):
    for t in range(len(ids)):
        if ids[t] in eos_ids:
            return t + 1, tok.decode(ids[:t], skip_special_tokens=True)
        s = tok.decode(ids[:t + 1], skip_special_tokens=True)
        k = s.find("\n")
        while k != -1:
            if s[:k].strip():
                return t + 1, s[:k]
            k = s.find("\n", k + 1)
    return len(ids), tok.decode(ids, skip_special_tokens=True)


def run_one(args, tok, backend, path, rows, prompts, budgets, suffix_ids, eos_ids, stop_ids, think_end_id,
            family, sampling, model_tag):
    T, top_p, top_k = sampling
    stem = Path(path).stem
    run = Path(args.out_root) / f"{stem}__{model_tag}"
    (run / "chunks").mkdir(parents=True, exist_ok=True)
    N, m, A = len(rows), len(budgets), args.ans_cap
    if (run / "raw.npz").exists():
        log.info(f"[v2] {run}/raw.npz exists — skipping {stem}")
        return
    log.info(f"[v2] ===== {stem}: N={N} m={m} -> {run}")
    timings = {}

    tr_path = run / "traces.jsonl.gz"
    if tr_path.exists():
        traces = [json.loads(l) for l in gzip.open(tr_path, "rt")]
        assert len(traces) == N, f"{tr_path} has {len(traces)} rows, expected {N}"
        log.info(f"[v2] loaded {N} traces from {tr_path}")
    else:
        t0 = time.time()
        traces = []
        nat_chunk = args.natural_chunk or N
        n_nat = (N + nat_chunk - 1) // nat_chunk
        for c in range(n_nat):
            cp = run / "chunks" / f"natural_{c:04d}.jsonl.gz"
            lo, hi = c * nat_chunk, min(N, (c + 1) * nat_chunk)
            if cp.exists():
                traces += [json.loads(l) for l in gzip.open(cp, "rt")]
                continue
            outs = backend.generate(prompts[lo:hi], max_tokens=args.max_think + args.resp_tokens, temperature=T,
                                    top_p=top_p, top_k=top_k, seeds=[qseed(args.seed, r["id"]) for r in rows[lo:hi]],
                                    stop_token_ids=stop_ids, tag=f"natural:{stem}:{c + 1}/{n_nat}")
            part = []
            for r, p, o in zip(rows[lo:hi], prompts[lo:hi], outs):
                R, closed, resp = split_trace(tok, o.token_ids, think_end_id, args.max_think)
                part.append({"id": r["id"], "prompt_len": len(p), "think_ids": R, "closed": bool(closed),
                             "resp_text": tok.decode(resp, skip_special_tokens=True), "gen_len": len(o.token_ids)})
            with gzip.open(cp.with_suffix(".tmp"), "wt") as f:
                for t in part:
                    f.write(json.dumps(t) + "\n")
            os.replace(cp.with_suffix(".tmp"), cp)
            traces += part
        assert [t["id"] for t in traces] == [r["id"] for r in rows], "natural chunks out of order"
        with gzip.open(tr_path.with_suffix(".tmp"), "wt") as f:
            for t in traces:
                f.write(json.dumps(t) + "\n")
        os.replace(tr_path.with_suffix(".tmp"), tr_path)
        timings["natural_s"] = round(time.time() - t0, 1)
    think_len = np.array([len(t["think_ids"]) for t in traces])
    closed = np.array([t["closed"] for t in traces])
    log.info(f"[v2] think tokens: mean={think_len.mean():.0f} median={np.median(think_len):.0f} "
             f"p95={np.percentile(think_len, 95):.0f} | unclosed at cap={1 - closed.mean():.3f}")

    t0 = time.time()
    n_chunks = (N + args.probe_chunk - 1) // args.probe_chunk
    for c in range(n_chunks):
        cp = run / "chunks" / f"probes_{c:04d}.npz"
        if cp.exists():
            continue
        lo, hi = c * args.probe_chunk, min(N, (c + 1) * args.probe_chunk)
        jobs, reqs = [], []
        for i in range(lo, hi):
            R = traces[i]["think_ids"]
            for j, B in enumerate(budgets):
                if B < len(R):
                    jobs.append((i, j, B)); reqs.append(prompts[i] + R[:B] + suffix_ids)
            jobs.append((i, m, len(R))); reqs.append(prompts[i] + R + suffix_ids)
        outs = backend.generate(reqs, max_tokens=A, temperature=0.0, logprobs=args.logprobs,
                                stop_token_ids=stop_ids, tag=f"probe:{stem}:{c + 1}/{n_chunks}")
        n = hi - lo
        txt_line = np.full((n, m + 1), "", dtype=object); txt_full = np.full((n, m + 1), "", dtype=object)
        len_line = np.zeros((n, m + 1), np.int16); len_full = np.zeros((n, m + 1), np.int16)
        lp = np.full((n, m + 1, A), np.nan, np.float32); ent = np.full((n, m + 1, A), np.nan, np.float32)
        mkr = np.zeros((n, m + 1), np.float32)
        for (i, j, B), o in zip(jobs, outs):
            ii = i - lo
            L1, s1 = line_stop(tok, o.token_ids, eos_ids)
            txt_line[ii, j], len_line[ii, j] = s1, L1
            txt_full[ii, j] = tok.decode(o.token_ids, skip_special_tokens=True)
            len_full[ii, j] = len(o.token_ids)
            k = min(len(o.lp), A)
            lp[ii, j, :k] = o.lp[:k]; ent[ii, j, :k] = o.ent[:k]
            mkr[ii, j] = marker_density(tok.decode(traces[i]["think_ids"][:B], skip_special_tokens=True), B)
        np.savez_compressed(cp, txt_line=txt_line.astype(str), txt_full=txt_full.astype(str),
                            len_line=len_line, len_full=len_full, lp=lp, ent=ent, mkr=mkr)
        log.info(f"[v2] chunk {c + 1}/{n_chunks} saved ({len(reqs)} probes)")
    timings["probe_s"] = round(time.time() - t0, 1)

    parts = [np.load(run / "chunks" / f"probes_{c:04d}.npz") for c in range(n_chunks)]
    cat = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].files}
    active = np.array([[B < think_len[i] for B in budgets] for i in range(N)], dtype=bool).reshape(N, m)

    modes = [r["mode"] for r in rows]; nopt = [int(r.get("n_options", 0) or 0) for r in rows]
    gold = [str(r["gold"]) for r in rows]
    end_c = np.array([G.grade(cat["txt_line"][i, m], gold[i], modes[i], nopt[i])[1] for i in range(N)])
    nat_c = np.array([G.grade(traces[i]["resp_text"], gold[i], modes[i], nopt[i], where="natural")[1]
                      for i in range(N)])
    fixed = []
    for j, B in enumerate(budgets):
        cj = [G.grade(cat["txt_line"][i, j] if active[i, j] else cat["txt_line"][i, m], gold[i], modes[i],
                      nopt[i])[1] for i in range(N)]
        fixed.append({"budget": B, "acc": round(float(np.mean(cj)), 4),
                      "mean_think": round(float(np.minimum(think_len, B).mean()), 1)})
    summary = {"N": N, "natural_acc": round(float(nat_c.mean()), 4), "end_probe_acc": round(float(end_c.mean()), 4),
               "mean_think": round(float(think_len.mean()), 1), "unclosed": round(float(1 - closed.mean()), 4),
               "mean_len_line": round(float(cat["len_line"][:, :m][active].mean()), 2) if m else None,
               "mean_len_full": round(float(cat["len_full"][:, :m][active].mean()), 2) if m else None,
               "mean_active_ckpts": round(float(active.sum(1).mean()), 2), "fixed": fixed, "timings": timings}
    log.info(f"[v2] {stem}: natural acc={summary['natural_acc']:.3f} | end-probe acc={summary['end_probe_acc']:.3f} "
             f"| probe len line/full={summary['mean_len_line']}/{summary['mean_len_full']} "
             f"| active ckpts={summary['mean_active_ckpts']}")
    for f_ in fixed:
        log.info(f"    fixed B={f_['budget']:>6}: acc={f_['acc']:.3f} think={f_['mean_think']:.0f}")

    np.savez_compressed(
        run / "raw.npz", budgets=np.array(budgets, dtype=int), think_len=think_len, closed=closed,
        prompt_len=np.array([t["prompt_len"] for t in traces]), gen_len=np.array([t["gen_len"] for t in traces]),
        active=active, txt_line=cat["txt_line"], txt_full=cat["txt_full"], len_line=cat["len_line"],
        len_full=cat["len_full"], lp=cat["lp"], ent=cat["ent"], mkr=cat["mkr"],
        resp_text=np.array([t["resp_text"] for t in traces]), gold=np.array(gold), mode=np.array(modes),
        n_options=np.array(nopt), category=np.array([str(r.get("category", "")) for r in rows]),
        qid=np.array([r["id"] for r in rows]))
    try:
        import vllm
        vv = vllm.__version__
    except Exception:
        vv = None
    meta = {"data": path, "stem": stem, "task": rows[0]["task"], "split": stem.rsplit("_", 1)[-1],
            "model": model_tag, "model_tag": model_tag, "family": family, "backend": args.backend,
            "vllm_version": vv, "N": N, "budgets": budgets, "max_think": args.max_think, "ans_cap": args.ans_cap,
            "logprobs_topk": args.logprobs, "sampling": {"temperature": T, "top_p": top_p, "top_k": top_k,
                                                         "seed": args.seed},
            "probe_tail": PROBE_TAIL, "suffix_ids": suffix_ids, "stop_ids": stop_ids, "end_only": args.end_only,
            "tokenizer_class": type(tok).__name__, "timings": timings}
    json.dump(meta, open(run / "meta.json", "w"), indent=2)
    json.dump(summary, open(run / "summary.json", "w"), indent=2)
    log.info(f"[v2] wrote {run}/raw.npz")


if __name__ == "__main__":
    main()
