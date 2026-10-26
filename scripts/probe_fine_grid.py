from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reason_learnstop_v2 as R
from gen_backend import make_backend

log = logging.getLogger("fine")
SETUP_ERROR = 2


def die(msg: str):
    log.error(f"[fine] ABORT: {msg}")
    sys.exit(SETUP_ERROR)


def fine_budgets(lo_step: int, lo_max: int, hi_step: int, max_think: int, original) -> list[int]:
    b = set(range(0, lo_max + 1, lo_step)) | set(range(lo_max + hi_step, max_think, hi_step)) | set(int(x) for x in original)
    return sorted(x for x in b if x < max_think)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model_tag", default="")
    ap.add_argument("--stems", required=True, help="comma-separated data stems, e.g. gsm8k_eval,math500_eval")
    ap.add_argument("--root", default="results/v2")
    ap.add_argument("--lo_step", type=int, default=128)
    ap.add_argument("--lo_max", type=int, default=1024)
    ap.add_argument("--hi_step", type=int, default=256)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--quantization", default="")
    ap.add_argument("--enforce_eager", action="store_true")
    ap.add_argument("--probe_chunk", type=int, default=128, help="questions per chunk (resume unit)")
    ap.add_argument("--n", type=int, default=0, help="first n questions only (smoke test; writes *_smoke files)")
    ap.add_argument("--min_agree", type=float, default=0.9,
                    help="abort if fewer probe answers than this match raw.npz (first chunk of every run; smoke: all)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S", force=True)

    tok = R.load_tokenizer(args.model)
    model_tag = args.model_tag or os.path.basename(args.model.rstrip("/"))
    mlow = args.model.lower()
    family = "qwen3" if "qwen3" in mlow else ("deepseek_r1" if ("deepseek" in mlow or "r1" in mlow) else "generic")
    think_start_id = tok.convert_tokens_to_ids("<think>")
    if think_start_id is None or think_start_id == tok.unk_token_id:
        think_start_id = None

    def ids_of(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    def build_prompt(row) -> list[int]:
        msgs = [{"role": "user", "content": R.instruction(row["mode"]) + row["question"]}]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if family == "qwen3":
            kw["enable_thinking"] = True
        text = tok.apply_chat_template(msgs, **kw)
        base = text.rstrip()[:-len("<think>")] if text.rstrip().endswith("<think>") else text
        if think_start_id is not None:
            return ids_of(base) + [think_start_id] + ids_of("\n")
        return ids_of(base + "<think>\n")

    suffix = ".smoke" if args.n else ""
    jobs = []
    for stem in [s for s in args.stems.split(",") if s.strip()]:
        run = Path(args.root) / f"{stem}__{model_tag}"
        out = run / f"fine_probes{suffix}.npz"
        if out.exists():
            log.info(f"[fine] {out} exists — skipping"); continue
        tr_path = run / "traces.jsonl.gz"
        if not tr_path.exists():
            die(f"{tr_path} missing — cannot re-probe {stem}")
        meta = json.load(open(run / "meta.json"))
        rows = R.load_rows(meta["data"], args.n)
        traces = [json.loads(l) for l in gzip.open(tr_path, "rt")][:len(rows)]
        if [t["id"] for t in traces] != [r["id"] for r in rows]:
            die(f"{stem}: trace/data order mismatch")
        prompts = [build_prompt(r) for r in rows]
        bad = sum(len(p) != t["prompt_len"] for p, t in zip(prompts, traces))
        if bad:
            die(f"{stem}: {bad} rebuilt prompts differ in length from the logged ones")
        log.info(f"[fine] {stem}: {len(rows)} rebuilt prompts match the logged lengths")
        budgets = fine_budgets(args.lo_step, args.lo_max, args.hi_step, meta["max_think"], meta["budgets"])
        jobs.append((stem, run, meta, rows, traces, prompts, budgets))
    if not jobs:
        log.info("[fine] nothing to do"); return

    stop_ids = jobs[0][2]["stop_ids"]; suffix_ids = jobs[0][2]["suffix_ids"]
    for j in jobs:
        if j[2]["suffix_ids"] != suffix_ids or j[2]["stop_ids"] != stop_ids:
            die("runs of one model log different probe suffixes or stop ids")
    eos_ids = set(stop_ids) | ({tok.pad_token_id} if isinstance(tok.pad_token_id, int) else set())
    A = jobs[0][2]["ans_cap"]
    max_len = max(len(p) + len(t["think_ids"]) for j in jobs for p, t in zip(j[5], j[4])) + len(suffix_ids) + A + 16
    log.info(f"[fine] model={args.model} stems={[j[0] for j in jobs]} max_model_len={max_len} "
             f"suffix={tok.decode(suffix_ids)!r}")
    backend = make_backend("vllm", args.model, tp=args.tp, gpu_mem=args.gpu_mem, max_model_len=max_len, seed=0,
                           quantization=(args.quantization or None), enforce_eager=args.enforce_eager,
                           max_logprobs=1)

    for stem, run, meta, rows, traces, prompts, budgets in jobs:
        t0 = time.time()
        N, m = len(rows), len(budgets)
        think_len = np.array([len(t["think_ids"]) for t in traces])
        log.info(f"[fine] ===== {stem}: N={N}, {m} budgets, "
                 f"{int(sum((np.array(budgets)[None, :] < think_len[:, None]).sum(1)))} probes")
        n_chunks = (N + args.probe_chunk - 1) // args.probe_chunk
        raw = np.load(run / "raw.npz", allow_pickle=False)
        ob = [int(x) for x in raw["budgets"]]
        checked = False
        for c in range(n_chunks):
            cp = run / "chunks" / f"fine{suffix}_{c:04d}.npz"
            if cp.exists():
                continue
            lo, hi = c * args.probe_chunk, min(N, (c + 1) * args.probe_chunk)
            idx, reqs = [], []
            for i in range(lo, hi):
                Rr = traces[i]["think_ids"]
                for j, B in enumerate(budgets):
                    if B < len(Rr):
                        idx.append((i, j)); reqs.append(prompts[i] + Rr[:B] + suffix_ids)
            outs = backend.generate(reqs, max_tokens=A, temperature=0.0, logprobs=0, stop_token_ids=stop_ids,
                                    tag=f"fine:{stem}:{c + 1}/{n_chunks}")
            if len(outs) != len(reqs):
                die(f"{stem}: {len(outs)} outputs for {len(reqs)} requests")
            n = hi - lo
            txt_line = np.full((n, m), "", dtype=object); txt_full = np.full((n, m), "", dtype=object)
            len_line = np.zeros((n, m), np.int16); len_full = np.zeros((n, m), np.int16)
            for (i, j), o in zip(idx, outs):
                L1, s1 = R.line_stop(tok, o.token_ids, eos_ids)
                txt_line[i - lo, j], len_line[i - lo, j] = s1, L1
                txt_full[i - lo, j] = tok.decode(o.token_ids, skip_special_tokens=True)
                len_full[i - lo, j] = len(o.token_ids)
            (run / "chunks").mkdir(exist_ok=True)
            np.savez_compressed(cp, txt_line=txt_line.astype(str), txt_full=txt_full.astype(str),
                                len_line=len_line, len_full=len_full)
            log.info(f"[fine] {stem}: chunk {c + 1}/{n_chunks} saved ({len(reqs)} probes)")
            if not checked:
                checked = True
                same = tot = 0
                for jo, B in enumerate(ob):
                    if B in budgets:
                        a = raw["active"][lo:hi, jo].astype(bool)
                        same += int((txt_line[:, budgets.index(B)][a] == raw["txt_line"][lo:hi, jo][a]).sum())
                        tot += int(a.sum())
                ag = same / max(tot, 1)
                log.info(f"[fine] {stem}: first-chunk agreement with raw.npz {ag:.4f} ({tot} probes)")
                if ag < args.min_agree:
                    cp.unlink()
                    die(f"{stem}: first-chunk agreement {ag:.4f} < {args.min_agree} — the probe setup differs "
                        f"from the original run")
        parts = [np.load(run / "chunks" / f"fine{suffix}_{c:04d}.npz") for c in range(n_chunks)]
        cat = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0].files}
        active = np.array(budgets)[None, :] < think_len[:, None]

        same, tot = 0, 0
        for jo, B in enumerate(ob):
            if B not in budgets:
                continue
            jf = budgets.index(B)
            a = raw["active"][:N, jo].astype(bool)
            same += int((cat["txt_line"][:, jf][a] == raw["txt_line"][:N, jo][a]).sum()); tot += int(a.sum())
        agree = same / max(tot, 1)
        log.info(f"[fine] {stem}: probe answers identical to raw.npz at the original budgets: {agree:.4f} ({tot} probes)")
        if agree < args.min_agree:
            die(f"{stem}: agreement {agree:.4f} < {args.min_agree} over all chunks — not writing fine_probes")

        np.savez_compressed(run / f"fine_probes{suffix}.npz", budgets=np.array(budgets, dtype=int), active=active,
                            txt_line=cat["txt_line"], txt_full=cat["txt_full"], len_line=cat["len_line"],
                            len_full=cat["len_full"], qid=np.array([r["id"] for r in rows]))
        json.dump({"stem": stem, "model_tag": model_tag, "N": N, "budgets": budgets, "lo_step": args.lo_step,
                   "lo_max": args.lo_max, "hi_step": args.hi_step, "agree_original": round(agree, 4),
                   "n_compared": tot, "seconds": round(time.time() - t0, 1)},
                  open(run / f"fine_meta{suffix}.json", "w"), indent=2)
        log.info(f"[fine] wrote {run}/fine_probes{suffix}.npz ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
