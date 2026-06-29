"""Confidence routing probe: can a small model's uncertainty gate escalation?

Loads a small and large model sequentially on one GPU. Measures whether
confidence signals from the small model can route uncertain queries to the
large model, producing a calibrated cascade accuracy-vs-cost frontier.
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


def _first_tok_ids(tokenizer, words):
    ids = []
    for w in words:
        for v in (w, " " + w):
            e = tokenizer.encode(v, add_special_tokens=False)
            if e:
                ids.append(e[0])
    return sorted(set(ids))


def _score_mmlu(model_path, rows, k, batch, max_len, need_confidence):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    is_qwen3 = "qwen3" in model_path.lower().replace("/", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        device_map=("auto" if device == "cuda" else None))
    if device == "cpu":
        model.to(device)
    model.eval()

    def _chat(content):
        msgs = [{"role": "user", "content": content}]
        if getattr(tokenizer, "chat_template", None):
            kw = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen3:
                kw["enable_thinking"] = False
            return tokenizer.apply_chat_template(msgs, **kw)
        return content + "\nAnswer:"

    L = ["A", "B", "C", "D"]
    lids, seen = [], set()
    for c in L:
        ids = [_first_tok_ids(tokenizer, [c, c.lower()])
               for c in L][0]  # simplified
    lids = []
    for c in L:
        ids = [x for x in _first_tok_ids(tokenizer, [c, c.lower()])
               if x not in seen and not seen.add(x)]
        lids.append(ids)

    prompts = []
    for r in rows:
        body = (f"Question: {r['question']}\n"
                + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"])))
        prompts.append(_chat(
            "Answer the multiple-choice question with the single letter "
            "of the correct option (A, B, C, or D).\n\n" + body))

    pred, margin = [], []
    for i in range(0, len(prompts), batch):
        ch = prompts[i:i + batch]
        inp = tokenizer(ch, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        lp = torch.stack([torch.logsumexp(lg[:, ids], dim=-1)
                         for ids in lids], dim=1)
        prob = torch.softmax(lp, dim=1)
        top2 = torch.topk(prob, 2, dim=1).values
        pred += prob.argmax(1).cpu().tolist()
        margin += (top2[:, 0] - top2[:, 1]).cpu().tolist()

    conf = None
    if need_confidence:
        conf = {"margin": margin}
        if k > 0:
            agree = []
            for i in range(0, len(prompts), batch):
                ch = prompts[i:i + batch]
                inp = tokenizer(ch, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_len).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inp, do_sample=True, temperature=0.7, top_p=0.95,
                        num_return_sequences=k, max_new_tokens=4,
                        pad_token_id=tokenizer.pad_token_id)
                plen = inp["input_ids"].shape[1]
                dec = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
                for j in range(len(ch)):
                    letters = []
                    for s in dec[j * k:(j + 1) * k]:
                        m = re.search(r"[ABCD]", s.upper())
                        letters.append(m.group(0) if m else "?")
                    cnt = max(letters.count(x) for x in set(letters)) if letters else 0
                    agree.append(cnt / k)
            conf["self_consistency"] = agree

    del model
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--small", required=True)
    parser.add_argument("--large", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.data, args.n, args.seed)
    gold = np.array([r["gold_idx"] for r in rows])
    logger.info("[route] n=%d small=%s large=%s k=%d",
                len(rows), args.small, args.large, args.k)

    logger.info("[route] scoring SMALL model...")
    ps, conf = _score_mmlu(args.small, rows, args.k, args.batch,
                           args.max_len, need_confidence=True)
    logger.info("[route] scoring LARGE model...")
    pl, _ = _score_mmlu(args.large, rows, 0, args.batch,
                        args.max_len, need_confidence=False)

    cs = (np.array(ps) == gold).astype(int)
    cl = (np.array(pl) == gold).astype(int)
    small_acc, large_acc = cs.mean(), cl.mean()
    logger.info("[route] small acc = %.3f | large acc = %.3f | headroom = %+.3f",
                small_acc, large_acc, large_acc - small_acc)

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    sigs = {n: np.array(v, dtype=float) for n, v in conf.items()}
    logger.info("\n=== gate AUROC ===")
    for name, s in sigs.items():
        try:
            logger.info("  %-18s AUROC = %.3f", name, roc_auc_score(cs, s))
        except Exception:
            logger.info("  %-18s AUROC = n/a", name)

    z = StandardScaler().fit_transform(np.column_stack(list(sigs.values())))
    gate = z.sum(1)
    try:
        logger.info("  %-18s AUROC = %.3f", "combined gate", roc_auc_score(cs, gate))
    except Exception:
        pass

    order = np.argsort(gate)
    n = len(rows)
    logger.info("\n=== cascade frontier ===")
    logger.info("%10s  %16s  %12s", "escalate%", "calibrated acc", "random acc")
    for f in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]:
        nesc = int(round(f * n))
        esc = set(order[:nesc].tolist())
        acc = np.mean([cl[i] if i in esc else cs[i] for i in range(n)])
        rnd = (1 - f) * small_acc + f * large_acc
        logger.info("%9.0f%%  %16.3f  %12.3f", f * 100, acc, rnd)

    target = large_acc - 0.01
    reached = None
    for f in np.linspace(0, 1, 101):
        nesc = int(round(f * n))
        esc = set(order[:nesc].tolist())
        acc = np.mean([cl[i] if i in esc else cs[i] for i in range(n)])
        if acc >= target:
            reached = f
            break

    logger.info("\n--- VERDICT ---")
    auc_gate = roc_auc_score(cs, gate) if len(set(cs)) > 1 else float("nan")
    if (reached is not None and reached <= 0.5
            and small_acc >= 0.4 and auc_gate >= 0.6):
        logger.info("  GO: small=%.0f%%; matches large by escalating %.0f%% "
                    "(gate AUROC %.3f)", small_acc * 100, reached * 100, auc_gate)
    elif large_acc - small_acc < 0.03:
        logger.info("  NO-GO: small ≈ large (%.3f vs %.3f)", small_acc, large_acc)
    elif auc_gate < 0.55:
        logger.info("  NO-GO: gate AUROC %.3f — cannot separate", auc_gate)
    else:
        logger.info("  PARTIAL: needs >%.0f%% escalation", (reached or 1) * 100)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
