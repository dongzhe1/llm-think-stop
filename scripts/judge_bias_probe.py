"""LLM-as-judge bias probe: measure position, length, and self-preference bias.

Scores each pair in both orders using log-probability margin (A vs B token).
Quantifies position bias (order-swap flip rate), length bias (preference for
longer response), self-preference (own-family win-rate), and human agreement.
"""
from __future__ import annotations

import argparse
import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

JUDGE_SYS = "You are an impartial judge comparing two AI responses to a question."
JUDGE_TMPL = (
    "Question:\n{q}\n\n"
    "Response A:\n{a}\n\n"
    "Response B:\n{b}\n\n"
    "Which response better answers the question, considering helpfulness, "
    "correctness, and relevance? Answer with only the single letter A or B."
)


def load_pairs(path, n, seed=42):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    rng = np.random.default_rng(seed)
    if n and n < len(rows):
        rows = [rows[i] for i in rng.choice(len(rows), size=n, replace=False)]
    return rows


def _answer_token_ids(tokenizer, letters):
    ids = []
    for w in letters:
        for variant in (w, " " + w):
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
    return sorted(set(ids))


def _build_prompt(tokenizer, q, first, second, is_qwen3):
    content = JUDGE_TMPL.format(q=q[:4000], a=first[:4000], b=second[:4000])
    msgs = [
        {"role": "system", "content": JUDGE_SYS},
        {"role": "user", "content": content},
    ]
    if getattr(tokenizer, "chat_template", None):
        kw = dict(tokenize=False, add_generation_prompt=True)
        if is_qwen3:
            kw["enable_thinking"] = False
        return tokenizer.apply_chat_template(msgs, **kw)
    return JUDGE_SYS + "\n\n" + content + "\nAnswer:"


def _score_AB(model, tokenizer, prompts, a_ids, b_ids, batch, max_len):
    import torch

    margins = np.empty(len(prompts), dtype=np.float32)
    cur = 0
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        inp = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            logits = model(**inp).logits[:, -1, :].float()
        la = torch.logsumexp(logits[:, a_ids], dim=-1)
        lb = torch.logsumexp(logits[:, b_ids], dim=-1)
        m = (la - lb).cpu().numpy()
        margins[cur:cur + len(chunk)] = m
        cur += len(chunk)
    return margins


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--judge", required=True)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--judge_family", default="")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=3072)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_pairs(args.pairs, args.n, args.seed)
    logger.info("[probe] judge=%s pairs=%d family=%s",
                args.judge, len(rows), args.judge_family or "n/a")

    tokenizer = AutoTokenizer.from_pretrained(args.judge, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    is_qwen3 = "qwen3" in args.judge.lower().replace("/", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    kw = dict(trust_remote_code=True,
              device_map=("auto" if device == "cuda" else None))
    if args.load_in_4bit and device == "cuda":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        kw["torch_dtype"] = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.judge, **kw)
    if device == "cpu":
        model.to(device)
    model.eval()

    a_ids = _answer_token_ids(tokenizer, ["A", "a"])
    b_ids = _answer_token_ids(tokenizer, ["B", "b"])
    a_ids = [i for i in a_ids if i not in set(b_ids)]
    b_ids = [i for i in b_ids if i not in set(a_ids)]
    if not a_ids or not b_ids:
        raise RuntimeError(
            "Could not resolve distinct A/B token ids for this tokenizer")

    p1, p2 = [], []
    for r in rows:
        q, ra, rb = r["question"], r["response_a"], r["response_b"]
        p1.append(_build_prompt(tokenizer, q, ra, rb, is_qwen3))
        p2.append(_build_prompt(tokenizer, q, rb, ra, is_qwen3))

    logger.info("[probe] scoring order 1 ...")
    m1 = _score_AB(model, tokenizer, p1, a_ids, b_ids, args.batch, args.max_len)
    logger.info("[probe] scoring order 2 ...")
    m2 = _score_AB(model, tokenizer, p2, a_ids, b_ids, args.batch, args.max_len)

    # Margins: order1 >0 picks resp_a; order2 >0 picks resp_b
    pick_first_1 = m1 > 0
    pick_first_2 = m2 > 0
    a_wins_1 = m1 > 0
    a_wins_2 = m2 < 0
    flip = a_wins_1 != a_wins_2

    # Order-robust preference: average both "resp_a is better" margins
    robust_a = (m1 - m2) / 2.0
    pref_a = robust_a > 0

    # Length bias
    def _tlen(s):
        return len(tokenizer.encode(str(s), add_special_tokens=False))

    la = np.array([_tlen(r["response_a"]) for r in rows])
    lb = np.array([_tlen(r["response_b"]) for r in rows])
    longer_is_a = la > lb
    gap = (la - lb).astype(float)
    valid_len = la != lb
    prefers_longer = (pref_a == longer_is_a)[valid_len]
    corr = (float(np.corrcoef(robust_a[valid_len], gap[valid_len])[0, 1])
            if valid_len.sum() > 2 else float("nan"))

    n = len(rows)
    logger.info("\n================ JUDGE BIAS PROBE ================")
    logger.info("pairs scored: %d", n)
    logger.info("\n--- POSITION BIAS ---")
    logger.info("  flip rate: %.3f  (0 = perfectly consistent)", flip.mean())
    logger.info("  first-position win-rate: %.3f  (>0.5 = favors first)",
                np.mean(np.concatenate([pick_first_1, pick_first_2])))
    logger.info("\n--- LENGTH BIAS ---")
    logger.info("  P(pick longer): %.3f  (0.5 = no bias)", prefers_longer.mean())
    logger.info("  corr(margin, length_gap): %+.3f  (>0 = longer favored)", corr)

    # Self-preference
    fam = args.judge_family.lower().strip()
    if fam:
        own, other = [], []
        for r, pa in zip(rows, pref_a):
            ma = str(r.get("model_a", "") or "").lower()
            mb = str(r.get("model_b", "") or "").lower()
            a_own, b_own = fam in ma, fam in mb
            if a_own and not b_own:
                own.append(1.0 if pa else 0.0)
            elif b_own and not a_own:
                own.append(0.0 if pa else 1.0)
            elif not a_own and not b_own:
                other.append(1.0 if pa else 0.0)
        logger.info("\n--- SELF-PREFERENCE ---")
        if own:
            logger.info("  own-family win-rate (n=%d): %.3f", len(own), np.mean(own))
        else:
            logger.info("  (no pairs with exactly one own-family side)")

    # Human agreement
    has_h = [r for r in rows if r.get("human") in ("a", "b")]
    if has_h:
        idx = [i for i, r in enumerate(rows) if r.get("human") in ("a", "b")]
        hum = np.array([1 if rows[i]["human"] == "a" else 0 for i in idx])
        agree = (pref_a[idx].astype(int) == hum).mean()
        logger.info("\n--- HUMAN AGREEMENT (order-robust) ---")
        logger.info("  accuracy (n=%d): %.3f", len(idx), agree)

    # Verdict
    logger.info("\n--- VERDICT ---")
    big_pos = (flip.mean() > 0.12 or abs(
        np.mean(np.concatenate([pick_first_1, pick_first_2])) - 0.5) > 0.05)
    big_len = (abs(prefers_longer.mean() - 0.5) > 0.05 or abs(corr) > 0.10)
    if big_pos or big_len:
        logger.info("  GO: measurable bias -> calibrator has room.")
    else:
        logger.info("  NO-GO-ish: near-unbiased on this set; try LLMBar-Adversarial.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
