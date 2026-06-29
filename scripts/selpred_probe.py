"""Selective prediction probe: do internal signals separate correct from wrong?

For each question, collects confidence signals (margin, max-prob, neg-entropy,
self-consistency) and measures AUROC of each signal and a logistic fusion for
predicting model correctness.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import string

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


def _normalize(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _qa_correct(pred, gold, aliases):
    p = _normalize(pred)
    golds = [_normalize(g) for g in [gold] + list(aliases) if g]
    if not p:
        return 0
    return int(any(p == g or (len(g) >= 3 and g in p) for g in golds))


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_jsonl(args.data, args.n, args.seed)
    task = rows[0].get("task", "mmlu")
    logger.info("[probe] task=%s n=%d model=%s k=%d",
                task, len(rows), args.model, args.k)

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

    def _chat(content):
        msgs = [{"role": "user", "content": content}]
        if getattr(tokenizer, "chat_template", None):
            kw = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen3:
                kw["enable_thinking"] = False
            return tokenizer.apply_chat_template(msgs, **kw)
        return content + "\nAnswer:"

    signals = {}
    correct = []

    if task == "mmlu":
        L = ["A", "B", "C", "D"]
        lids = [_first_tok_ids(tokenizer, [c, c.lower()]) for c in L]
        seen = set()
        for i in range(len(lids)):
            lids[i] = [x for x in lids[i] if x not in seen and not seen.add(x)]

        prompts = []
        for r in rows:
            body = (f"Question: {r['question']}\n"
                    + "\n".join(f"{L[i]}. {o}" for i, o in enumerate(r["options"])))
            prompts.append(_chat(
                "Answer the multiple-choice question with the single letter "
                "of the correct option (A, B, C, or D).\n\n" + body))

        margin, maxp, negent, pred = [], [], [], []
        for i in range(0, len(prompts), args.batch):
            ch = prompts[i:i + args.batch]
            inp = tokenizer(ch, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_len).to(model.device)
            with torch.no_grad():
                lg = model(**inp).logits[:, -1, :].float()
            lp = torch.stack([torch.logsumexp(lg[:, ids], dim=-1)
                             for ids in lids], dim=1)
            prob = torch.softmax(lp, dim=1)
            top2 = torch.topk(prob, 2, dim=1).values
            margin += (top2[:, 0] - top2[:, 1]).cpu().tolist()
            maxp += top2[:, 0].cpu().tolist()
            negent += (-(-(prob * torch.log(prob + 1e-9)).sum(1))).cpu().tolist()
            pred += prob.argmax(1).cpu().tolist()

        signals["margin"] = margin
        signals["max_prob"] = maxp
        signals["neg_entropy"] = negent
        correct = [int(p == r["gold_idx"]) for p, r in zip(pred, rows)]

        if args.k > 0:
            agree = []
            for i in range(0, len(prompts), args.batch):
                ch = prompts[i:i + args.batch]
                inp = tokenizer(ch, return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_len).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inp, do_sample=True, temperature=args.temp, top_p=0.95,
                        num_return_sequences=args.k, max_new_tokens=4,
                        pad_token_id=tokenizer.pad_token_id)
                plen = inp["input_ids"].shape[1]
                dec = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
                for j in range(len(ch)):
                    letters = []
                    for s in dec[j * args.k:(j + 1) * args.k]:
                        m = re.search(r"[ABCD]", s.upper())
                        letters.append(m.group(0) if m else "?")
                    cnt = max(letters.count(x) for x in set(letters)) if letters else 0
                    agree.append(cnt / args.k)
            signals["self_consistency"] = agree

    else:  # triviaqa
        prompts = [
            _chat(f"Answer the question with a short answer of a few words.\n\n"
                  f"Question: {r['question']}")
            for r in rows
        ]
        seqlp, ftmax, preds = [], [], []
        for i in range(0, len(prompts), args.batch):
            ch = prompts[i:i + args.batch]
            inp = tokenizer(ch, return_tensors="pt", padding=True,
                            truncation=True, max_length=args.max_len).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inp, do_sample=False, max_new_tokens=24,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True, output_scores=True)
            plen = inp["input_ids"].shape[1]
            gen = out.sequences[:, plen:]
            scores = torch.stack(out.scores, dim=1)
            lprobs = torch.log_softmax(scores.float(), dim=-1)
            for b in range(len(ch)):
                toks = gen[b]
                mask = toks != tokenizer.pad_token_id
                tlp = lprobs[b, torch.arange(len(toks)), toks]
                tlp = tlp[mask[:len(tlp)]] if mask.any() else tlp
                seqlp.append(float(tlp.mean()) if len(tlp) else -20.0)
                ftmax.append(float(lprobs[b, 0].max()))
                preds.append(tokenizer.decode(toks, skip_special_tokens=True).strip().split("\n")[0])
        signals["seq_logprob"] = seqlp
        signals["first_tok_maxlp"] = ftmax
        correct = [_qa_correct(p, r["gold"], r.get("aliases", []))
                   for p, r in zip(preds, rows)]

        if args.k > 0:
            agree = []
            for i in range(0, len(prompts), args.batch):
                ch = prompts[i:i + args.batch]
                inp = tokenizer(ch, return_tensors="pt", padding=True,
                                truncation=True, max_length=args.max_len).to(model.device)
                with torch.no_grad():
                    out = model.generate(
                        **inp, do_sample=True, temperature=args.temp, top_p=0.95,
                        num_return_sequences=args.k, max_new_tokens=24,
                        pad_token_id=tokenizer.pad_token_id)
                plen = inp["input_ids"].shape[1]
                dec = tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)
                for j in range(len(ch)):
                    ans = [_normalize(s.strip().split("\n")[0])
                           for s in dec[j * args.k:(j + 1) * args.k]]
                    cnt = max(ans.count(x) for x in set(ans)) if ans else 0
                    agree.append(cnt / args.k)
            signals["self_consistency"] = agree

    # AUROC of each signal + logistic fusion
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array(correct)
    logger.info("\n[probe] accuracy = %.3f (n=%d, correct=%d)",
                y.mean(), len(y), int(y.sum()))
    if y.min() == y.max():
        logger.info("[probe] all-correct or all-wrong -> AUROC undefined")
        return

    names = list(signals)
    logger.info("\n=== AUROC of correct-vs-wrong ===")
    aucs = {}
    for name in names:
        s = np.array(signals[name], dtype=float)
        try:
            a = roc_auc_score(y, s)
        except Exception:
            a = float("nan")
        aucs[name] = a
        logger.info("  %-18s AUROC = %.3f", name, a)

    best = max((a, name) for name, a in aucs.items() if a == a)
    x = np.column_stack([np.array(signals[name], dtype=float) for name in names])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(clf, x, y, cv=5, method="predict_proba")[:, 1]
    fus = roc_auc_score(y, p)
    logger.info("  %-18s AUROC = %.3f  (5-fold CV)", "FUSION (logistic)", fus)

    logger.info("\n--- VERDICT ---")
    if fus > best[0] + 0.01 and fus > 0.6:
        logger.info("  GO: fusion %.3f beats best single (%s %.3f)", fus, best[1], best[0])
    elif best[0] > 0.6:
        logger.info("  PARTIAL: best single (%s %.3f) already separates well", best[1], best[0])
    else:
        logger.info("  NO-GO-ish: signals weak (best %s %.3f)", best[1], best[0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
