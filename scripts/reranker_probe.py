"""Reranker calibration probe: ECE, ranking quality, and cross-reranker disagreement.

Scores BEIR candidate pools with multiple rerankers (causal LLM logprob scoring
or cross-encoder). Reports per-reranker ECE, nDCG@10, MRR, and Kendall-tau
disagreement between reranker pairs.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from itertools import combinations

import numpy as np

logger = logging.getLogger(__name__)


def load_pairs(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def _first_tok_ids(tokenizer, words):
    ids = []
    for w in words:
        for v in (w, " " + w):
            e = tokenizer.encode(v, add_special_tokens=False)
            if e:
                ids.append(e[0])
    return sorted(set(ids))


def score_causal(path, rows, batch, max_len):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    is_qwen3 = "qwen3" in path.lower().replace("/", "")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device == "cuda" else torch.float32),
        device_map=("auto" if device == "cuda" else None))
    if device == "cpu":
        model.to(device)
    model.eval()

    def _prompt(q, p):
        c = (
            "Judge whether the Document is relevant to the Query. "
            "Answer with only 'yes' or 'no'.\n"
            f"Query: {q}\nDocument: {p}\nRelevant:"
        )
        if getattr(tokenizer, "chat_template", None):
            kw = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen3:
                kw["enable_thinking"] = False
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": c}], **kw)
        return c

    y_ids = _first_tok_ids(tokenizer, ["yes", "Yes", "YES"])
    n_ids = _first_tok_ids(tokenizer, ["no", "No", "NO"])
    y_ids = [i for i in y_ids if i not in set(n_ids)]
    n_ids = [i for i in n_ids if i not in set(y_ids)]

    probs = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), batch):
        ch = rows[i:i + batch]
        ps = [_prompt(r["query"], r["passage"]) for r in ch]
        inp = tokenizer(ps, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(model.device)
        with torch.no_grad():
            lg = model(**inp).logits[:, -1, :].float()
        ly = torch.logsumexp(lg[:, y_ids], dim=-1)
        ln = torch.logsumexp(lg[:, n_ids], dim=-1)
        probs[i:i + len(ch)] = torch.sigmoid(ly - ln).cpu().numpy()

    del model
    if device == "cuda":
        import torch as _t
        _t.cuda.empty_cache()
    return probs


def score_crossencoder(path, rows, batch):
    from sentence_transformers import CrossEncoder

    ce = CrossEncoder(path, max_length=512)
    raw = ce.predict([(r["query"], r["passage"]) for r in rows],
                     batch_size=batch, show_progress_bar=False)
    raw = np.asarray(raw, dtype=np.float32).ravel()
    return 1.0 / (1.0 + np.exp(-raw)) if (raw.min() < 0 or raw.max() > 1) else raw


def _ece(probs, labels, bins=10):
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    n = len(probs)
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (probs >= lo) & (probs < hi if b < bins - 1 else probs <= hi)
        if mask.sum():
            e += (mask.sum() / n) * abs(labels[mask].mean() - probs[mask].mean())
    return e


def _ndcg_mrr(by_q):
    ndcgs, mrrs = [], []
    for scores, labels in by_q.values():
        order = np.argsort(-np.asarray(scores, dtype=float))
        lab = np.asarray(labels, dtype=float)[order]
        k = min(10, len(lab))
        dcg = sum((2 ** lab[i] - 1) / np.log2(i + 2) for i in range(k))
        ideal = np.sort(lab)[::-1]
        idcg = sum((2 ** ideal[i] - 1) / np.log2(i + 2) for i in range(min(k, len(ideal))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        rr = 0.0
        for i, l in enumerate(lab):
            if l > 0:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)
    return float(np.mean(ndcgs)), float(np.mean(mrrs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--rerankers", required=True)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_len", type=int, default=512)
    args = parser.parse_args()

    rows = load_pairs(args.pairs)
    labels = np.array([int(r["label"]) for r in rows])
    qids = [r["qid"] for r in rows]
    logger.info("[probe] pairs=%d queries=%d pos-rate=%.3f",
                len(rows), len(set(qids)), labels.mean())

    specs = []
    for tok_ in args.rerankers.split(","):
        name, path, typ = tok_.split("=")
        specs.append((name.strip(), path.strip(), typ.strip()))

    prob_by_reranker = {}
    for name, path, typ in specs:
        logger.info("[probe] scoring '%s' (%s) ...", name, typ)
        if typ == "crossencoder":
            p = score_crossencoder(path, rows, args.batch)
        else:
            p = score_causal(path, rows, args.batch, args.max_len)
        prob_by_reranker[name] = np.asarray(p, dtype=float)

    logger.info("\n=== per-reranker: ECE and ranking ===")
    logger.info("%-12s  %8s  %10s  %8s", "reranker", "ECE", "nDCG@10", "MRR")
    for name in prob_by_reranker:
        p = prob_by_reranker[name]
        byq = defaultdict(lambda: ([], []))
        for i, q in enumerate(qids):
            byq[q][0].append(p[i])
            byq[q][1].append(labels[i])
        nd, mr = _ndcg_mrr(byq)
        logger.info("%-12s  %8.3f  %10.3f  %8.3f", name, _ece(p, labels), nd, mr)

    # Cross-reranker disagreement
    try:
        from scipy.stats import kendalltau as _kendalltau
    except ImportError:
        _kendalltau = None

    logger.info("\n=== cross-reranker Kendall tau ===")
    names = list(prob_by_reranker)
    q_to_idx = defaultdict(list)
    for i, q in enumerate(qids):
        q_to_idx[q].append(i)

    taus_all = []
    if _kendalltau and len(names) >= 2:
        for a, b in combinations(names, 2):
            taus = []
            for q, idx in q_to_idx.items():
                if len(idx) >= 3:
                    t, _ = _kendalltau(prob_by_reranker[a][idx],
                                       prob_by_reranker[b][idx])
                    if t == t:  # not NaN
                        taus.append(t)
            mt = float(np.mean(taus)) if taus else float("nan")
            taus_all.append(mt)
            logger.info("  %s vs %s: tau = %+.3f", a, b, mt)
    mean_tau = float(np.mean(taus_all)) if taus_all else float("nan")

    # Fusion preview
    fused = np.mean(np.column_stack(list(prob_by_reranker.values())), axis=1)
    byq_f = defaultdict(lambda: ([], []))
    for i, q in enumerate(qids):
        byq_f[q][0].append(fused[i])
        byq_f[q][1].append(labels[i])
    fnd, _ = _ndcg_mrr(byq_f)
    logger.info("\n=== fusion preview ===")
    logger.info("  fusion ECE=%.3f  nDCG@10=%.3f", _ece(fused, labels), fnd)

    logger.info("\n--- VERDICT ---")
    worst_ece = max(_ece(prob_by_reranker[n], labels) for n in names)
    if worst_ece > 0.10 and (mean_tau == mean_tau and mean_tau < 0.8):
        logger.info("  GO: miscalibrated (max ECE %.3f) and disagree (tau %.3f)",
                    worst_ece, mean_tau)
    elif worst_ece <= 0.10:
        logger.info("  NO-GO: well-calibrated (max ECE %.3f)", worst_ece)
    else:
        logger.info("  PARTIAL: max ECE %.3f, mean tau %.3f", worst_ece, mean_tau)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    main()
