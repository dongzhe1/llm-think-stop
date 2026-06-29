"""Build reranking JSONL (candidate pools with relevance labels) for the reranker probe.

For each query we take its relevant passages (label 1) plus K random non-relevant
passages from the corpus (label 0). This is sufficient to measure reranker
calibration and cross-reranker disagreement.

Source: BEIR datasets on HuggingFace (scifact, fiqa, nfcorpus, trec-covid, ...).
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _field(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="scifact")
    parser.add_argument("--qrels_split", default="test")
    parser.add_argument("--n_queries", type=int, default=100)
    parser.add_argument("--n_neg", type=int, default=20)
    parser.add_argument("--max_doc_chars", type=int, default=1200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = np.random.default_rng(args.seed)
    base = f"BeIR/{args.dataset}"

    corpus = load_dataset(base, "corpus")["corpus"]
    queries = load_dataset(base, "queries")["queries"]
    qrels = load_dataset(f"{base}-qrels")[args.qrels_split]
    logger.info("[beir/%s] corpus=%d queries=%d qrels=%d",
                args.dataset, len(corpus), len(queries), len(qrels))

    cdoc = {}
    for r in corpus:
        cid = str(_field(r, "_id", "id", "doc_id"))
        title = _field(r, "title") or ""
        text = _field(r, "text") or ""
        cdoc[cid] = (str(title) + " " + str(text)).strip()[:args.max_doc_chars]

    qtext = {
        str(_field(r, "_id", "id")): str(_field(r, "text", "query"))
        for r in queries
    }
    all_pids = list(cdoc)

    rel = {}
    for r in qrels:
        qid = str(_field(r, "query-id", "query_id", "qid"))
        pid = str(_field(r, "corpus-id", "corpus_id", "doc_id", "pid"))
        score = _field(r, "score", "relevance", "label")
        if score is None:
            continue
        rel.setdefault(qid, {})[pid] = int(score)

    qids = [q for q in rel if q in qtext]
    if args.n_queries and args.n_queries < len(qids):
        qids = [qids[i] for i in
                rng.choice(len(qids), size=args.n_queries, replace=False)]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    npairs = 0
    with open(args.out, "w") as f:
        for qid in qids:
            pos = [p for p, s in rel[qid].items() if s > 0 and p in cdoc]
            if not pos:
                continue
            negpool = [p for p in all_pids if p not in rel[qid]]
            negs = list(rng.choice(negpool,
                                   size=min(args.n_neg, len(negpool)),
                                   replace=False)) if negpool else []
            for p in pos:
                f.write(json.dumps({
                    "qid": qid, "query": qtext[qid], "pid": p,
                    "passage": cdoc[p], "label": 1,
                }) + "\n")
                npairs += 1
            for p in negs:
                f.write(json.dumps({
                    "qid": qid, "query": qtext[qid], "pid": str(p),
                    "passage": cdoc[str(p)], "label": 0,
                }) + "\n")
                npairs += 1
    logger.info("wrote %d pairs over %d queries -> %s",
                npairs, len(qids), args.out)


if __name__ == "__main__":
    main()
