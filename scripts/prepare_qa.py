"""Build QA JSONL for the selective-prediction / routing probes.

Tasks:
    mmlu     — cais/mmlu  (multiple-choice, 4 options)
    triviaqa — mandarjoshi/trivia_qa (rc.nocontext, open-ended)
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["mmlu", "triviaqa"], default="mmlu")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--split", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = np.random.default_rng(args.seed)
    rows = []

    if args.task == "mmlu":
        split = args.split or "test"
        ds = load_dataset("cais/mmlu", "all", split=split)
        logger.info("[mmlu] %d rows", len(ds))
        for r in ds:
            opts = r.get("choices")
            gold = r.get("answer")
            q = r.get("question")
            if not opts or gold is None or q is None or len(opts) != 4:
                continue
            rows.append({
                "task": "mmlu", "question": str(q),
                "options": [str(o) for o in opts], "gold_idx": int(gold),
            })
    else:
        split = args.split or "validation"
        ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split=split)
        logger.info("[triviaqa] %d rows", len(ds))
        for r in ds:
            q = r.get("question")
            ans = r.get("answer") or {}
            gold = ans.get("value") or ans.get("normalized_value")
            aliases = ans.get("aliases") or ans.get("normalized_aliases") or []
            if not q or not gold:
                continue
            rows.append({
                "task": "qa", "question": str(q), "gold": str(gold),
                "aliases": [str(a) for a in aliases],
            })

    if args.n and args.n < len(rows):
        idx = rng.choice(len(rows), size=args.n, replace=False)
        rows = [rows[i] for i in idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logger.info("wrote %d rows -> %s", len(rows), args.out)


if __name__ == "__main__":
    main()
