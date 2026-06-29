"""Build reasoning-task JSONL files for the budget / learnstop / transfer probes.

Datasets are downloaded from HuggingFace Hub.

Tasks:
    gsm8k    — openai/gsm8k
    math500  — HuggingFaceH4/MATH-500
    mmlu_pro — TIGER-Lab/MMLU-Pro (options embedded in question text)
    aime     — AI-MO/aimo-validation-aime
    gpqa     — Idavidrein/gpqa (gpqa_diamond)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re

import numpy as np

logger = logging.getLogger(__name__)


def _gsm8k_gold(ans: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", ans)
    return (m.group(1) if m else ans).replace(",", "").strip().rstrip(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["gsm8k", "math500", "mmlu_pro", "aime", "gpqa"],
                        default="gsm8k")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--split", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    rng = np.random.default_rng(args.seed)
    rows = []

    if args.task == "gsm8k":
        split = args.split or "test"
        ds = load_dataset("openai/gsm8k", "main", split=split)
        logger.info("[gsm8k] %d rows", len(ds))
        for r in ds:
            q, a = r.get("question"), r.get("answer")
            if not q or not a:
                continue
            rows.append({"task": "gsm8k", "question": str(q),
                         "gold": _gsm8k_gold(str(a))})

    elif args.task == "math500":
        split = args.split or "test"
        ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
        logger.info("[math500] %d rows", len(ds))
        for r in ds:
            q, a = r.get("problem"), r.get("answer")
            if not q or a is None:
                continue
            rows.append({"task": "math", "question": str(q),
                         "gold": str(a).strip()})

    elif args.task == "mmlu_pro":
        split = args.split or "test"
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        logger.info("[mmlu_pro] %d rows", len(ds))
        LETTERS = "ABCDEFGHIJ"
        for r in ds:
            q = r.get("question")
            opts = r.get("options") or []
            ans_idx = r.get("answer_index")
            if not q or not opts or ans_idx is None or ans_idx >= len(opts):
                continue
            opts_text = "\n".join(
                f"{LETTERS[i]}. {o}" for i, o in enumerate(opts[:10]))
            rows.append({
                "task": "mmlu_pro",
                "question": str(q) + "\n" + opts_text,
                "gold": LETTERS[ans_idx],
                "category": str(r.get("category", "")),
            })

    elif args.task == "aime":
        split = args.split or "train"
        ds = load_dataset("AI-MO/aimo-validation-aime", split=split)
        logger.info("[aime] %d rows", len(ds))
        for r in ds:
            q = r.get("problem") or r.get("question")
            a = r.get("answer")
            if not q or a is None:
                continue
            rows.append({"task": "aime", "question": str(q),
                         "gold": str(a).strip()})

    else:  # gpqa
        split = args.split or "train"
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)
        logger.info("[gpqa] %d rows", len(ds))
        skipped = 0
        for r in ds:
            q = r.get("Question", "").strip()
            correct_text = r.get("Correct Answer", "").strip()
            wrong = [r.get(f"Incorrect Answer {i}", "").strip()
                     for i in [1, 2, 3]]
            if not q or not correct_text:
                skipped += 1
                continue
            choices = [correct_text] + wrong
            q_rng = np.random.default_rng(abs(hash(q)) % (2 ** 31))
            perm = q_rng.permutation(4)
            shuffled = [choices[p] for p in perm]
            correct_letter = "ABCD"[int(np.where(perm == 0)[0][0])]
            opts_text = "\n".join(
                f"{L}. {c}" for L, c in zip("ABCD", shuffled))
            rows.append({
                "task": "gpqa",
                "question": str(q) + "\n" + opts_text,
                "gold": correct_letter,
            })
        if skipped:
            logger.info("[gpqa] skipped %d rows", skipped)

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
