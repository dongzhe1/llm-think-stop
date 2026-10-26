from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grading_v2 import find_boxed

LETTERS = "ABCDEFGHIJ"


def gsm8k_gold(ans: str) -> str:
    m = re.search(r"####\s*([-\d,\.]+)", ans)
    return (m.group(1) if m else ans).replace(",", "").strip().rstrip(".")


def stable_seed(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)


def sample(rows, n, seed):
    if not n or n >= len(rows):
        return list(rows)
    idx = np.random.default_rng(seed).choice(len(rows), size=n, replace=False)
    return [rows[i] for i in sorted(idx)]


def tag(rows, task, split):
    for k, r in enumerate(rows):
        r["id"] = f"{task}-{split}-{k:05d}"
    return rows


def build_gsm8k(args):
    from datasets import load_dataset

    def conv(ds):
        return [{"task": "gsm8k", "mode": "numeric", "n_options": 0,
                 "question": str(r["question"]), "gold": gsm8k_gold(str(r["answer"]))} for r in ds]
    ev = conv(load_dataset("openai/gsm8k", "main", split="test"))
    tr = sample(conv(load_dataset("openai/gsm8k", "main", split="train")), args.n_train, args.seed)
    return sample(ev, args.n_eval_gsm8k, args.seed), tr


def _load_math_train():
    from datasets import load_dataset
    subjects = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"]
    errs = []
    try:
        rows = []
        for s in subjects:
            rows += list(load_dataset("EleutherAI/hendrycks_math", s, split="train"))
        return rows
    except Exception as e:
        errs.append(f"EleutherAI/hendrycks_math: {e}")
    try:
        return list(load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train"))
    except Exception as e:
        errs.append(f"DigitalLearningGmbH/MATH-lighteval: {e}")
    raise RuntimeError("could not load a MATH train split:\n  " + "\n  ".join(errs))


def build_math500(args):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ev = [{"task": "math500", "mode": "math", "n_options": 0, "question": str(r["problem"]),
           "gold": str(r["answer"]).strip(), "level": r.get("level", ""), "category": r.get("subject", "")}
          for r in ds]
    seen = {r["question"].strip() for r in ev}
    pool = []
    for r in _load_math_train():
        q = str(r["problem"]).strip()
        g = find_boxed(str(r["solution"]), last=True)
        if not q or g is None or q in seen:
            continue
        pool.append({"task": "math500", "mode": "math", "n_options": 0, "question": q, "gold": g.strip(),
                     "level": str(r.get("level", "")), "category": str(r.get("type", ""))})
    print(f"[math500] eval={len(ev)}  train-pool candidates={len(pool)}")
    return ev, sample(pool, args.n_train, args.seed)


def build_mmlu_pro(args):
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rows = []
    for r in ds:
        opts = list(r.get("options") or [])[:10]
        ai = r.get("answer_index")
        if not r.get("question") or not opts or ai is None or ai >= len(opts):
            continue
        q = str(r["question"]) + "\n" + "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(opts))
        rows.append({"task": "mmlu_pro", "mode": "mc", "n_options": len(opts), "question": q,
                     "gold": LETTERS[ai], "category": str(r.get("category", ""))})
    perm = np.random.default_rng(args.seed).permutation(len(rows))
    ev = [rows[i] for i in sorted(perm[:args.n_eval_mmlu])]
    tr = [rows[i] for i in sorted(perm[args.n_eval_mmlu:args.n_eval_mmlu + args.n_train])]
    return ev, tr


def _gpqa_rows(cfg):
    from datasets import load_dataset
    out = []
    for r in load_dataset("Idavidrein/gpqa", cfg, split="train"):
        q = str(r.get("Question", "")).strip()
        c = str(r.get("Correct Answer", "")).strip()
        w = [str(r.get(f"Incorrect Answer {i}", "")).strip() for i in (1, 2, 3)]
        if not q or not c:
            continue
        perm = np.random.default_rng(stable_seed(q)).permutation(4)
        choices = [c] + w
        shuffled = [choices[p] for p in perm]
        gold = "ABCD"[int(np.where(perm == 0)[0][0])]
        out.append({"task": "gpqa", "mode": "mc", "n_options": 4,
                    "question": q + "\n" + "\n".join(f"{L}. {t}" for L, t in zip("ABCD", shuffled)),
                    "gold": gold, "_q": q})
    return out


def build_gpqa(args):
    ev = _gpqa_rows("gpqa_diamond")
    seen = {r["_q"] for r in ev}
    tr = [r for r in _gpqa_rows("gpqa_extended") if r["_q"] not in seen]
    for r in ev + tr:
        r.pop("_q", None)
    print(f"[gpqa] eval={len(ev)}  train={len(tr)}")
    return ev, sample(tr, args.n_train, args.seed)


def build_aqua(args):
    from datasets import load_dataset

    def conv(ds):
        out = []
        for r in ds:
            opts = [re.sub(r"^\s*\(?([A-E])\)\s*", r"\1. ", o) for o in r["options"]]
            out.append({"task": "aqua", "mode": "mc", "n_options": len(opts),
                        "question": str(r["question"]) + "\n" + "\n".join(opts),
                        "gold": str(r["correct"]).strip().upper()})
        return out
    ev = conv(load_dataset("deepmind/aqua_rat", "raw", split="test")) + \
        conv(load_dataset("deepmind/aqua_rat", "raw", split="validation"))
    tr = sample(conv(load_dataset("deepmind/aqua_rat", "raw", split="train")), args.n_train, args.seed)
    return ev, tr


def build_bbh_free(args):
    from datasets import load_dataset
    ev, tr = [], []
    for sub in ("dyck_languages", "word_sorting"):
        rows = [{"task": "bbh_free", "mode": "text", "n_options": 0, "category": sub,
                 "question": str(r["input"]), "gold": str(r["target"]).strip()}
                for r in load_dataset("lukaemon/bbh", sub, split="test")]
        perm = np.random.default_rng(args.seed).permutation(len(rows))
        ev += [rows[i] for i in sorted(perm[:150])]
        tr += [rows[i] for i in sorted(perm[150:250])]
    return ev, tr


BUILDERS = {"gsm8k": build_gsm8k, "math500": build_math500, "mmlu_pro": build_mmlu_pro,
            "gpqa": build_gpqa, "aqua": build_aqua, "bbh_free": build_bbh_free}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="gsm8k,math500,mmlu_pro")
    ap.add_argument("--out_dir", default="data/v2")
    ap.add_argument("--n_train", type=int, default=500)
    ap.add_argument("--n_eval_gsm8k", type=int, default=0, help="0 = full 1319-question test split")
    ap.add_argument("--n_eval_mmlu", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        ev, tr = BUILDERS[task](args)
        evq = {r["question"].strip() for r in ev}
        n_overlap = sum(r["question"].strip() in evq for r in tr)
        assert n_overlap == 0, f"{task}: {n_overlap} train questions overlap the eval set"
        for split, rows in (("eval", tag(ev, task, "eval")), ("train", tag(tr, task, "train"))):
            p = os.path.join(args.out_dir, f"{task}_{split}.jsonl")
            with open(p, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"wrote {len(rows):5d} -> {p}")


if __name__ == "__main__":
    main()
