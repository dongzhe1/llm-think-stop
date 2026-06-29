"""Build pairwise-comparison JSONL for the judge-bias probe.

Sources:
    rewardbench — allenai/reward-bench (carries model identities)
    mtbench     — lmsys/mt_bench_human_judgments
    llmbar      — princeton-nlp/LLMBar (fetched from GitHub raw)

The a/b slot is randomized (seeded) so position is uncorrelated with quality.
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)


def _as_text(x):
    if isinstance(x, list):
        parts = []
        for m in x:
            parts.append(str(m.get("content", "")) if isinstance(m, dict)
                         else str(m))
        return "\n".join(p for p in parts if p)
    return str(x)


def _get(r, *names):
    for n in names:
        if n in r and r[n] is not None:
            return r[n]
    return None


def iter_rewardbench():
    from datasets import get_dataset_split_names, load_dataset

    splits = get_dataset_split_names("allenai/reward-bench")
    split = "filtered" if "filtered" in splits else splits[0]
    logger.info("[rewardbench] split=%s (available: %s)", split, splits)
    ds = load_dataset("allenai/reward-bench", split=split)
    logger.info("[rewardbench] %d rows", len(ds))
    for r in ds:
        q = _get(r, "prompt", "input", "question")
        chosen = _get(r, "chosen", "response_chosen")
        rejected = _get(r, "rejected", "response_rejected")
        cm = _get(r, "chosen_model", "model_chosen")
        rm = _get(r, "rejected_model", "model_rejected")
        if q is None or chosen is None or rejected is None:
            continue
        yield dict(q=str(q), better=_as_text(chosen), worse=_as_text(rejected),
                   better_model=cm, worse_model=rm)


def iter_mtbench():
    from datasets import load_dataset

    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    logger.info("[mtbench] %d rows", len(ds))
    for r in ds:
        w = _get(r, "winner")
        if w not in ("model_a", "model_b"):
            continue
        ca, cb = _get(r, "conversation_a"), _get(r, "conversation_b")

        def q_of(conv):
            for m in conv:
                if m.get("role") == "user":
                    return m.get("content", "")
            return ""

        def a_of(conv):
            outs = [m.get("content", "") for m in conv
                    if m.get("role") == "assistant"]
            return outs[-1] if outs else ""

        better_is_a = (w == "model_a")
        yield dict(
            q=str(q_of(ca)),
            better=a_of(ca) if better_is_a else a_of(cb),
            worse=a_of(cb) if better_is_a else a_of(ca),
            better_model=_get(r, "model_a") if better_is_a else _get(r, "model_b"),
            worse_model=_get(r, "model_b") if better_is_a else _get(r, "model_a"),
        )


def iter_llmbar(config):
    import urllib.request

    cfg = config.replace("\\", "/").strip("/")
    urls = [
        f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar/{cfg}/dataset.json",
        f"https://raw.githubusercontent.com/princeton-nlp/LLMBar/master/Dataset/LLMBar/{cfg}/dataset.json",
    ]
    data = None
    for url in urls:
        try:
            logger.info("[llmbar] fetching %s", url)
            with urllib.request.urlopen(url) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception:
            logger.warning("  fetch failed, trying next URL")
    if data is None:
        raise SystemExit(
            "[llmbar] could not fetch dataset.json. "
            "Available configs: Natural, Adversarial/Neighbor, "
            "Adversarial/GPTInst, Adversarial/GPTOut, Adversarial/Manual."
        )
    for r in data:
        label = int(_get(r, "label", "preference"))
        o1, o2 = _get(r, "output_1", "output1"), _get(r, "output_2", "output2")
        q = _get(r, "input", "instruction")
        if q is None or o1 is None or o2 is None:
            continue
        yield dict(q=str(q),
                   better=str(o1) if label == 1 else str(o2),
                   worse=str(o2) if label == 1 else str(o1),
                   better_model=None, worse_model=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["rewardbench", "mtbench", "llmbar"],
                        default="rewardbench")
    parser.add_argument("--config", default="Natural")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    it = {
        "rewardbench": iter_rewardbench,
        "mtbench": iter_mtbench,
        "llmbar": lambda: iter_llmbar(args.config),
    }[args.source]()

    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for r in it:
            if rng.random() < 0.5:
                a, b, ma, mb, human = (r["better"], r["worse"],
                                       r["better_model"], r["worse_model"], "a")
            else:
                a, b, ma, mb, human = (r["worse"], r["better"],
                                       r["worse_model"], r["better_model"], "b")
            if not a.strip() or not b.strip():
                continue
            f.write(json.dumps({
                "question": r["q"], "response_a": a, "response_b": b,
                "human": human, "model_a": ma, "model_b": mb,
            }) + "\n")
            n += 1
            if args.limit and n >= args.limit:
                break
    logger.info("wrote %d pairs -> %s (source=%s)", n, args.out, args.source)


if __name__ == "__main__":
    main()
