from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_fine_grid import fine_budgets
from reason_learnstop_v2 import DEFAULT_BUDGETS, PROBE_TAIL

N_EVAL, N_TRAIN, ANS_CAP = 240, 160, 48
TASKS = {"gsm8k": dict(mode="numeric", n_options=0, max_think=8192, think=1200, acc=0.92),
         "math500": dict(mode="math", n_options=0, max_think=16384, think=3500, acc=0.85),
         "mmlu_pro": dict(mode="mc", n_options=10, max_think=12288, think=2000, acc=0.72)}
MODELS = {"Qwen3-32B": dict(family="qwen3", scale=0.9, acc=0.03, nothink=0.0),
          "Qwen3-8B": dict(family="qwen3", scale=1.0, acc=0.0, nothink=0.0),
          "DeepSeek-R1-Distill-Qwen-7B": dict(family="deepseek_r1", scale=1.3, acc=-0.08, nothink=0.0),
          "AdaptThink-7B-delta0.05": dict(family="deepseek_r1", scale=0.6, acc=-0.08, nothink=0.5)}
SETTINGS = [("gsm8k", "Qwen3-32B"), ("gsm8k", "DeepSeek-R1-Distill-Qwen-7B"), ("gsm8k", "AdaptThink-7B-delta0.05"),
            ("math500", "Qwen3-8B"), ("mmlu_pro", "Qwen3-8B")]
SEED_SETTINGS = [("gsm8k", "Qwen3-32B"), ("mmlu_pro", "Qwen3-8B")]
CATEGORIES = ["math", "physics", "chemistry", "engineering", "law", "history", "psychology", "business"]
LETTERS = "ABCDEFGHIJ"


def answer_text(mode: str, ans: str) -> str:
    return f" \\boxed{{{ans}}}" if mode == "math" else f" {ans}"


def golds(rng, mode: str, n_options: int, N: int) -> np.ndarray:
    if mode == "mc":
        return np.array([LETTERS[k] for k in rng.integers(0, n_options, N)])
    if mode == "math":
        return np.array([f"\\frac{{{a}}}{{{a + b}}}" if rng.random() < 0.3 else str(a)
                         for a, b in zip(rng.integers(1, 60, N), rng.integers(1, 9, N))])
    return np.array([str(x) for x in rng.integers(2, 999, N)])


def wrong(rng, mode: str, n_options: int, gold: str) -> str:
    if mode == "mc":
        return rng.choice([c for c in LETTERS[:n_options] if c != gold])
    return str(int(rng.integers(1000, 1100)))


class Toy:

    def __init__(self, rng, task: str, model: str, N: int):
        t, mo = TASKS[task], MODELS[model]
        self.rng, self.mode, self.nopt, self.max_think = rng, t["mode"], t["n_options"], t["max_think"]
        think = np.exp(rng.normal(np.log(t["think"] * mo["scale"]), 0.6, N))
        nothink = rng.random(N) < mo["nothink"]
        self.think = np.where(nothink, rng.integers(1, 5, N), np.clip(np.round(think), 20, t["max_think"])).astype(int)
        self.closed = self.think < t["max_think"]
        self.F = rng.random(N) < np.clip(t["acc"] + mo["acc"], 0.05, 0.98)
        self.t_ok = np.where(self.F, self.think * rng.beta(2.0, 3.0, N), np.inf)
        self.gold = golds(rng, self.mode, self.nopt, N)
        self.wrong = [wrong(rng, self.mode, self.nopt, g) for g in self.gold]

    def probe(self, i: int, B: float) -> tuple[str, bool]:
        ok = B >= self.t_ok[i]
        if self.rng.random() < 0.05:
            ok = not ok
        if ok:
            return answer_text(self.mode, self.gold[i]), True
        w = self.wrong[i] if self.rng.random() < 0.6 else wrong(self.rng, self.mode, self.nopt, self.gold[i])
        return answer_text(self.mode, w), False

    def decoded(self, s: str) -> tuple[str, str, int, int]:
        n_line = int(self.rng.integers(2, 9))
        n_full = int(min(ANS_CAP, n_line + self.rng.integers(0, ANS_CAP)))
        return s, s + "\n\nThis follows from the steps above.", n_line, n_full


def write_run(root: Path, task: str, split: str, model: str, seed: int, fine: bool):
    rng = np.random.default_rng([seed, list(TASKS).index(task), list(MODELS).index(model), int(split == "eval")])
    N = N_EVAL if split == "eval" else N_TRAIN
    toy = Toy(rng, task, model, N)
    budgets = sorted({int(b) for b in DEFAULT_BUDGETS.split(",")})
    m = len(budgets)
    active = np.array(budgets)[None, :] < toy.think[:, None]
    txt_line = np.full((N, m + 1), "", dtype=object); txt_full = np.full((N, m + 1), "", dtype=object)
    len_line = np.zeros((N, m + 1), np.int16); len_full = np.zeros((N, m + 1), np.int16)
    lp = np.full((N, m + 1, ANS_CAP), np.nan, np.float32); ent = np.full((N, m + 1, ANS_CAP), np.nan, np.float32)
    mkr = np.zeros((N, m + 1), np.float32)
    for i in range(N):
        for j in range(m + 1):
            if j < m and not active[i, j]:
                continue
            if j < m:
                s, ok = toy.probe(i, budgets[j])
            else:
                s, ok = (answer_text(toy.mode, toy.gold[i]), True) if toy.F[i] else \
                    (answer_text(toy.mode, toy.wrong[i]), False)
            txt_line[i, j], txt_full[i, j], len_line[i, j], len_full[i, j] = toy.decoded(s)
            b = len_full[i, j]
            lp[i, j, :b] = -np.abs(rng.normal(0.1 if ok else 0.8, 0.1 if ok else 0.5, b))
            ent[i, j, :b] = np.abs(rng.normal(0.2 if ok else 1.0, 0.1 if ok else 0.4, b))
            mkr[i, j] = abs(rng.normal(1.0, 0.5))
    nat_ok = toy.F ^ (rng.random(N) < 0.03)
    resp = [f"Putting it together.\n\n{PROBE_TAIL}{answer_text(toy.mode, toy.gold[i] if nat_ok[i] else toy.wrong[i])}"
            for i in range(N)]
    resp_len = rng.integers(50, 400, N)
    gen_len = np.where(toy.closed, toy.think + 1 + resp_len, toy.max_think + 1024)
    stem = f"{task}_{split}"
    run = root / f"seed{seed}" / f"{stem}__{model}"
    run.mkdir(parents=True, exist_ok=True)
    qid = np.array([f"{task}-{split}-{i:05d}" for i in range(N)])
    np.savez_compressed(
        run / "raw.npz", budgets=np.array(budgets, dtype=int), think_len=toy.think, closed=toy.closed,
        prompt_len=rng.integers(80, 400, N), gen_len=gen_len, active=active, txt_line=txt_line.astype(str),
        txt_full=txt_full.astype(str), len_line=len_line, len_full=len_full, lp=lp, ent=ent, mkr=mkr,
        resp_text=np.array(resp), gold=toy.gold, mode=np.array([toy.mode] * N), n_options=np.full(N, toy.nopt),
        category=np.array([rng.choice(CATEGORIES) if task == "mmlu_pro" else "" for _ in range(N)]), qid=qid)
    meta = {"data": f"data/{stem}.jsonl", "stem": stem, "task": task, "split": split, "model": model,
            "model_tag": model, "family": MODELS[model]["family"], "backend": "fake", "vllm_version": None, "N": N,
            "budgets": budgets, "max_think": toy.max_think, "ans_cap": ANS_CAP, "logprobs_topk": 20,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20 if MODELS[model]["family"] == "qwen3" else 0,
                         "seed": seed},
            "probe_tail": PROBE_TAIL, "suffix_ids": [], "stop_ids": [], "end_only": False, "tokenizer_class": "fake",
            "timings": {}}
    json.dump(meta, open(run / "meta.json", "w"), indent=2)
    if not fine:
        return
    fb = fine_budgets(128, 1024, 256, toy.max_think, budgets)
    fa = np.array(fb)[None, :] < toy.think[:, None]
    f_line = np.full((N, len(fb)), "", dtype=object); f_full = np.full((N, len(fb)), "", dtype=object)
    f_ll = np.zeros((N, len(fb)), np.int16); f_lf = np.zeros((N, len(fb)), np.int16)
    for i in range(N):
        for j, B in enumerate(fb):
            if not fa[i, j]:
                continue
            if B in budgets:
                jo = budgets.index(B)
                f_line[i, j], f_full[i, j], f_ll[i, j], f_lf[i, j] = \
                    txt_line[i, jo], txt_full[i, jo], len_line[i, jo], len_full[i, jo]
                continue
            f_line[i, j], f_full[i, j], f_ll[i, j], f_lf[i, j] = toy.decoded(toy.probe(i, B)[0])
    np.savez_compressed(run / "fine_probes.npz", budgets=np.array(fb, dtype=int), active=fa,
                        txt_line=f_line.astype(str), txt_full=f_full.astype(str), len_line=f_ll, len_full=f_lf, qid=qid)


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python make_fake_probes.py <out_dir>")
    root = Path(sys.argv[1])
    for task, model in SETTINGS:
        for split in ("eval", "train"):
            write_run(root, task, split, model, 0, fine=split == "eval" and "AdaptThink" not in model)
    for seed in (1, 2):
        for task, model in SEED_SETTINGS:
            for split in ("eval", "train"):
                write_run(root, task, split, model, seed, fine=False)
    n = len(list(root.rglob("raw.npz")))
    print(f"wrote {n} synthetic probe records -> {root}")


if __name__ == "__main__":
    main()
