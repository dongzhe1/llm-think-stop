from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class GenOut:
    token_ids: list[int]
    lp: list[float] = field(default_factory=list)
    ent: list[float] = field(default_factory=list)


def _topk_entropy(logprobs: list[float]) -> float:
    return float(-sum(math.exp(x) * x for x in logprobs if x > -1e9))


class VLLMBackend:
    def __init__(self, model: str, tp: int = 1, gpu_mem: float = 0.92, max_model_len: int | None = None,
                 seed: int = 0, quantization: str | None = None, enforce_eager: bool = False,
                 max_logprobs: int = 20):
        import os
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM
        kw = dict(model=model, tensor_parallel_size=tp, dtype="bfloat16", gpu_memory_utilization=gpu_mem,
                  enable_prefix_caching=True, seed=seed, trust_remote_code=True, max_logprobs=max_logprobs)
        if max_model_len:
            kw["max_model_len"] = int(max_model_len)
        if quantization:
            kw["quantization"] = quantization
        if enforce_eager:
            kw["enforce_eager"] = True
        if tp > 1:
            kw["disable_custom_all_reduce"] = True
        log.info(f"[vllm] LLM({kw})")
        self.llm = LLM(**kw)
        self.name = "vllm"

    def generate(self, prompts: list[list[int]], *, max_tokens: int, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seeds: list[int] | None = None, logprobs: int = 0,
                 stop_token_ids: list[int] | None = None, tag: str = "") -> list[GenOut]:
        from vllm import SamplingParams
        sps = []
        for i in range(len(prompts)):
            kw = dict(max_tokens=int(max_tokens), temperature=float(temperature), top_p=float(top_p))
            if top_k and top_k > 0:
                kw["top_k"] = int(top_k)
            if seeds is not None:
                kw["seed"] = int(seeds[i])
            if logprobs:
                kw["logprobs"] = int(logprobs)
            if stop_token_ids:
                kw["stop_token_ids"] = list(stop_token_ids)
            sps.append(SamplingParams(**kw))
        t0 = time.time()
        res = self.llm.generate([{"prompt_token_ids": list(p)} for p in prompts], sps, use_tqdm=True)
        n_tok = 0
        outs = []
        for r in res:
            o = r.outputs[0]
            ids = list(o.token_ids)
            n_tok += len(ids)
            g = GenOut(token_ids=ids)
            if logprobs and o.logprobs is not None:
                for t, d in zip(ids, o.logprobs):
                    items = sorted((v.logprob for v in d.values()), reverse=True)[:logprobs]
                    chosen = d[t].logprob if t in d else items[-1]
                    g.lp.append(float(chosen))
                    g.ent.append(_topk_entropy(items))
            outs.append(g)
        dt = time.time() - t0
        log.info(f"[vllm]{'['+tag+']' if tag else ''} {len(prompts)} prompts, {n_tok} new tokens "
                 f"in {dt:.0f}s ({n_tok / max(dt, 1e-6):.0f} tok/s)")
        return outs


class HFBackend:

    def __init__(self, model: str, seed: int = 0, batch_tokens: int = 262144, max_batch: int = 64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        attn = "sdpa"
        try:
            import flash_attn
            attn = "flash_attention_2"
        except ImportError:
            pass
        self.model = AutoModelForCausalLM.from_pretrained(
            model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation=attn).eval()
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.seed, self.batch_tokens, self.max_batch = seed, batch_tokens, max_batch
        self.name = "hf"

    def generate(self, prompts, *, max_tokens, temperature=0.0, top_p=1.0, top_k=0, seeds=None,
                 logprobs=0, stop_token_ids=None, tag=""):
        torch = self.torch
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outs: list[GenOut | None] = [None] * len(prompts)
        stops = self.eos | set(stop_token_ids or [])
        i, t0 = 0, time.time()
        while i < len(order):
            L = len(prompts[order[i]]) + max_tokens
            bs = max(1, min(self.max_batch, self.batch_tokens // max(L, 1)))
            idx = order[i:i + bs]
            L = max(len(prompts[k]) for k in idx)
            ids = torch.full((len(idx), L), self.pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, k in enumerate(idx):
                p = prompts[k]
                ids[r, L - len(p):] = torch.tensor(p)
                att[r, L - len(p):] = 1
            ids, att = ids.to(self.model.device), att.to(self.model.device)
            kw = dict(max_new_tokens=max_tokens, pad_token_id=self.pad, return_dict_in_generate=True,
                      eos_token_id=list(stops))
            if temperature > 0:
                torch.manual_seed(int(seeds[idx[0]]) if seeds else self.seed)
                kw.update(do_sample=True, temperature=temperature, top_p=top_p)
                if top_k and top_k > 0:
                    kw["top_k"] = top_k
            else:
                kw["do_sample"] = False
            if logprobs:
                kw["output_logits"] = True
            with torch.no_grad():
                out = self.model.generate(input_ids=ids, attention_mask=att, **kw)
            seq = out.sequences[:, L:]
            for r, k in enumerate(idx):
                toks = seq[r].tolist()
                n = len(toks)
                for t_i, t in enumerate(toks):
                    if t in stops:
                        n = t_i + 1
                        break
                g = GenOut(token_ids=toks[:n])
                if logprobs:
                    for t_i in range(n):
                        lg = torch.log_softmax(out.logits[t_i][r].float(), dim=-1)
                        top = torch.topk(lg, logprobs).values.tolist()
                        g.lp.append(float(lg[toks[t_i]]))
                        g.ent.append(_topk_entropy(top))
                outs[k] = g
            del out, seq, ids, att
            torch.cuda.empty_cache()
            i += bs
            log.info(f"[hf]{'['+tag+']' if tag else ''} {i}/{len(prompts)} ({time.time() - t0:.0f}s)")
        return outs


def make_backend(kind: str, model: str, **kw):
    if kind == "vllm":
        return VLLMBackend(model, **{k: v for k, v in kw.items()
                                     if k in ("tp", "gpu_mem", "max_model_len", "seed", "quantization",
                                              "enforce_eager", "max_logprobs")})
    if kind == "hf":
        return HFBackend(model, **{k: v for k, v in kw.items() if k in ("seed", "batch_tokens", "max_batch")})
    raise ValueError(kind)
