# llm-think-stop

**When does learning to stop help? A diagnostic framework for reasoning-LLM early-exit.**

`llm-think-stop` is a probe toolkit that instruments the reasoning-LLM inference pipeline to answer one question: *can a lightweight learned classifier, operating only on output-level signals (confidence, entropy, stability, backtracking density), decide per-question when to stop thinking — and does it beat strong scalar baselines (confidence exit, entropy exit, self-consistency)?*

It supports three experiment regimes — **budget forcing** (prove the opportunity: models overthink), **learned stop** (train a logistic classifier on prefix-observable features), and **cross-task transfer** (train on one task, calibrate on another with conformal risk control). Results are saved as self-contained NumPy archives so all downstream analysis (bootstrap CIs, feature ablation, paired significance tests, conformal risk summaries) runs offline on CPU.

## Contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [What this library does](#what-this-library-does)
- [Script reference](#script-reference)
  - [Data preparation](#data-preparation)
  - [Core probes — reasoning early-exit](#core-probes--reasoning-early-exit)
  - [Supplementary probes](#supplementary-probes)
  - [Analysis scripts](#analysis-scripts)
  - [Utility scripts](#utility-scripts)
- [Output files (per run)](#output-files-per-run)
- [Project structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## Quick start

```bash
# 1. Install dependencies
pip install torch transformers scikit-learn numpy scipy

# 2. Prepare reasoning data (CPU, internet required — downloads from HuggingFace Hub)
python scripts/prepare_reason.py --task gsm8k --n 300 --out data/gsm8k.jsonl

# 3. Run the budget-forcing probe — proves models overthink (GPU, ~20 min for n=200)
python scripts/reason_budget_probe.py \
  --data data/gsm8k.jsonl \
  --model Qwen/Qwen3-8B \
  --n 200 --max_think 2048 \
  --budgets 0,128,256,512,1024

# 4. Run the learned-stop probe — trains a per-question logistic stopper (GPU, ~40 min)
python scripts/reason_learnstop_probe.py \
  --data data/gsm8k.jsonl \
  --model Qwen/Qwen3-8B \
  --n 200 --max_think 3072 \
  --budgets 0,128,192,256,384,512,640,768,1024,1536

# 5. Analyze results offline (CPU)
python scripts/analyze_results.py <run_directory>
```

## Installation

### CPU-only (analysis scripts)

```bash
pip install numpy scipy scikit-learn
```

### GPU (probe scripts — model inference)

```bash
# Core stack
pip install torch>=2.1 transformers accelerate

# Data preparation
pip install datasets

# Optional: FlashAttention for O(N) memory scaling
pip install flash-attn

# Optional: 4-bit quantization for large models
pip install bitsandbytes
```

> **Note:** All models are downloaded automatically from HuggingFace Hub on first use. The scripts default to public model IDs like `Qwen/Qwen3-8B`, `Qwen/Qwen3-32B`. For offline clusters, point `--model` to a local path.

## What this library does

Reasoning LLMs (Qwen3, DeepSeek-R1, etc.) generate long chain-of-thought traces before answering. They often **overthink** — continuing to reason long after the answer is already determined. This library instruments the full inference pipeline to answer:

> Can a per-question learned stop classifier, using only prefix-observable features (confidence, entropy, answer stability, backtracking markers), recover the accuracy-vs-cost Pareto frontier — and does it outperform strong scalar baselines (confidence exit, entropy exit, self-consistency)?

The three experiment regimes:

| Regime | Script | What it measures |
|--------|--------|------------------|
| **Budget forcing** | `reason_budget_probe.py` | Full reasoning trace once, then force-answer at every budget checkpoint. Proves the *opportunity*: a fixed budget far below the model's natural length already matches full accuracy. |
| **Learned stop** | `reason_learnstop_probe.py` | Trains a logistic classifier on per-checkpoint features with grouped 5-fold CV. Tests whether *adaptive* per-question stopping beats the best *fixed* budget. |
| **Cross-task transfer** | `reason_transfer_probe.py` | Trains the classifier on a source task, calibrates the stop threshold on a small target calibration set via split-conformal. Tests whether the learned stop signal *generalizes*. |

The key baselines implemented (all post-hoc from the same saved traces):

| Baseline | Signal | Reference |
|----------|--------|-----------|
| `confidence_exit` | Answer-token mean log-probability | DEER-style |
| `entropy_exit` | Answer-token mean entropy | EAT-style |
| `self_consistency` | Run-length of stable answer | Certaindex-style |
| `confidence_leap` | First large probability jump | Confidence Leaps |
| `deer_transition` | Transition from thinking to answer confidence spike | DEER |
| `eat_entropy_stability` | Entropy stabilisation after `</Think>` | EAT |
| `puma_convergence` | Semantic convergence proxy (stability × confidence) | PUMA-style |
| `terminator_light` | Earliest correct answer from supervision | TERMINATOR-style |
| `concise` | "Think briefly" prompt (no learned component) | System-prompt baseline |

## Script reference

### Data preparation

These scripts download public datasets from HuggingFace Hub and write JSONL files. Run on a CPU machine with internet access.

---

#### `prepare_reason.py`

Builds reasoning-task JSONL for the budget / learnstop / transfer probes. Downloads from HuggingFace Hub.

```bash
python scripts/prepare_reason.py --task gsm8k    --n 300 --out data/gsm8k.jsonl
python scripts/prepare_reason.py --task math500  --n 500 --out data/math500.jsonl
python scripts/prepare_reason.py --task mmlu_pro --n 800 --out data/mmlu_pro.jsonl
python scripts/prepare_reason.py --task aime     --n 90  --out data/aime.jsonl
python scripts/prepare_reason.py --task gpqa     --n 200 --out data/gpqa.jsonl
```

| Flag | Description |
|------|-------------|
| `--task` | Dataset: `gsm8k`, `math500`, `mmlu_pro`, `aime`, `gpqa` |
| `--n` | Number of questions to sample |
| `--out` | Output JSONL path |
| `--seed` | Random seed (default 42) |

Output format (one JSON object per line):
```json
{"task": "gsm8k", "question": "...", "gold": "42"}
```

---

#### `prepare_pairs.py`

Builds pairwise comparison JSONL for the judge bias probe. Downloads RewardBench / MT-Bench / LLMBar from HuggingFace Hub.

```bash
python scripts/prepare_pairs.py --source rewardbench --out data/rewardbench.jsonl
python scripts/prepare_pairs.py --source mtbench     --out data/mtbench.jsonl
python scripts/prepare_pairs.py --source llmbar --config "Adversarial/Neighbor" --out data/llmbar_adv.jsonl
```

| Flag | Description |
|------|-------------|
| `--source` | Dataset: `rewardbench`, `mtbench`, `llmbar` |
| `--config` | LLMBar subset (default `Natural`) |
| `--out` | Output JSONL path |
| `--limit` | Max number of pairs (0 = all) |

---

#### `prepare_qa.py`

Builds multiple-choice / QA JSONL for the selective prediction and routing probes.

```bash
python scripts/prepare_qa.py --task mmlu     --n 600 --out data/mmlu.jsonl
python scripts/prepare_qa.py --task triviaqa --n 600 --out data/triviaqa.jsonl
```

| Flag | Description |
|------|-------------|
| `--task` | `mmlu` (multiple-choice) or `triviaqa` (open-ended QA) |
| `--n` | Number of questions to sample |
| `--out` | Output JSONL path |

---

#### `prepare_rerank.py`

Builds candidate pools from BEIR datasets for the reranker calibration probe.

```bash
python scripts/prepare_rerank.py --dataset scifact --n_queries 100 --n_neg 20 --out data/rerank_scifact.jsonl
```

| Flag | Description |
|------|-------------|
| `--dataset` | BEIR dataset name (`scifact`, `fiqa`, `nfcorpus`, `trec-covid`, ...) |
| `--n_queries` | Number of queries to sample |
| `--n_neg` | Number of negative passages per query |
| `--out` | Output JSONL path |

---

### Core probes — reasoning early-exit

These are the main experiments. Each requires a GPU and produces a self-contained result directory under `results/`.

---

#### `reason_budget_probe.py`

**Opportunity probe.** Generates the full reasoning trace once per question, then force-answers (`</think>` + greedy decode) at every budget checkpoint. Measures accuracy-vs-think-tokens to answer: *(a) Does thinking help? (b) Is the model overthinking? (c) Is there a realizable per-question stop signal?*

```bash
python scripts/reason_budget_probe.py \
  --data data/gsm8k.jsonl \
  --model Qwen/Qwen3-8B \
  --n 200 --max_think 2048 \
  --budgets 0,64,128,256,512,1024
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *required* | Input JSONL (from `prepare_reason.py`) |
| `--model` | *required* | HuggingFace model ID or local path |
| `--n` | `200` | Number of questions |
| `--max_think` | `2048` | Max new tokens for the thinking trace |
| `--budgets` | `0,64,128,256,512,1024` | Comma-separated budgets (think tokens before force-answer) |
| `--ans_tokens` | `48` | Max answer tokens per forced decode |
| `--batch` | `32` | Batch size for answer decodes |

Output: prints a budget-vs-accuracy table and a GO / NO-GO verdict. No files saved (lightweight probe).

---

#### `reason_learnstop_probe.py`

**Decisive probe.** Runs the model once per question, forces an answer at every budget checkpoint, extracts causal per-checkpoint features (confidence, entropy, stability, vote share, backtracking density, budget position), trains a logistic stop-classifier with grouped 5-fold CV, and simulates the stop policy across a sweep of thresholds. Compares the learned adaptive frontier against scalar baselines and the fixed-budget frontier.

```bash
python scripts/reason_learnstop_probe.py \
  --data data/gsm8k.jsonl \
  --model Qwen/Qwen3-8B \
  --n 200 --max_think 3072 \
  --budgets 0,128,192,256,384,512,640,768,1024,1536 \
  --out_dir results/learnstop
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *required* | Input JSONL |
| `--model` | *required* | HuggingFace model ID or local path |
| `--n` | `200` | Number of questions |
| `--max_think` | `3072` | Max think tokens for the full reasoning trace |
| `--budgets` | `0,128,192,256,384,512,640,768,1024,1536` | Comma-separated budget checkpoints |
| `--ans_tokens` | `48` | Max answer tokens per forced decode |
| `--folds` | `5` | Number of CV folds (grouped by question) |
| `--out_dir` | `results/learnstop` | Root output directory |
| `--concise` | *off* | Also run a "think briefly" prompt baseline |
| `--probe_template` | `terse` | Probe answer prompt: `terse`, `no_reasoning`, `the_answer_is` |
| `--temperature` | `0.0` | Decoding temperature (0 = greedy) |

Features used by the logistic classifier (all causal — available at the checkpoint, no future information):

- Budget index and fraction
- Answer-token mean log-probability
- Answer-token mean entropy
- Answer stability (same as previous checkpoint?)
- Run-length of stable answer
- Vote share of current answer
- Backtracking marker density (`wait`, `hmm`, `but`, `let me`, `actually`, ...)
- Think tokens consumed so far

Output: `<out_dir>/<dataset>_<model>_n<N>_<timestamp>/` (see [Output files](#output-files-per-run)).

---

#### `reason_transfer_probe.py`

**Cross-task transfer.** Trains the stop classifier on a source task, then applies it to a target task with split-conformal threshold calibration. Produces both in-distribution (source, grouped 5-fold CV) and out-of-distribution (target, split-conformal) results. Measures the generalization gap and whether the learned signal transfers.

```bash
python scripts/reason_transfer_probe.py \
  --source_data data/gsm8k.jsonl \
  --target_data data/math500.jsonl \
  --model Qwen/Qwen3-8B \
  --n 300 --max_think 3072 \
  --budgets 0,128,192,256,384,512,640,768,1024,1536 \
  --out_dir results/transfer
```

| Flag | Default | Description |
|------|---------|-------------|
| `--source_data` | *required* | JSONL for training the classifier |
| `--target_data` | *required* | JSONL for calibration + evaluation |
| `--model` | *required* | HuggingFace model ID or local path |
| `--n` | `300` | Questions per dataset (≥200 recommended; target is split cal/test) |
| `--max_think` | `3072` | Max think tokens |
| `--budgets` | `0,128,192,256,384,512,640,768,1024,1536` | Comma-separated budgets |
| `--cal_frac` | `0.5` | Fraction of target used for conformal calibration |
| `--folds` | `5` | CV folds for in-distribution evaluation |
| `--out_dir` | `results/transfer` | Root output directory |

Output: `<out_dir>/<source>-to-<target>_<model>_n<N>_<timestamp>/` with both `source_raw.npz` and `target_raw.npz`.

---

### Supplementary probes

These are lighter-weight exploratory probes from earlier proposal directions (GO / NO-GO decision tools).

---

#### `judge_bias_probe.py`

Quantifies position bias, length bias, and self-preference in an LLM-as-judge. Scores each pair in both orders using log-probability margin (A vs B token) and reports order-swap flip rate and first-position win-rate.

```bash
python scripts/judge_bias_probe.py \
  --pairs data/rewardbench.jsonl \
  --judge Qwen/Qwen3-32B \
  --n 400 --judge_family qwen
```

| Flag | Description |
|------|-------------|
| `--pairs` | Input JSONL (from `prepare_pairs.py`) |
| `--judge` | HuggingFace model ID or local path |
| `--n` | Number of pairs to score |
| `--judge_family` | Substring identifying the judge's model family (for self-preference) |
| `--load_in_4bit` | Use 4-bit quantization |

---

#### `selpred_probe.py`

Selective prediction probe. Measures AUROC of individual confidence signals (margin, max-prob, neg-entropy, self-consistency) and a logistic fusion for predicting model correctness.

```bash
python scripts/selpred_probe.py \
  --data data/mmlu.jsonl \
  --model Qwen/Qwen3-8B \
  --n 500 --k 5
```

| Flag | Description |
|------|-------------|
| `--data` | Input JSONL (from `prepare_qa.py`) |
| `--model` | HuggingFace model ID or local path |
| `--n` | Number of questions |
| `--k` | Number of self-consistency samples (0 = skip) |

---

#### `route_probe.py`

Confidence routing probe. Loads a small and large model sequentially on one GPU. Measures whether the small model's confidence can gate escalation to the large model, producing a calibrated cascade accuracy-vs-cost frontier.

```bash
python scripts/route_probe.py \
  --data data/mmlu.jsonl \
  --small Qwen/Qwen3-8B \
  --large Qwen/Qwen3-32B \
  --n 500 --k 5
```

| Flag | Description |
|------|-------------|
| `--data` | Input JSONL (from `prepare_qa.py`) |
| `--small` | Small model path |
| `--large` | Large model path |
| `--n` | Number of questions |
| `--k` | Self-consistency samples |

---

#### `reranker_probe.py`

Reranker calibration probe. Scores BEIR candidate pools with multiple rerankers (causal LLM logprob scoring or cross-encoder), reporting per-reranker ECE, nDCG@10, MRR, and cross-reranker Kendall-tau disagreement.

```bash
python scripts/reranker_probe.py \
  --pairs data/rerank_scifact.jsonl \
  --rerankers "qwen=Qwen/Qwen3-Reranker-8B=causal,gemma=BAAI/bge-reranker-v2-gemma=causal"
```

| Flag | Description |
|------|-------------|
| `--pairs` | Input JSONL (from `prepare_rerank.py`) |
| `--rerankers` | Comma-separated `name=path=type` triples (`type`: `causal` or `crossencoder`) |

---

### Analysis scripts

These read saved run directories — no GPU, no re-computation. All compute baselines, bootstrap CIs, and paired significance tests from the saved NumPy archives.

---

#### `analyze_results.py`

Main analysis entry point. Reads a run directory (`raw.npz`), computes all baselines (confidence exit, entropy exit, self-consistency, confidence leap, DEER, EAT, PUMA, TERMINATOR-light), bootstrap 95% CIs on adapt gain for each method, paired bootstrap (learned vs each baseline), feature ablation (drop-one-feature retraining), conformal risk control grid, and cost-model comparison.

```bash
python scripts/analyze_results.py <run_directory>

# For transfer runs, select which NPZ:
python scripts/analyze_results.py <run_directory> --npz source_raw.npz
python scripts/analyze_results.py <run_directory> --npz target_raw.npz
```

| Flag | Default | Description |
|------|---------|-------------|
| `run_dir` | *required* | Path to a run directory |
| `--npz` | `raw.npz` | Which NPZ file (`raw.npz`, `source_raw.npz`, `target_raw.npz`) |
| `--boot` | `1000` | Number of bootstrap resamples |
| `--no_ablation` | *off* | Skip slower feature-ablation retraining |

Output: prints tables to stdout and writes `analysis/` subdirectory with `baselines.csv`, `bootstrap_ci.csv`, `conformal_caltest.csv`, `analysis.json`.

---

#### `analyze_extended.py`

Extended analyses beyond the core baselines. Includes model-class comparison (logistic vs gradient boosting vs MLP), feature importance (permutation and coefficient-based), comprehensive feature ablation across multiple feature subsets, and additional conformal baselines (Learn-then-Test, Conformal Thinking).

```bash
python scripts/analyze_extended.py <run_directory>
python scripts/analyze_extended.py <run_directory> --skip_slow
```

| Flag | Description |
|------|-------------|
| `run_dir` | Path to a run directory |
| `--npz` | Which NPZ file (default `raw.npz`) |
| `--skip_slow` | Skip model_class, feature_importance, and extended_ablation |

Output: writes `extended_analysis.json` to the run's `analysis/` directory.

---

#### `paper_stats.py`

Aggregates results across all completed runs to produce LaTeX-ready paper tables. Scans `results/learnstop/` and `results/transfer/` for runs with completed `analysis/analysis.json`, then compiles Table 1 (in-distribution main results), Table 2 (transfer results), ablation, conformal, concise, and paired bootstrap tables.

```bash
python scripts/paper_stats.py
```

No arguments. Output: writes `results/paper_stats/paper_tables.txt` and individual `table*.csv` files.

---

#### `transfer_matrix_offline.py`

Post-hoc cross-task and cross-model transfer evaluation. Reads two existing run directories (with pre-computed features), trains the stop classifier on the source, and evaluates on the target with conformal calibration — no GPU inference needed.

```bash
# Single pair
python scripts/transfer_matrix_offline.py \
  --source <source_run_dir> \
  --target <target_run_dir> \
  --label "GSM8K→MATH500 32B"

# Full matrix (all configured task/model pairs)
python scripts/transfer_matrix_offline.py --all --out transfer_matrix.json
```

| Flag | Description |
|------|-------------|
| `--source` | Source run directory (with `raw.npz`) |
| `--target` | Target run directory (with `raw.npz`) |
| `--label` | Human-readable label for this pair |
| `--all` | Run the full transfer matrix (all configured pairs) |
| `--out` | Output JSON path |

---

### Utility scripts

---

#### `result_io.py`

Shared result-persistence helpers imported by the probe scripts. Writes the self-contained run directory layout (`meta.json`, `raw.npz`, `conformal.csv`, `frontier_adaptive.csv`, `frontier_fixed.csv`, `summary.json`). Not invoked directly.

---

#### `serving_profile.py`

Measures real serving cost (latency, throughput, GPU memory) for a given model under a given backend, generating the serving profile used in the paper's cost analysis.

```bash
python scripts/serving_profile.py \
  --model Qwen/Qwen3-32B \
  --data data/gsm8k.jsonl \
  --n 50 --backend hf \
  --out serving_profile.json
```

| Flag | Description |
|------|-------------|
| `--model` | HuggingFace model ID or local path |
| `--data` | Input JSONL (questions used for profiling) |
| `--n` | Number of questions |
| `--backend` | Inference backend: `hf` (HuggingFace Transformers) or `vllm` |
| `--out` | Output JSON path |

---

#### `test_nan_fix.py`

CPU-only regression test verifying the NaN fix for entropy computation under masked/special tokens. Covers three scenarios: entropy NaN from masked tokens, pre-sklearn X guard, and post-StandardScaler guard.

```bash
python scripts/test_nan_fix.py
```

No arguments. Exits 0 on success, raises on failure.

---

## Output files (per run)

Each probe run produces in `results/<probe>/<tag>/`:

| File | Content |
|------|---------|
| `raw.npz` | NumPy archive: `budgets`, `correct`, `think_lens`, `ans`, `conf_lp`, `conf_ent`, `p_stop`, `X`, `y` — the full per-question × per-checkpoint matrices |
| `conformal.csv` | Conformal risk control grid: α, threshold τ for each method, test risk, test accuracy, token saving % |
| `frontier_adaptive.csv` | Full adaptive frontier sweep: threshold, accuracy, mean tokens, saving vs full |
| `frontier_fixed.csv` | Fixed-budget frontier: budget, accuracy, saving vs full |
| `summary.json` | Headline numbers: full accuracy, mean think tokens, learned peak gain, operating threshold |
| `analysis/` | Post-hoc CPU analysis output (from `analyze_results.py` / `analyze_extended.py`) |
| `analysis/baselines.csv` | One row per method: peak gain, operating threshold |
| `analysis/bootstrap_ci.csv` | Bootstrap 95% CI on adapt gain for each method |
| `analysis/conformal_caltest.csv` | Conformal calibration-test split results |
| `analysis/analysis.json` | Structured analysis summary (baselines + bootstrap + ablation + conformal) |
| `analysis/extended_analysis.json` | Extended baselines (DEER, EAT, PUMA, TERMINATOR-light, cost models, Learn-then-Test) |

For transfer runs, `source_raw.npz` and `target_raw.npz` replace `raw.npz`.

## Project structure

```
llm-think-stop/
├── scripts/                           # Executable experiment scripts
│   ├── prepare_reason.py              #   Build reasoning JSONL (gsm8k/math500/mmlu_pro/aime/gpqa)
│   ├── prepare_pairs.py               #   Build pairwise comparison JSONL (rewardbench/mtbench/llmbar)
│   ├── prepare_qa.py                  #   Build QA JSONL (mmlu/triviaqa)
│   ├── prepare_rerank.py              #   Build BEIR candidate pools
│   ├── reason_budget_probe.py         #   Budget-forcing probe (opportunity)
│   ├── reason_learnstop_probe.py      #   Learned-stop probe (main method)
│   ├── reason_transfer_probe.py       #   Cross-task transfer probe
│   ├── judge_bias_probe.py            #   LLM judge bias quantification
│   ├── selpred_probe.py               #   Selective prediction probe
│   ├── route_probe.py                 #   Confidence routing probe
│   ├── reranker_probe.py              #   Reranker calibration probe
│   ├── analyze_results.py             #   Main analysis (baselines + bootstrap + ablation + conformal)
│   ├── analyze_extended.py            #   Extended analysis (DEER/EAT/PUMA/TERMINATOR + cost models)
│   ├── paper_stats.py                 #   Paper table generation
│   ├── transfer_matrix_offline.py     #   Post-hoc transfer matrix (CPU-only)
│   ├── serving_profile.py             #   Serving cost profiling
│   ├── result_io.py                   #   Shared result persistence (imported, not invoked directly)
│   └── test_nan_fix.py               #   NaN fix regression test
└── README.md
```

## Citation

If you use this code in your research, please cite:

```
@software{llm-think-stop,
  title = {llm-think-stop: When Does Learning to Stop Help? A Diagnostic Framework for Reasoning-LLM Early-Exit},
  url = {https://github.com/dongzhe1/llm-think-stop},
  year = {2026},
}
```

## License

MIT
