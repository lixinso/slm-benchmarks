# Llama 3.1 8B (Ollama) Benchmark

This folder contains a minimal, repeatable benchmark harness for `llama3.1:8b` using Ollama's local HTTP API.

## MMLU Benchmark Results (TODO Area)

This table summarizes the MMLU benchmark results for different configurations.

| Task | `limit=10` | `limit=20` | Official |
| :--- | :---: | :---: | :---: |
| **Overall (mmlu)** | **0.6930** | **0.6842** | **0.667** |
| **humanities** | 0.7615 | 0.7231 | N/A |
| - formal_logic | 0.5000 | 0.6000 | N/A |
| - high_school_european_history | 0.8000 | 0.6000 | N/A |
| - high_school_us_history | 0.9000 | 0.8500 | N/A |
| - high_school_world_history | 0.9000 | 0.8000 | N/A |
| - international_law | 1.0000 | 0.8500 | N/A |
| - jurisprudence | 0.7000 | 0.7500 | N/A |
| - logical_fallacies | 0.9000 | 0.7500 | N/A |
| - moral_disputes | 0.5000 | 0.6000 | N/A |
| - moral_scenarios | 0.5000 | 0.6500 | N/A |
| - philosophy | 0.9000 | 0.7500 | N/A |
| - prehistory | 0.8000 | 0.8500 | N/A |
| - professional_law | 0.6000 | 0.5000 | N/A |
| - world_religions | 0.9000 | 0.8500 | N/A |
| **other** | 0.7077 | 0.7231 | N/A |
| - business_ethics | 0.7000 | 0.8000 | N/A |
| - clinical_knowledge | 0.7000 | 0.8000 | N/A |
| - college_medicine | 0.9000 | 0.8000 | N/A |
| - global_facts | 0.6000 | 0.5500 | N/A |
| - human_aging | 0.6000 | 0.6000 | N/A |
| - management | 0.8000 | 0.7500 | N/A |
| - marketing | 0.6000 | 0.7000 | N/A |
| - medical_genetics | 0.8000 | 0.8000 | N/A |
| - miscellaneous | 0.9000 | 0.8500 | N/A |
| - nutrition | 0.8000 | 0.7500 | N/A |
| - professional_accounting | 0.4000 | 0.4500 | N/A |
| - professional_medicine | 0.9000 | 0.9000 | N/A |
| - virology | 0.5000 | 0.6500 | N/A |
| **social sciences** | 0.7667 | 0.7458 | N/A |
| - econometrics | 0.4000 | 0.4000 | N/A |
| - high_school_geography | 0.8000 | 0.8500 | N/A |
| - high_school_government_and_politics | 0.9000 | 0.9000 | N/A |
| - high_school_macroeconomics | 0.8000 | 0.6000 | N/A |
| - high_school_microeconomics | 0.6000 | 0.7000 | N/A |
| - high_school_psychology | 0.9000 | 0.8500 | N/A |
| - human_sexuality | 0.9000 | 0.8000 | N/A |
| - professional_psychology | 0.9000 | 0.8000 | N/A |
| - public_relations | 0.6000 | 0.5500 | N/A |
| - security_studies | 0.8000 | 0.8500 | N/A |
| - sociology | 0.7000 | 0.7000 | N/A |
| - us_foreign_policy | 0.9000 | 0.9500 | N/A |
| **stem** | 0.5895 | 0.5921 | N/A |
| - abstract_algebra | 0.6000 | 0.4500 | N/A |
| - anatomy | 0.7000 | 0.7500 | N/A |
| - astronomy | 0.8000 | 0.8500 | N/A |
| - college_biology | 0.9000 | 0.9000 | N/A |
| - college_chemistry | 0.6000 | 0.5000 | N/A |
| - college_computer_science | 0.4000 | 0.4500 | N/A |
| - college_mathematics | 0.1000 | 0.3000 | N/A |
| - college_physics | 0.5000 | 0.5500 | N/A |
| - computer_security | 0.7000 | 0.6500 | N/A |
| - conceptual_physics | 0.7000 | 0.8000 | N/A |
| - electrical_engineering | 0.5000 | 0.5000 | N/A |
| - elementary_mathematics | 0.4000 | 0.3000 | N/A |
| - high_school_biology | 0.9000 | 0.9000 | N/A |
| - high_school_chemistry | 0.6000 | 0.6000 | N/A |
| - high_school_computer_science | 0.8000 | 0.8500 | N/A |
| - high_school_mathematics | 0.3000 | 0.3500 | N/A |
| - high_school_physics | 0.4000 | 0.3500 | N/A |
| - high_school_statistics | 0.8000 | 0.6500 | N/A |
| - machine_learning | 0.5000 | 0.5500 | N/A | 

## What “benchmark Llama 3.1 8B” should mean

- Capability (quality): how often it gets tasks right (accuracy / pass@k / etc.)
- Efficiency (runtime): how fast/cheap it runs on your machine (latency, tokens/sec, memory)
- Reliability: variance across runs, long-context degradation, refusal/safety behavior (if relevant)

A good benchmark is just a repeatable recipe + logged metadata, not one number.

## 1) Decide the benchmark target (pick 1–2 primary goals)

- Local assistant/chat: instruction following + latency/TTFT matter most
- Coding: HumanEval/MBPP + pass@k
- Math/reasoning: GSM8K + reasoning MC tasks (ARC, HellaSwag)
- Long-context: 32k/64k/128k retrieval & QA stress tests

If you only pick one: do (A) MMLU + (B) HumanEval + (C) latency/tokens/sec.

## 2) What to measure (use consistent metrics)

Quality (from `POPULAR_SLM_BENCHMARK_METRICS.md`):

- Multiple-choice: accuracy (and optionally calibration error)
- Free-form QA: exact match / F1 (task-dependent)
- Coding: pass@1 (and pass@k if you sample)
- Long-context: accuracy vs context length; retrieval hit-rate

Efficiency:

- TTFT (time-to-first-token)
- Tokens/sec (steady-state generation speed)
- End-to-end latency for fixed prompt+output length
- Peak RAM / VRAM (or unified memory on macOS)
- (Optional) Power draw / energy per token if you care about cost

## 3) Pick a small but representative benchmark suite

From `POPULAR_SLM_BENCHMARK_DATASETS.md`, a practical “starter suite”:

- General knowledge: MMLU (or a smaller subset if time-constrained)
- Reasoning/commonsense: ARC-Challenge + HellaSwag
- Math: GSM8K
- Coding: HumanEval (optionally MBPP)
- Truthfulness (optional): TruthfulQA
- Long-context (important for Llama 3.1): a “needle-in-a-haystack” style retrieval test at 8k/32k/128k, plus a few long-doc QA samples

Keep it small at first; add breadth later.

## 4) Fix the inference setup so results are comparable

Benchmark numbers change a lot depending on:

- Backend: Ollama vs llama.cpp vs Transformers (each has different kernels)
- Quantization: fp16 vs 8-bit vs 4-bit
- Context length & max output tokens
- Sampling params: temperature/top_p/seed

Suggested defaults:

- For accuracy / MC: temperature=0, fixed max_tokens
- For pass@k: sample with temperature>0 and set k (e.g., 5 or 10), log seeds
- Always log: backend version, model tag, quantization, hardware, and params

## 5) A simple experiment matrix (minimal but informative)

Run these slices (don’t do everything at once):

- Quantization: one “fast local” (e.g., 4-bit) + one “higher quality” (8-bit/fp16 if feasible)
- Context: 4k and 32k (and 128k only for long-context tests)
- Workload: short prompt (≈200 tokens) + long prompt (≈4k tokens)
- Output length: 128 tokens and 512 tokens

## 6) How to run on macOS (practical path)

Since your models doc already suggests Ollama (see `POPULAR_SLM_MODELS.md`), a pragmatic workflow is:

- Use Ollama for runtime benchmarks (TTFT/tokens/sec) and basic prompt suites.
- Use a standardized harness (e.g., EleutherAI lm-eval-harness) if you want canonical MMLU/ARC/HellaSwag scoring—otherwise you’ll end up re-implementing evaluation.

## 7) What to log (so your benchmark is “real”)

For every run, record:

- Model identifier (exact tag), backend, quantization
- Machine: CPU, RAM, GPU/Metal, macOS version
- Prompt length, output length, context window used
- Params: temperature, top_p, top_k, repetition penalty, seed
- Metrics: accuracy/pass@k + TTFT + tokens/sec + peak memory
- Aggregate: mean + p50/p95 over (say) 20–50 prompts

---

## Prereqs

- Install Ollama: https://ollama.com
- Pull the model:
  - `ollama pull llama3.1:8b`
- Ensure Ollama is running (default API is `http://localhost:11434`).

## Run

From the repo root:

```bash
python3 llama-3.1-8b/bench_ollama.py \
  --model llama3.1:8b \
  --prompts llama-3.1-8b/prompts.jsonl \
  --runs 5 \
  --temperature 0 \
  --num_predict 256
```

The script prints one JSON object per run (JSONL), plus summary stats at the end.

## Metrics

- `ttft_s`: time-to-first-token (streaming)
- `gen_toks_per_s`: generation throughput computed from Ollama `eval_count / eval_duration`
- `prompt_toks_per_s`: prompt-processing throughput computed from `prompt_eval_count / prompt_eval_duration`
- `wall_s`: end-to-end latency

## Tips for fair comparisons

- Use `--temperature 0` for deterministic runs.
- Fix `--num_predict` and keep the same prompt suite.
- Run when your machine is otherwise idle.
- Record: macOS version, hardware, Ollama version, and model tag.

### MPS Memory Errors on macOS

If you encounter an "MPS backend out of memory" error while running benchmarks on a Mac with Apple Silicon, you can try one of the following:

- **Reduce `batch_size`**: Lowering the `batch_size` in the `lm_eval` command can significantly reduce memory usage.
- **Set `PYTORCH_MPS_HIGH_WATERMARK_RATIO`**: As a last resort, you can set the `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable to `0.0`. This disables the upper limit for memory allocations, which may allow the benchmark to complete. However, be aware that this can cause system instability or failure.

  ```bash
  PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/lm_eval ...
  ```




## A Note on MMLU Benchmark Size
The MMLU dataset used by `lm-evaluation-harness` consists of 15,908 questions across 57 subtasks. The number of questions varies for each subtask. A full benchmark run (without the `--limit` flag) will run on all of these questions.


## References

- [Introducing Llama 3.1: Our most capable models to date](https://ai.meta.com/blog/meta-llama-3-1/)

