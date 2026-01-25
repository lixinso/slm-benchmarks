# slm-benchmarks
A public repository on Benchmarking all kinds of SLM Models

## Documentation

- **[Popular Open-Source SLM Models](SLM_MODELS/POPULAR_SLM_MODELS.md)** - Comprehensive list of open-source Small Language Models that can be downloaded and run locally on consumer hardware.
- **[Popular SLM Benchmark Datasets](BENCHMARK_DATASETS/POPULAR_SLM_BENCHMARK_DATASETS.md)** - Comprehensive guide to benchmark datasets used for evaluating Small Language Models, including general knowledge, reasoning, coding, math, multilingual, and domain-specific benchmarks.
- **[Popular SLM Benchmark Metrics](BENCHMARK_METRICS/POPULAR_SLM_BENCHMARK_METRICS.md)** - Detailed guide to evaluation metrics used for benchmarking SLMs, including accuracy-based metrics, code generation metrics (pass@k), generation quality metrics (ROUGE, BLEU), and efficiency metrics.

## Benchmark Results

### MMLU (5-shot)
*Note: Preliminary results based on limited samples (10 per subtask).*

| Model | Overall | STEM | Humanities | Social Sciences | Other | Details |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Llama 3.1 8B** | **69.30%** | 58.95% | 76.15% | 76.67% | 70.77% | [Link](SLM_MODELS/llama-3.1-8b/mmlu_benchmark_setup_and_results_preliminary.md) |

## About

This repository provides benchmarking resources and information for Small Language Models (SLMs) that can run locally. Our focus is on open-source models that are practical for deployment on consumer hardware without requiring cloud services.
