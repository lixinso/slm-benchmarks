# MMLU Benchmark Run - 2026-01-28 (limit 100)

This document tracks the results of an MMLU benchmark run with a limit of 100 samples per subtask.

## Steps to Reproduce

The benchmark was run with the following command, using the `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` environment variable to prevent MPS memory errors:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/lm_eval --model hf --model_args '{"pretrained": "SLM_MODELS/llama-3.1-8b-hf", "device_map": "auto"}' --tasks mmlu --device mps --batch_size 8 --output_path SLM_MODELS/llama-3.1-8b/mmlu_results_limit_100.json --limit 100
```

*Note: This command requires the prerequisites outlined in `mmlu_benchmark_setup_and_results_preliminary.md` to be met, including a Python virtual environment and the downloaded Hugging Face model.*

## Results

The overall MMLU accuracy for this run was **0.6784**.

### Summary

| Groups            | Value  | Stderr |
|-------------------|---:|---:|
| mmlu              | 0.6784 | 0.0059 |
| - humanities      | 0.7200 | 0.0121 |
| - other           | 0.7123 | 0.0120 |
| - social sciences | 0.7683 | 0.0119 |
| - stem            | 0.5700 | 0.0109 |

### Detailed Results

| Tasks | Version | Filter | n-shot | Metric | | Value | | Stderr |
| :--- | ---: | :--- | ---: | :--- | :--- | ---: | :--- | ---: |
| mmlu | 2 | none | | acc | ↑ | 0.6784 | ± | 0.0059 |
| - humanities | 2 | none | | acc | ↑ | 0.7200 | ± | 0.0121 |
| - formal_logic | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.0503 |
| - high_school_european_history | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0441 |
| - high_school_us_history | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0359 |
| - high_school_world_history | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0416 |
| - international_law | 1 | none | 0 | acc | ↑ | 0.8200 | ± | 0.0386 |
| - jurisprudence | 1 | none | 0 | acc | ↑ | 0.7600 | ± | 0.0429 |
| - logical_fallacies | 1 | none | 0 | acc | ↑ | 0.7900 | ± | 0.0409 |
| - moral_disputes | 1 | none | 0 | acc | ↑ | 0.7100 | ± | 0.0456 |
| - moral_scenarios | 1 | none | 0 | acc | ↑ | 0.5600 | ± | 0.0499 |
| - philosophy | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0435 |
| - prehistory | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0416 |
| - professional_law | 1 | none | 0 | acc | ↑ | 0.5100 | ± | 0.0502 |
| - world_religions | 1 | none | 0 | acc | ↑ | 0.8100 | ± | 0.0394 |
| - other | 2 | none | | acc | ↑ | 0.7123 | ± | 0.0120 |
| - business_ethics | 1 | none | 0 | acc | ↑ | 0.6900 | ± | 0.0465 |
| - clinical_knowledge | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0441 |
| - college_medicine | 1 | none | 0 | acc | ↑ | 0.7100 | ± | 0.0456 |
| - global_facts | 1 | none | 0 | acc | ↑ | 0.4000 | ± | 0.0492 |
| - human_aging | 1 | none | 0 | acc | ↑ | 0.6700 | ± | 0.0473 |
| - management | 1 | none | 0 | acc | ↑ | 0.8200 | ± | 0.0386 |
| - marketing | 1 | none | 0 | acc | ↑ | 0.8800 | ± | 0.0327 |
| - medical_genetics | 1 | none | 0 | acc | ↑ | 0.7700 | ± | 0.0423 |
| - miscellaneous | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0368 |
| - nutrition | 1 | none | 0 | acc | ↑ | 0.7900 | ± | 0.0409 |
| - professional_accounting | 1 | none | 0 | acc | ↑ | 0.5400 | ± | 0.0501 |
| - professional_medicine | 1 | none | 0 | acc | ↑ | 0.8600 | ± | 0.0349 |
| - virology | 1 | none | 0 | acc | ↑ | 0.5500 | ± | 0.0500 |
| - social sciences | 2 | none | | acc | ↑ | 0.7683 | ± | 0.0119 |
| - econometrics | 1 | none | 0 | acc | ↑ | 0.5200 | ± | 0.0502 |
| - high_school_geography | 1 | none | 0 | acc | ↑ | 0.7900 | ± | 0.0409 |
| - high_school_government_and_politics | 1 | none | 0 | acc | ↑ | 0.9100 | ± | 0.0288 |
| - high_school_macroeconomics | 1 | none | 0 | acc | ↑ | 0.7300 | ± | 0.0446 |
| - high_school_microeconomics | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0416 |
| - high_school_psychology | 1 | none | 0 | acc | ↑ | 0.8900 | ± | 0.0314 |
| - human_sexuality | 1 | none | 0 | acc | ↑ | 0.8100 | ± | 0.0394 |
| - professional_psychology | 1 | none | 0 | acc | ↑ | 0.6700 | ± | 0.0473 |
| - public_relations | 1 | none | 0 | acc | ↑ | 0.6900 | ± | 0.0465 |
| - security_studies | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0441 |
| - sociology | 1 | none | 0 | acc | ↑ | 0.8300 | ± | 0.0378 |
| - us_foreign_policy | 1 | none | 0 | acc | ↑ | 0.8600 | ± | 0.0349 |
| - stem | 2 | none | | acc | ↑ | 0.5700 | ± | 0.0109 |
| - abstract_algebra | 1 | none | 0 | acc | ↑ | 0.3500 | ± | 0.0479 |
| - anatomy | 1 | none | 0 | acc | ↑ | 0.6500 | ± | 0.0479 |
| - astronomy | 1 | none | 0 | acc | ↑ | 0.7300 | ± | 0.0446 |
| - college_biology | 1 | none | 0 | acc | ↑ | 0.8300 | ± | 0.0378 |
| - college_chemistry | 1 | none | 0 | acc | ↑ | 0.4700 | ± | 0.0502 |
| - college_computer_science | 1 | none | 0 | acc | ↑ | 0.5700 | ± | 0.0498 |
| - college_mathematics | 1 | none | 0 | acc | ↑ | 0.3600 | ± | 0.0482 |
| - college_physics | 1 | none | 0 | acc | ↑ | 0.4300 | ± | 0.0498 |
| - computer_security | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0435 |
| - conceptual_physics | 1 | none | 0 | acc | ↑ | 0.5700 | ± | 0.0498 |
| - electrical_engineering | 1 | none | 0 | acc | ↑ | 0.6600 | ± | 0.0476 |
| - elementary_mathematics | 1 | none | 0 | acc | ↑ | 0.4600 | ± | 0.0501 |
| - high_school_biology | 1 | none | 0 | acc | ↑ | 0.7600 | ± | 0.0429 |
| - high_school_chemistry | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.0492 |
| - high_school_computer_science | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0441 |
| - high_school_mathematics | 1 | none | 0 | acc | ↑ | 0.4100 | ± | 0.0494 |
| - high_school_physics | 1 | none | 0 | acc | ↑ | 0.4800 | ± | 0.0502 |
| - high_school_statistics | 1 | none | 0 | acc | ↑ | 0.5300 | ± | 0.0502 |
| - machine_learning | 1 | none | 0 | acc | ↑ | 0.4800 | ± | 0.0502 |
