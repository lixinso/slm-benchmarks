# MMLU Benchmark Run - 2026-01-26 (limit 50)

This document tracks the results of an MMLU benchmark run with a limit of 50 samples per subtask.

## Steps to Reproduce

The benchmark was run with the following command, using the `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` environment variable to prevent MPS memory errors:

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 .venv/bin/lm_eval --model hf --model_args '{"pretrained": "SLM_MODELS/llama-3.1-8b-hf", "device_map": "auto"}' --tasks mmlu --device mps --batch_size 8 --output_path SLM_MODELS/llama-3.1-8b/mmlu_results_limit_50.json --limit 50
```

*Note: This command requires the prerequisites outlined in `mmlu_benchmark_setup_and_results_preliminary.md` to be met, including a Python virtual environment and the downloaded Hugging Face model.*

## Results

The overall MMLU accuracy for this run was **0.6825**.

### Summary

| Groups            | Value  | Stderr |
|-------------------|---:|---:|
| mmlu              | 0.6825 | 0.0083 |
| - humanities      | 0.7169 | 0.0173 |
| - other           | 0.7169 | 0.0169 |
| - social sciences | 0.7533 | 0.0170 |
| - stem            | 0.5905 | 0.0153 |

### Detailed Results

| Tasks | Version | Filter | n-shot | Metric | | Value | | Stderr |
| :--- | ---: | :--- | ---: | :--- | :--- | ---: | :--- | ---: |
| mmlu | 2 | none | | acc | ↑ | 0.6825 | ± | 0.0083 |
| - humanities | 2 | none | | acc | ↑ | 0.7169 | ± | 0.0173 |
| - formal_logic | 1 | none | 0 | acc | ↑ | 0.5600 | ± | 0.0709 |
| - high_school_european_history | 1 | none | 0 | acc | ↑ | 0.7200 | ± | 0.0641 |
| - high_school_us_history | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0524 |
| - high_school_world_history | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - international_law | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0524 |
| - jurisprudence | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - logical_fallacies | 1 | none | 0 | acc | ↑ | 0.8200 | ± | 0.0549 |
| - moral_disputes | 1 | none | 0 | acc | ↑ | 0.6200 | ± | 0.0693 |
| - moral_scenarios | 1 | none | 0 | acc | ↑ | 0.5800 | ± | 0.0705 |
| - philosophy | 1 | none | 0 | acc | ↑ | 0.6800 | ± | 0.0666 |
| - prehistory | 1 | none | 0 | acc | ↑ | 0.7600 | ± | 0.0610 |
| - professional_law | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.0714 |
| - world_religions | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0524 |
| - other | 2 | none | | acc | ↑ | 0.7169 | ± | 0.0169 |
| - business_ethics | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0627 |
| - clinical_knowledge | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0627 |
| - college_medicine | 1 | none | 0 | acc | ↑ | 0.7200 | ± | 0.0641 |
| - global_facts | 1 | none | 0 | acc | ↑ | 0.3800 | ± | 0.0693 |
| - human_aging | 1 | none | 0 | acc | ↑ | 0.7000 | ± | 0.0655 |
| - management | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0524 |
| - marketing | 1 | none | 0 | acc | ↑ | 0.8600 | ± | 0.0496 |
| - medical_genetics | 1 | none | 0 | acc | ↑ | 0.7600 | ± | 0.0610 |
| - miscellaneous | 1 | none | 0 | acc | ↑ | 0.8600 | ± | 0.0496 |
| - nutrition | 1 | none | 0 | acc | ↑ | 0.8400 | ± | 0.0524 |
| - professional_accounting | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.0714 |
| - professional_medicine | 1 | none | 0 | acc | ↑ | 0.8200 | ± | 0.0549 |
| - virology | 1 | none | 0 | acc | ↑ | 0.5600 | ± | 0.0709 |
| - social sciences | 2 | none | | acc | ↑ | 0.7533 | ± | 0.0170 |
| - econometrics | 1 | none | 0 | acc | ↑ | 0.4400 | ± | 0.0709 |
| - high_school_geography | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0571 |
| - high_school_government_and_politics | 1 | none | 0 | acc | ↑ | 0.9400 | ± | 0.0339 |
| - high_school_macroeconomics | 1 | none | 0 | acc | ↑ | 0.7000 | ± | 0.0655 |
| - high_school_microeconomics | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - high_school_psychology | 1 | none | 0 | acc | ↑ | 0.8800 | ± | 0.0464 |
| - human_sexuality | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - professional_psychology | 1 | none | 0 | acc | ↑ | 0.6400 | ± | 0.0686 |
| - public_relations | 1 | none | 0 | acc | ↑ | 0.6600 | ± | 0.0677 |
| - security_studies | 1 | none | 0 | acc | ↑ | 0.7400 | ± | 0.0627 |
| - sociology | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0571 |
| - us_foreign_policy | 1 | none | 0 | acc | ↑ | 0.8800 | ± | 0.0464 |
| - stem | 2 | none | | acc | ↑ | 0.5905 | ± | 0.0153 |
| - abstract_algebra | 1 | none | 0 | acc | ↑ | 0.3800 | ± | 0.0693 |
| - anatomy | 1 | none | 0 | acc | ↑ | 0.6800 | ± | 0.0666 |
| - astronomy | 1 | none | 0 | acc | ↑ | 0.8200 | ± | 0.0549 |
| - college_biology | 1 | none | 0 | acc | ↑ | 0.9000 | ± | 0.0429 |
| - college_chemistry | 1 | none | 0 | acc | ↑ | 0.4600 | ± | 0.0712 |
| - college_computer_science | 1 | none | 0 | acc | ↑ | 0.5600 | ± | 0.0709 |
| - college_mathematics | 1 | none | 0 | acc | ↑ | 0.3800 | ± | 0.0693 |
| - college_physics | 1 | none | 0 | acc | ↑ | 0.4600 | ± | 0.0712 |
| - computer_security | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - conceptual_physics | 1 | none | 0 | acc | ↑ | 0.5800 | ± | 0.0705 |
| - electrical_engineering | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.0700 |
| - elementary_mathematics | 1 | none | 0 | acc | ↑ | 0.4800 | ± | 0.0714 |
| - high_school_biology | 1 | none | 0 | acc | ↑ | 0.7800 | ± | 0.0592 |
| - high_school_chemistry | 1 | none | 0 | acc | ↑ | 0.6200 | ± | 0.0693 |
| - high_school_computer_science | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0571 |
| - high_school_mathematics | 1 | none | 0 | acc | ↑ | 0.4000 | ± | 0.0700 |
| - high_school_physics | 1 | none | 0 | acc | ↑ | 0.4800 | ± | 0.0714 |
| - high_school_statistics | 1 | none | 0 | acc | ↑ | 0.5400 | ± | 0.0712 |
| - machine_learning | 1 | none | 0 | acc | ↑ | 0.5200 | ± | 0.0714 |
