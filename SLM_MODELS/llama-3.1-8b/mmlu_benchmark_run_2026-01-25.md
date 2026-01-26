# MMLU Benchmark Run - 2026-01-25

This document tracks the results of an MMLU benchmark run with a limit of 20 samples per subtask.

## Steps to Reproduce

The benchmark was run with the following command:

```bash
.venv/bin/lm_eval --model hf --model_args '{"pretrained": "SLM_MODELS/llama-3.1-8b-hf", "device_map": "auto"}' --tasks mmlu --device mps --batch_size 8 --output_path SLM_MODELS/llama-3.1-8b/mmlu_results.json --limit 20
```

*Note: This command requires the prerequisites outlined in `mmlu_benchmark_setup_and_results_preliminary.md` to be met, including a Python virtual environment and the downloaded Hugging Face model.*

## Results

The overall MMLU accuracy for this run was **0.6842**.

Here is a summary of the accuracy results:

| Groups            | Value  | Stderr |
|-------------------|---:|---:|
| mmlu              | 0.6842 | 0.0131 |
| - humanities      | 0.7231 | 0.0275 |
| - other           | 0.7231 | 0.0274 |
| - social sciences | 0.7458 | 0.0270 |
| - stem            | 0.5921 | 0.0237 |

## Detailed Results

| Tasks | Version | Filter | n-shot | Metric | | Value | | Stderr |
| :--- | ---: | :--- | ---: | :--- | :--- | ---: | :--- | ---: |
| mmlu | 2 | none | | acc | ↑ | 0.6842 | ± | 0.0131 |
| - humanities | 2 | none | | acc | ↑ | 0.7231 | ± | 0.0275 |
| - formal_logic | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - high_school_european_history | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - high_school_us_history | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - high_school_world_history | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - international_law | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - jurisprudence | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - logical_fallacies | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - moral_disputes | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - moral_scenarios | 1 | none | 0 | acc | ↑ | 0.6500 | ± | 0.1094 |
| - philosophy | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - prehistory | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - professional_law | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.1147 |
| - world_religions | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - other | 2 | none | | acc | ↑ | 0.7231 | ± | 0.0274 |
| - business_ethics | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - clinical_knowledge | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - college_medicine | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - global_facts | 1 | none | 0 | acc | ↑ | 0.5500 | ± | 0.1141 |
| - human_aging | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - management | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - marketing | 1 | none | 0 | acc | ↑ | 0.7000 | ± | 0.1051 |
| - medical_genetics | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - miscellaneous | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - nutrition | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - professional_accounting | 1 | none | 0 | acc | ↑ | 0.4500 | ± | 0.1141 |
| - professional_medicine | 1 | none | 0 | acc | ↑ | 0.9000 | ± | 0.0688 |
| - virology | 1 | none | 0 | acc | ↑ | 0.6500 | ± | 0.1094 |
| - social sciences | 2 | none | | acc | ↑ | 0.7458 | ± | 0.0270 |
| - econometrics | 1 | none | 0 | acc | ↑ | 0.4000 | ± | 0.1124 |
| - high_school_geography | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - high_school_government_and_politics | 1 | none | 0 | acc | ↑ | 0.9000 | ± | 0.0688 |
| - high_school_macroeconomics | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - high_school_microeconomics | 1 | none | 0 | acc | ↑ | 0.7000 | ± | 0.1051 |
| - high_school_psychology | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - human_sexuality | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - professional_psychology | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - public_relations | 1 | none | 0 | acc | ↑ | 0.5500 | ± | 0.1141 |
| - security_studies | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - sociology | 1 | none | 0 | acc | ↑ | 0.7000 | ± | 0.1051 |
| - us_foreign_policy | 1 | none | 0 | acc | ↑ | 0.9500 | ± | 0.0500 |
| - stem | 2 | none | | acc | ↑ | 0.5921 | ± | 0.0237 |
| - abstract_algebra | 1 | none | 0 | acc | ↑ | 0.4500 | ± | 0.1141 |
| - anatomy | 1 | none | 0 | acc | ↑ | 0.7500 | ± | 0.0993 |
| - astronomy | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - college_biology | 1 | none | 0 | acc | ↑ | 0.9000 | ± | 0.0688 |
| - college_chemistry | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.1147 |
| - college_computer_science | 1 | none | 0 | acc | ↑ | 0.4500 | ± | 0.1141 |
| - college_mathematics | 1 | none | 0 | acc | ↑ | 0.3000 | ± | 0.1051 |
| - college_physics | 1 | none | 0 | acc | ↑ | 0.5500 | ± | 0.1141 |
| - computer_security | 1 | none | 0 | acc | ↑ | 0.6500 | ± | 0.1094 |
| - conceptual_physics | 1 | none | 0 | acc | ↑ | 0.8000 | ± | 0.0918 |
| - electrical_engineering | 1 | none | 0 | acc | ↑ | 0.5000 | ± | 0.1147 |
| - elementary_mathematics | 1 | none | 0 | acc | ↑ | 0.3000 | ± | 0.1051 |
| - high_school_biology | 1 | none | 0 | acc | ↑ | 0.9000 | ± | 0.0688 |
| - high_school_chemistry | 1 | none | 0 | acc | ↑ | 0.6000 | ± | 0.1124 |
| - high_school_computer_science | 1 | none | 0 | acc | ↑ | 0.8500 | ± | 0.0819 |
| - high_school_mathematics | 1 | none | 0 | acc | ↑ | 0.3500 | ± | 0.1094 |
| - high_school_physics | 1 | none | 0 | acc | ↑ | 0.3500 | ± | 0.1094 |
| - high_school_statistics | 1 | none | 0 | acc | ↑ | 0.6500 | ± | 0.1094 |
| - machine_learning | 1 | none | 0 | acc | ↑ | 0.5500 | ± | 0.1141 |
