# Llama 3.1 8B MMLU Benchmark - Preliminary Setup and Results

## 1. Introduction
This document outlines the steps taken to set up and run a preliminary MMLU (Massive Multitask Language Understanding) benchmark for the Llama 3.1 8B model using the `lm-evaluation-harness` framework. The goal was to establish a functional benchmarking pipeline for capability assessment.

## 2. Prerequisites
*   **Ollama**: Used for local serving of LLMs (though ultimately not used for MMLU evaluation due to API limitations with `lm-eval`).
*   **Python 3.9+**: For running `lm-evaluation-harness`.
*   **Virtual Environment**: Recommended for managing Python dependencies.
*   **Hugging Face Account**: Required to access gated models like Llama 3.1 and download their weights. Access to the specific model must be granted by the model authors.

## 3. Setup Steps

### 3.1. Create and Activate Virtual Environment
A virtual environment was created in the project root to isolate dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate
```
The `.venv` folder was added to `.gitignore`.

### 3.2. Install `lm-evaluation-harness` and Dependencies
`lm-evaluation-harness` was installed with its API-related dependencies within the virtual environment. `bitsandbytes` was also required for quantization.
```bash
.venv/bin/pip install "lm-eval[api]"
.venv/bin/pip install -U bitsandbytes
```
*(Note: Initial attempts with `lm-eval[ollama]` failed as the extra was not provided by the installed version. Also, direct 8-bit/4-bit quantization with `bitsandbytes` failed due to CUDA requirement on macOS and the lack of a suitable `bitsandbytes` version for non-CUDA. This led to using `device_map="auto"`.)*

### 3.3. Hugging Face Authentication
To download the Llama 3.1 model weights, authentication with Hugging Face was required.
1.  **Obtain User Access Token**: A read-access token was generated from the Hugging Face website (Settings -> Access Tokens).
2.  **Login via CLI**: The token was used to log in via the Hugging Face CLI.
    ```bash
    .venv/bin/huggingface-cli login --token <YOUR_HF_TOKEN>
    ```
    *(Note: Access to the `meta-llama/Meta-Llama-3.1-8B-Instruct` model itself also needed to be explicitly requested and approved on the Hugging Face model page.)*

### 3.4. Download Llama 3.1 8B Instruct Model Weights
The model weights were downloaded to a local directory for `lm-evaluation-harness` to load directly.
```bash
.venv/bin/huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir SLM_MODELS/llama-3.1-8b-hf --local-dir-use-symlinks False
```

## 4. MMLU Benchmark Execution (Preliminary Run)

### 4.1. Challenges Encountered and Solutions
*   **Ollama API Incompatibility**: Initial attempts to use `lm-eval` with Ollama's OpenAI-compatible API (`openai-chat-completions` or `local-completions`) failed because MMLU requires log-likelihoods, which are not directly supported by the chat completion endpoint in `lm-eval` and the `completions` endpoint was not found.
*   **`bitsandbytes` CUDA Requirement**: When attempting to load the downloaded model with 8-bit or 4-bit quantization on macOS, `bitsandbytes` raised an `ImportError` due to a CUDA dependency, and no compatible non-CUDA version was found for the installed `bitsandbytes` version.
*   **Solution**: The model was loaded directly by `lm-eval` using the Hugging Face backend (`--model hf`) with `device_map="auto"` to allow automatic distribution of model layers between GPU (MPS) and CPU, bypassing the `bitsandbytes` issue.

### 4.2. Benchmark Command
The following command was successfully executed for a preliminary run (limited to 10 samples per subtask):
```bash
.venv/bin/lm_eval --model hf --model_args '{"pretrained": "SLM_MODELS/llama-3.1-8b-hf", "device_map": "auto"}' --tasks mmlu --device mps --batch_size 8 --output_path SLM_MODELS/llama-3.1-8b/mmlu_results.json --limit 10
```

## 5. Preliminary Results (Limited Run - 10 samples per subtask)

| Tasks                            | Version | Filter | n-shot | Metric |    | Value  |    | Stderr |
| :------------------------------- | ------: | :----- | -----: | :----- | :--- | -----: | :--- | -----: |
| mmlu                             |       2 | none   |        | acc    | ↑  | 0.6930 | ±  | 0.0185 |
| - humanities                     |       2 | none   |        | acc    | ↑  | 0.7615 | ±  | 0.0360 |
|  - formal_logic                  |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
|  - high_school_european_history  |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - high_school_us_history        |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - high_school_world_history     |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - international_law             |       1 | none   |      0 | acc    | ↑  | 1.0000 | ±  | 0.0000 |
|  - jurisprudence                 |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - logical_fallacies             |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - moral_disputes                |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
|  - moral_scenarios               |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
|  - philosophy                    |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - prehistory                    |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - professional_law              |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - world_religions               |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
| - other                          |       2 | none   |        | acc    | ↑  | 0.7077 | ±  | 0.0396 |
|  - business_ethics               |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - clinical_knowledge            |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - college_medicine              |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - global_facts                  |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - human_aging                   |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - management                    |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - marketing                     |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - medical_genetics              |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - miscellaneous                 |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - nutrition                     |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - professional_accounting       |       1 | none   |      0 | acc    | ↑  | 0.4000 | ±  | 0.1633 |
|  - professional_medicine         |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - virology                      |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
| - social sciences                |       2 | none   |        | acc    | ↑  | 0.7667 | ±  | 0.0379 |
|  - econometrics                  |       1 | none   |      0 | acc    | ↑  | 0.4000 | ±  | 0.1633 |
|  - high_school_geography         |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - high_school_government_and_politics |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - high_school_macroeconomics    |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - high_school_microeconomics    |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - high_school_psychology        |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - human_sexuality               |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - professional_psychology       |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - public_relations              |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - security_studies              |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - sociology                     |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - us_foreign_policy             |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
| - stem                           |       2 | none   |        | acc    | ↑  | 0.5895 | ±  | 0.0341 |
|  - abstract_algebra              |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - anatomy                       |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - astronomy                     |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - college_biology               |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - college_chemistry             |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - college_computer_science      |       1 | none   |      0 | acc    | ↑  | 0.4000 | ±  | 0.1633 |
|  - college_mathematics           |       1 | none   |      0 | acc    | ↑  | 0.1000 | ±  | 0.1000 |
|  - college_physics               |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
|  - computer_security             |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - conceptual_physics            |       1 | none   |      0 | acc    | ↑  | 0.7000 | ±  | 0.1528 |
|  - electrical_engineering        |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |
|  - elementary_mathematics        |       1 | none   |      0 | acc    | ↑  | 0.4000 | ±  | 0.1633 |
|  - high_school_biology           |       1 | none   |      0 | acc    | ↑  | 0.9000 | ±  | 0.1000 |
|  - high_school_chemistry         |       1 | none   |      0 | acc    | ↑  | 0.6000 | ±  | 0.1633 |
|  - high_school_computer_science  |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - high_school_mathematics       |       1 | none   |      0 | acc    | ↑  | 0.3000 | ±  | 0.1528 |
|  - high_school_physics           |       1 | none   |      0 | acc    | ↑  | 0.4000 | ±  | 0.1633 |
|  - high_school_statistics        |       1 | none   |      0 | acc    | ↑  | 0.8000 | ±  | 0.1333 |
|  - machine_learning              |       1 | none   |      0 | acc    | ↑  | 0.5000 | ±  | 0.1667 |

## 6. Next Steps
To run the full MMLU benchmark, remove the `--limit 10` flag from the command. Be aware that the full benchmark will take a significant amount of time to complete.
