# Popular Benchmark Datasets for Small Language Models (SLMs)

This document provides a comprehensive guide to popular benchmark datasets used for evaluating Small Language Models (SLMs). These datasets help assess various capabilities including general knowledge, reasoning, coding, multilingual understanding, and more.

## Table of Contents
- [Introduction](#introduction)
- [General Knowledge Benchmarks](#general-knowledge-benchmarks)
- [Reasoning Benchmarks](#reasoning-benchmarks)
- [Coding Benchmarks](#coding-benchmarks)
- [Math and Scientific Reasoning](#math-and-scientific-reasoning)
- [Multilingual Benchmarks](#multilingual-benchmarks)
- [Long Context Benchmarks](#long-context-benchmarks)
- [Domain-Specific Benchmarks](#domain-specific-benchmarks)
- [Comprehensive Evaluation Frameworks](#comprehensive-evaluation-frameworks)
- [How to Use These Benchmarks](#how-to-use-these-benchmarks)
- [Best Practices](#best-practices)

---

## Introduction

Benchmarking is essential for:
- **Comparing models** - Understand relative strengths and weaknesses
- **Tracking progress** - Measure improvements during training or fine-tuning
- **Task selection** - Choose the right model for specific use cases
- **Research** - Contribute to the community's understanding of SLM capabilities

---

## General Knowledge Benchmarks

### MMLU (Massive Multitask Language Understanding)

- **Description**: Tests knowledge across 57 subjects including STEM, humanities, social sciences, and more
- **Tasks**: 15,908 multiple-choice questions
- **Difficulty**: Ranges from elementary to professional level
- **Format**: 4-choice multiple choice questions
- **Key Areas**: Mathematics, History, Computer Science, Law, Medicine, Ethics, etc.
- **Why Important**: Gold standard for measuring broad knowledge and reasoning
- **Access**: 
  - Hugging Face: `cais/mmlu`
  - Paper: [Measuring Massive Multitask Language Understanding](https://arxiv.org/abs/2009.03300)
  - GitHub: [hendrycks/test](https://github.com/hendrycks/test)
- **Typical SLM Scores**: 
  - 3B models: 40-55%
  - 7B models: 55-70%
  - 8B+ models: 65-75%

### ARC (AI2 Reasoning Challenge)

- **Description**: Grade-school science questions requiring reasoning
- **Versions**: 
  - ARC-Easy: 2,376 easier questions
  - ARC-Challenge: 1,172 harder questions requiring complex reasoning
- **Format**: Multiple-choice questions with 4 options
- **Key Skills**: Scientific reasoning, common sense, multi-step inference
- **Why Important**: Tests reasoning beyond simple fact recall
- **Access**:
  - Hugging Face: `ai2_arc`
  - Website: [allenai.org/data/arc](https://allenai.org/data/arc)
- **Typical SLM Scores** (ARC-Challenge):
  - 3B models: 50-65%
  - 7B models: 60-75%
  - 8B+ models: 70-80%

### TruthfulQA

- **Description**: Measures model's tendency to reproduce falsehoods and misconceptions
- **Tasks**: 817 questions across 38 categories
- **Format**: Multiple-choice and generation tasks
- **Key Areas**: Health, law, finance, politics, conspiracies
- **Why Important**: Tests factual accuracy and truthfulness
- **Access**:
  - Hugging Face: `truthful_qa`
  - GitHub: [sylinrl/TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- **Note**: Smaller models often struggle with this benchmark

### CommonsenseQA

- **Description**: Tests commonsense reasoning through question answering
- **Tasks**: 12,247 questions with one correct answer each
- **Format**: 5-choice multiple choice
- **Key Skills**: Everyday reasoning, world knowledge
- **Why Important**: Evaluates practical reasoning abilities
- **Access**:
  - Hugging Face: `commonsense_qa`
  - Website: [taucommonsenseqa.github.io](https://www.tau-nlp.sites.tau.ac.il/commonsenseqa)

---

## Reasoning Benchmarks

### HellaSwag

- **Description**: Tests commonsense natural language inference
- **Tasks**: 10,000+ sentence completion problems
- **Format**: Choose the most plausible continuation from 4 options
- **Key Skills**: Commonsense reasoning, activity understanding
- **Why Important**: Requires physical and social commonsense
- **Access**:
  - Hugging Face: `hellaswag`
  - Website: [rowanzellers.com/hellaswag](https://rowanzellers.com/hellaswag/)
- **Typical SLM Scores**:
  - 3B models: 60-75%
  - 7B models: 75-85%
  - 8B+ models: 80-90%

### WinoGrande

- **Description**: Tests commonsense reasoning through pronoun resolution
- **Tasks**: 44,000 problems across 5 difficulty levels
- **Format**: Fill-in-the-blank with pronoun reference
- **Key Skills**: Commonsense reasoning, contextual understanding
- **Why Important**: Adversarially constructed to be challenging for models
- **Access**:
  - Hugging Face: `winogrande`
  - Website: [winogrande.allenai.org](https://winogrande.allenai.org/)

### PIQA (Physical Interaction QA)

- **Description**: Tests physical commonsense reasoning
- **Tasks**: 21,000 multiple-choice questions
- **Format**: Choose the more sensible solution to achieve a goal
- **Key Skills**: Physical reasoning, practical knowledge
- **Why Important**: Evaluates understanding of physical world
- **Access**:
  - Hugging Face: `piqa`
  - Website: [yonatanbisk.com/piqa](https://yonatanbisk.com/piqa/)

### BoolQ

- **Description**: Question answering dataset with yes/no questions
- **Tasks**: 15,942 naturally occurring questions
- **Format**: Binary (yes/no) questions with passages
- **Key Skills**: Reading comprehension, inference
- **Why Important**: Tests understanding of implicit information
- **Access**:
  - Hugging Face: `boolq`
  - GitHub: [google-research-datasets/boolean-questions](https://github.com/google-research-datasets/boolean-questions)

---

## Coding Benchmarks

### HumanEval

- **Description**: Evaluates code generation from docstrings
- **Tasks**: 164 hand-written Python programming problems
- **Format**: Function signature + docstring → complete implementation
- **Key Skills**: Algorithm implementation, Python proficiency
- **Why Important**: Standard benchmark for code generation
- **Access**:
  - Hugging Face: `openai_humaneval`
  - GitHub: [openai/human-eval](https://github.com/openai/human-eval)
- **Typical SLM Scores**:
  - 3B models: 20-40% (pass@1)
  - 7B models: 35-65% (pass@1)
  - 8B+ models: 55-75% (pass@1)
- **Evaluation**: Uses pass@k metric (percentage passing unit tests)

### MBPP (Mostly Basic Python Problems)

- **Description**: Python programming problems at various difficulty levels
- **Tasks**: 974 short Python programming problems
- **Format**: Text description → Python function
- **Key Skills**: Basic Python programming, problem solving
- **Why Important**: Broader coverage than HumanEval, includes easier problems
- **Access**:
  - Hugging Face: `mbpp`
  - GitHub: [google-research/google-research/tree/master/mbpp](https://github.com/google-research/google-research/tree/master/mbpp)
- **Subsets**: 
  - MBPP (full): 974 problems
  - MBPP-sanitized: 427 problems (manually verified)

### DS-1000

- **Description**: Data science code generation benchmark
- **Tasks**: 1,000 problems covering data science libraries
- **Libraries**: NumPy, Pandas, SciPy, Scikit-learn, PyTorch, TensorFlow, Matplotlib
- **Format**: Real-world data science tasks
- **Why Important**: Tests practical data science coding skills
- **Access**:
  - GitHub: [xlang-ai/DS-1000](https://github.com/xlang-ai/DS-1000)

### CodeXGLUE

- **Description**: Comprehensive code understanding and generation benchmark suite
- **Tasks**: Multiple tasks including:
  - Code-to-code translation
  - Code summarization
  - Code search
  - Code repair
  - Clone detection
- **Languages**: Multiple programming languages
- **Why Important**: Tests diverse code-related capabilities
- **Access**:
  - GitHub: [microsoft/CodeXGLUE](https://github.com/microsoft/CodeXGLUE)

---

## Math and Scientific Reasoning

### GSM8K (Grade School Math 8K)

- **Description**: Grade school math word problems requiring multi-step reasoning
- **Tasks**: 8,500 problems (7,473 training, 1,000 test)
- **Format**: Natural language math problems
- **Key Skills**: Arithmetic reasoning, multi-step problem solving
- **Why Important**: Tests mathematical reasoning and chain-of-thought
- **Access**:
  - Hugging Face: `gsm8k`
  - GitHub: [openai/grade-school-math](https://github.com/openai/grade-school-math)
- **Typical SLM Scores**:
  - 3B models: 15-35%
  - 7B models: 30-60%
  - 8B+ models: 50-75%

### MATH

- **Description**: Challenging competition-level mathematics problems
- **Tasks**: 12,500 problems across 7 subjects
- **Subjects**: Algebra, Counting & Probability, Geometry, Intermediate Algebra, Number Theory, Prealgebra, Precalculus
- **Difficulty**: Ranges from AMC 10 to IMO level
- **Format**: Free-form answers with step-by-step solutions
- **Why Important**: Tests advanced mathematical reasoning
- **Access**:
  - Hugging Face: `hendrycks/math`
  - GitHub: [hendrycks/math](https://github.com/hendrycks/math)
- **Note**: Very challenging - even large models score <50%

### SciQ

- **Description**: Science question answering dataset
- **Tasks**: 13,679 crowdsourced science questions
- **Format**: Multiple-choice with support paragraph
- **Topics**: Physics, Chemistry, Biology
- **Why Important**: Tests scientific knowledge
- **Access**:
  - Hugging Face: `sciq`
  - Website: [allenai.org/data/sciq](https://allenai.org/data/sciq)

### TheoremQA

- **Description**: Theorem-driven question answering in STEM
- **Tasks**: 800 questions requiring theorem application
- **Format**: Questions requiring mathematical theorem application
- **Why Important**: Tests deep mathematical and scientific reasoning
- **Access**:
  - GitHub: [wenhuchen/TheoremQA](https://github.com/wenhuchen/TheoremQA)

---

## Multilingual Benchmarks

### XNLI (Cross-lingual NLI)

- **Description**: Cross-lingual natural language inference
- **Languages**: 15 languages including English, French, Spanish, German, Arabic, Hindi, Chinese, etc.
- **Tasks**: 7,500 premise-hypothesis pairs
- **Format**: Classify relationship as entailment, contradiction, or neutral
- **Why Important**: Tests multilingual understanding
- **Access**:
  - Hugging Face: `xnli`
  - Website: [cims.nyu.edu/~sbowman/xnli](https://cims.nyu.edu/~sbowman/xnli/)

### XCOPA

- **Description**: Cross-lingual Choice of Plausible Alternatives
- **Languages**: 11 languages
- **Tasks**: Causal reasoning in multiple languages
- **Format**: Given premise, choose correct cause or effect
- **Why Important**: Tests causal reasoning across languages
- **Access**:
  - Hugging Face: `xcopa`
  - GitHub: [cambridgeltl/xcopa](https://github.com/cambridgeltl/xcopa)

### MGSM (Multilingual Grade School Math)

- **Description**: GSM8K translated into 10 languages
- **Languages**: Bengali, Chinese, French, German, Japanese, Russian, Spanish, Swahili, Telugu, Thai
- **Tasks**: Math word problems in multiple languages
- **Why Important**: Tests mathematical reasoning in non-English languages
- **Access**:
  - Hugging Face: `juletxara/mgsm`

### Belebele

- **Description**: Massively multilingual reading comprehension
- **Languages**: 122 languages
- **Tasks**: Reading comprehension questions
- **Format**: Passage + multiple-choice question
- **Why Important**: Most comprehensive multilingual benchmark
- **Access**:
  - Hugging Face: `facebook/belebele`
  - Paper: [Belebele benchmark](https://arxiv.org/abs/2308.16884)

---

## Long Context Benchmarks

### SCROLLS

- **Description**: Standardized CompaRison Over Long Language Sequences
- **Tasks**: 7 diverse tasks requiring long-context understanding
- **Context Lengths**: Up to 16K tokens
- **Tasks Include**:
  - QMSum: Meeting summarization
  - SQuALITY: Long-document QA
  - Qasper: Question answering on scientific papers
  - NarrativeQA: Story understanding
- **Why Important**: Tests long-context reasoning capabilities
- **Access**:
  - Hugging Face: `tau/scrolls`
  - Website: [scrolls-benchmark.com](https://www.scrolls-benchmark.com/)

### LongBench

- **Description**: Comprehensive long-context understanding benchmark
- **Tasks**: 21 tasks across 6 categories
- **Context Lengths**: Up to 128K tokens
- **Languages**: English and Chinese
- **Categories**: Single-doc QA, multi-doc QA, summarization, few-shot learning, code, synthetic
- **Why Important**: Tests extreme long-context capabilities
- **Access**:
  - GitHub: [THUDM/LongBench](https://github.com/THUDM/LongBench)

### RULER

- **Description**: Tests retrieval capabilities at various context lengths
- **Context Lengths**: 4K to 128K tokens
- **Tasks**: Needle-in-haystack and multi-needle retrieval
- **Why Important**: Stress tests long-context retrieval
- **Access**:
  - GitHub: [hsiehjackson/RULER](https://github.com/hsiehjackson/RULER)

---

## Domain-Specific Benchmarks

### MedQA / MedMCQA

- **Description**: Medical question answering datasets
- **Tasks**: 
  - MedQA: US Medical License Exam questions
  - MedMCQA: Indian medical entrance exam questions
- **Format**: Multiple-choice medical questions
- **Why Important**: Tests medical knowledge
- **Access**:
  - Hugging Face: `bigbio/med_qa`, `medmcqa`

### LegalBench

- **Description**: Legal reasoning benchmark
- **Tasks**: 162 tasks across 6 types of legal reasoning
- **Format**: Various legal reasoning tasks
- **Why Important**: Tests legal domain knowledge
- **Access**:
  - GitHub: [HazyResearch/legalbench](https://github.com/HazyResearch/legalbench)

### FinanceBench

- **Description**: Financial question answering
- **Tasks**: Questions on financial documents
- **Format**: QA over financial reports
- **Why Important**: Tests financial domain expertise
- **Access**:
  - GitHub: [patronus-ai/financebench](https://github.com/patronus-ai/financebench)

---

## Comprehensive Evaluation Frameworks

### Open LLM Leaderboard (Hugging Face)

- **Description**: Comprehensive evaluation across multiple benchmarks
- **Benchmarks Included**:
  - MMLU (general knowledge)
  - ARC (reasoning)
  - HellaSwag (commonsense)
  - TruthfulQA (truthfulness)
  - Winogrande (reasoning)
  - GSM8K (math)
- **Why Important**: Standard comparison point for models
- **Access**: [huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **Note**: Most widely used for model comparison

### HELM (Holistic Evaluation of Language Models)

- **Description**: Comprehensive, multi-metric evaluation framework
- **Metrics**: Accuracy, calibration, robustness, fairness, bias, toxicity, efficiency
- **Scenarios**: 42+ scenarios across diverse domains
- **Why Important**: Most comprehensive evaluation framework
- **Access**: 
  - Website: [crfm.stanford.edu/helm](https://crfm.stanford.edu/helm)
  - GitHub: [stanford-crfm/helm](https://github.com/stanford-crfm/helm)

### EleutherAI LM Evaluation Harness

- **Description**: Unified framework for evaluating language models
- **Benchmarks**: 60+ evaluation tasks
- **Why Important**: Most widely used evaluation library
- **Features**:
  - Standardized evaluation
  - Easy to add new tasks
  - Reproducible results
- **Access**:
  - GitHub: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **Installation**:
```bash
pip install lm-eval
```

### BIG-bench (Beyond the Imitation Game Benchmark)

- **Description**: Extremely diverse set of tasks
- **Tasks**: 200+ diverse tasks from the research community
- **Format**: Various (multiple choice, generation, etc.)
- **Why Important**: Tests diverse capabilities beyond standard benchmarks
- **Access**:
  - GitHub: [google/BIG-bench](https://github.com/google/BIG-bench)

---

## How to Use These Benchmarks

### Quick Start with EleutherAI Harness

The easiest way to benchmark your SLM:

```bash
# Install
pip install lm-eval

# Evaluate on MMLU
lm_eval --model hf \
    --model_args pretrained=microsoft/phi-2 \
    --tasks mmlu \
    --device cuda:0 \
    --batch_size 8

# Evaluate on multiple benchmarks
lm_eval --model hf \
    --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3 \
    --tasks mmlu,arc_challenge,hellaswag,gsm8k \
    --device cuda:0 \
    --batch_size 8 \
    --output_path results/
```

### Using Hugging Face Datasets

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load benchmark dataset
dataset = load_dataset("cais/mmlu", "all")

# Load model
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Evaluate on a sample
sample = dataset["test"][0]
question = sample["question"]
choices = sample["choices"]
correct_answer = sample["answer"]

# Format prompt
prompt = f"Question: {question}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"

# Generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=5)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(f"Response: {response}")
print(f"Correct answer: {choices[correct_answer]}")
```

### Using HELM

```bash
# Install HELM
pip install crfm-helm

# Run evaluation
helm-run --run-entries <model_name>:task=<task_name> --max-eval-instances 100

# Example
helm-run --run-entries mistral-7b:task=mmlu --max-eval-instances 1000
```

### Custom Evaluation Script

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

def evaluate_mmlu(model_name, subset="all", num_samples=None):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Load dataset
    dataset = load_dataset("cais/mmlu", subset)["test"]
    if num_samples:
        dataset = dataset.select(range(num_samples))
    
    correct = 0
    total = 0
    
    for sample in tqdm(dataset):
        # Format question
        question = sample["question"]
        choices = sample["choices"]
        answer_idx = sample["answer"]
        
        # Create prompt
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer:"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Extract answer
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        predicted_answer = response.strip()[0].upper() if response else ""
        correct_answer = chr(65 + answer_idx)
        
        if predicted_answer == correct_answer:
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy on MMLU ({subset}): {accuracy:.2f}%")
    return accuracy

# Run evaluation
evaluate_mmlu("microsoft/phi-2", num_samples=100)
```

---

## Best Practices

### Choosing Benchmarks

1. **Match your use case**: 
   - Code generation → HumanEval, MBPP
   - General chat → MMLU, HellaSwag, ARC
   - Multilingual → XNLI, Belebele, MGSM
   - Long context → SCROLLS, LongBench

2. **Use multiple benchmarks**: No single benchmark captures all capabilities

3. **Consider your constraints**:
   - Limited compute → Start with smaller subsets
   - Quick iteration → Use faster benchmarks (ARC, HellaSwag)
   - Comprehensive evaluation → Use frameworks (HELM, LM Eval Harness)

### Evaluation Tips

1. **Use consistent settings**:
   - Same temperature (usually 0 for benchmarks)
   - Same prompt format
   - Same generation parameters

2. **Report multiple metrics**:
   - Not just accuracy
   - Include pass@k for code, F1 for generation tasks

3. **Test on multiple shots**:
   - 0-shot (no examples)
   - Few-shot (with examples)
   - Compare performance

4. **Consider calibration**:
   - Are confidence scores meaningful?
   - Test on TruthfulQA

5. **Monitor efficiency**:
   - Tokens per second
   - Memory usage
   - Cost per token

### Reproducibility

- **Fix random seeds**: Ensure consistent results
- **Document versions**: Model version, library versions, dataset versions
- **Share code**: Make evaluations reproducible
- **Report infrastructure**: Hardware, quantization, batch size

### Common Pitfalls

❌ **Don't**:
- Cherry-pick benchmarks where your model performs well
- Overfit to specific benchmark formats during training
- Compare models evaluated with different settings
- Ignore variance across runs
- Only report aggregate scores (show per-task results)

✅ **Do**:
- Use standard evaluation frameworks
- Report results on multiple benchmarks
- Include baseline comparisons
- Document all evaluation details
- Test on held-out data

---

## Resources

### Evaluation Tools
- **EleutherAI LM Evaluation Harness**: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- **HELM**: [github.com/stanford-crfm/helm](https://github.com/stanford-crfm/helm)
- **OpenAI Evals**: [github.com/openai/evals](https://github.com/openai/evals)

### Leaderboards
- **Open LLM Leaderboard**: [huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **HELM Leaderboard**: [crfm.stanford.edu/helm/latest](https://crfm.stanford.edu/helm/latest/)
- **Papers with Code**: [paperswithcode.com/sota](https://paperswithcode.com/sota)

### Dataset Collections
- **Hugging Face Datasets**: [huggingface.co/datasets](https://huggingface.co/datasets)
- **Papers with Code Datasets**: [paperswithcode.com/datasets](https://paperswithcode.com/datasets)

### Communities
- **EleutherAI Discord**: Active community for LLM evaluation
- **Hugging Face Forums**: Discussions on benchmarking
- **r/LocalLLaMA**: Reddit community for local model discussion

---

## Quick Reference Table

| Benchmark | Type | Size | Difficulty | Best For |
|-----------|------|------|------------|----------|
| MMLU | Knowledge | 15K | Medium-Hard | General knowledge |
| ARC-C | Reasoning | 1.2K | Hard | Science reasoning |
| HellaSwag | Reasoning | 10K | Medium | Commonsense |
| HumanEval | Coding | 164 | Hard | Code generation |
| GSM8K | Math | 8.5K | Medium | Math reasoning |
| MBPP | Coding | 974 | Easy-Medium | Python coding |
| TruthfulQA | Truthfulness | 817 | Hard | Factual accuracy |
| WinoGrande | Reasoning | 44K | Medium | Commonsense |
| XNLI | Multilingual | 7.5K | Medium | Cross-lingual |
| LongBench | Long Context | Varies | Hard | Context understanding |

---

## Contributing

Benchmarks evolve constantly. If you know of important benchmarks not listed here, please contribute! This is especially valuable for:
- Emerging benchmarks
- Domain-specific evaluations
- Regional/language-specific datasets
- Novel evaluation paradigms

---

*Last Updated: December 2025*

*Note: Benchmark scores and availability are subject to change. Always verify current leaderboard standings and dataset availability before use.*
