# Popular Benchmark Metrics for Small Language Models (SLMs)

This document provides a comprehensive guide to the most popular evaluation metrics used for benchmarking Small Language Models (SLMs). Understanding these metrics is essential for comparing models, tracking progress, and selecting the right model for your use case.

## Table of Contents
- [Introduction](#introduction)
- [Accuracy-Based Metrics](#accuracy-based-metrics)
- [Probabilistic Metrics](#probabilistic-metrics)
- [Code Generation Metrics](#code-generation-metrics)
- [Text Generation Quality Metrics](#text-generation-quality-metrics)
- [Task-Specific Metrics](#task-specific-metrics)
- [Truthfulness and Calibration Metrics](#truthfulness-and-calibration-metrics)
- [Efficiency Metrics](#efficiency-metrics)
- [Composite Metrics](#composite-metrics)
- [Best Practices](#best-practices)
- [Metrics Quick Reference](#metrics-quick-reference)

---

## Introduction

Benchmark metrics provide quantitative ways to evaluate language model performance. Different tasks require different metrics, and understanding which metrics to use is crucial for:
- **Model comparison** - Objectively compare different models
- **Progress tracking** - Measure improvements during training or fine-tuning
- **Task suitability** - Determine if a model is appropriate for a specific use case
- **Research communication** - Report results in a standardized way

### Key Principles
- **Match metrics to tasks** - Use appropriate metrics for each evaluation task
- **Report multiple metrics** - No single metric captures all aspects of performance
- **Consider baselines** - Compare against random, majority class, or human performance
- **Understand limitations** - Every metric has biases and blind spots

---

## Accuracy-Based Metrics

These are the most common metrics for classification and multiple-choice tasks.

### Accuracy

- **Definition**: Percentage of correct predictions out of all predictions
- **Formula**: `Accuracy = (Correct Predictions) / (Total Predictions)`
- **Range**: 0% to 100% (higher is better)
- **Use Cases**: 
  - Multiple-choice questions (MMLU, ARC, HellaSwag)
  - Binary classification (BoolQ)
  - General classification tasks
- **Typical SLM Scores**:
  - MMLU: 40-75% depending on model size
  - ARC-Challenge: 50-80%
  - HellaSwag: 60-90%
- **Advantages**:
  - Easy to understand and interpret
  - Standard across most benchmarks
  - Good for balanced datasets
- **Limitations**:
  - Can be misleading on imbalanced datasets
  - Doesn't capture confidence or uncertainty
  - Equal weight to all errors (no notion of "close" answers)
- **Example**:
  ```
  Total questions: 100
  Correct answers: 75
  Accuracy = 75/100 = 75%
  ```

### Precision

- **Definition**: Of all positive predictions, what percentage were actually correct
- **Formula**: `Precision = True Positives / (True Positives + False Positives)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Cases**:
  - Information retrieval
  - Named entity recognition
  - Classification tasks where false positives are costly
- **When to Use**: When you care about minimizing false alarms
- **Example**:
  ```
  Model predicted 100 items as "positive"
  80 were actually positive (True Positives)
  20 were actually negative (False Positives)
  Precision = 80 / (80 + 20) = 0.80
  ```

### Recall (Sensitivity)

- **Definition**: Of all actual positives, what percentage were correctly identified
- **Formula**: `Recall = True Positives / (True Positives + False Negatives)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Cases**:
  - Information retrieval
  - Named entity recognition
  - Classification tasks where missing positives is costly
- **When to Use**: When you care about minimizing missed detections
- **Example**:
  ```
  100 items are actually positive
  80 were correctly identified (True Positives)
  20 were missed (False Negatives)
  Recall = 80 / (80 + 20) = 0.80
  ```

### F1 Score

- **Definition**: Harmonic mean of precision and recall
- **Formula**: `F1 = 2 × (Precision × Recall) / (Precision + Recall)`
- **Range**: 0.0 to 1.0 (higher is better)
- **Use Cases**:
  - Named entity recognition
  - Information extraction
  - Any task requiring balance between precision and recall
- **Advantages**:
  - Balances precision and recall
  - More informative than accuracy on imbalanced datasets
  - Standard metric for many NLP tasks
- **Limitations**:
  - Can be hard to interpret for non-technical audiences
  - Equal weight to precision and recall (may not always be desired)
- **Variants**:
  - **Macro-F1**: Average F1 across all classes (treats all classes equally)
  - **Micro-F1**: Calculate from global true positives, false positives, false negatives
  - **Weighted-F1**: Weighted average by class support
- **Example**:
  ```
  Precision = 0.80, Recall = 0.75
  F1 = 2 × (0.80 × 0.75) / (0.80 + 0.75) = 0.774
  ```

### Top-k Accuracy

- **Definition**: Percentage of samples where correct answer is in top-k predictions
- **Formula**: `Top-k Accuracy = (Samples with correct answer in top-k) / (Total Samples)`
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - Multi-class classification with many classes
  - When multiple answers might be acceptable
- **Common Values**: k=3, k=5, k=10
- **Note**: Top-1 accuracy is standard accuracy
- **Example**:
  ```
  Top-5 accuracy = 85% means the correct answer appears 
  in the model's top 5 predictions 85% of the time
  ```

---

## Probabilistic Metrics

These metrics evaluate how well a model estimates probabilities and handles uncertainty.

### Perplexity (PPL)

- **Definition**: Measures how "surprised" a model is by the test data. Lower perplexity means better prediction of text.
- **Formula**: `PPL = exp(average negative log-likelihood)`
  - More precisely: `PPL = exp(-1/N × Σ log P(word_i | context))`
- **Range**: 1 to ∞ (lower is better)
- **Use Cases**:
  - Language modeling evaluation
  - Comparing different model architectures
  - Measuring text prediction quality
- **Typical SLM Scores**:
  - 3B models: 15-25 (depending on dataset)
  - 7B models: 10-18
  - State-of-the-art: <10 on standard benchmarks
- **Interpretation**:
  - PPL of 10 means the model is as confused as if it had to choose uniformly among 10 words
  - Lower perplexity = better language model
- **Advantages**:
  - Directly measures language modeling capability
  - Doesn't require labeled data
  - Can be computed on any text
- **Limitations**:
  - Hard to interpret absolute values
  - Not directly comparable across different tokenizers
  - Doesn't correlate perfectly with downstream task performance
- **Example**:
  ```
  Model achieves perplexity of 12.5 on WikiText-2
  This is better than a model with perplexity 18.3
  ```

### Cross-Entropy Loss

- **Definition**: Average negative log-likelihood of the correct predictions
- **Formula**: `CrossEntropy = -1/N × Σ log P(correct_class)`
- **Range**: 0 to ∞ (lower is better)
- **Use Cases**:
  - Training loss for classification tasks
  - Evaluating probability calibration
- **Relationship to Perplexity**: `Perplexity = exp(CrossEntropy)`
- **Advantages**:
  - Smooth, differentiable metric
  - Penalizes confident wrong predictions heavily
  - Standard training objective
- **Limitations**:
  - Less interpretable than accuracy
  - Sensitive to outliers
- **Example**:
  ```
  Lower cross-entropy = better probability estimates
  Cross-entropy of 0.5 → Perplexity of e^0.5 ≈ 1.65
  ```

### Bits Per Byte (BPB)

- **Definition**: Average number of bits needed to encode each byte
- **Formula**: `BPB = log2(PPL)`
- **Range**: 0 to ∞ (lower is better)
- **Use Cases**:
  - Character-level language modeling
  - Compression-based evaluation
- **Interpretation**: Lower BPB means better compression/prediction
- **Example**:
  ```
  BPB of 1.0 means model compresses data to 1 bit per byte
  Random baseline would be ~8 bits per byte
  ```

### Calibration Error

- **Definition**: Measures how well predicted probabilities match actual frequencies
- **Common Metrics**:
  - **Expected Calibration Error (ECE)**: Average difference between confidence and accuracy
  - **Maximum Calibration Error (MCE)**: Maximum difference across bins
- **Range**: 0% to 100% (lower is better)
- **Use Cases**:
  - Evaluating prediction confidence
  - Safety-critical applications
  - TruthfulQA benchmark
- **Interpretation**: Well-calibrated model: if it says 70% confident, it's right 70% of the time
- **Example**:
  ```
  ECE of 5% means on average, confidence differs from 
  actual accuracy by 5 percentage points
  ```

---

## Code Generation Metrics

Specialized metrics for evaluating code synthesis and programming tasks.

### Pass@k

- **Definition**: Percentage of problems solved by at least one of k generated solutions
- **Formula**: Generate k solutions per problem, check if any pass all test cases
- **Common Values**: k=1, k=5, k=10, k=100
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - HumanEval benchmark
  - MBPP (Mostly Basic Python Problems)
  - Any code generation task with unit tests
- **Typical SLM Scores (HumanEval)**:
  - 3B models: 20-40% (pass@1)
  - 7B models: 35-65% (pass@1)
  - 8B+ models: 55-75% (pass@1)
- **Calculation**: 
  ```
  For each problem:
    Generate k code samples
    Run unit tests on each sample
    Problem solved if ANY sample passes all tests
  Pass@k = (Problems solved) / (Total problems)
  ```
- **Advantages**:
  - Objective - based on actual code execution
  - Allows for multiple attempts (pass@10 useful for reranking)
  - Standard metric in code generation community
- **Limitations**:
  - Requires comprehensive test suites
  - Doesn't evaluate code quality or style
  - Can be gamed with trivial solutions that pass tests
- **Variants**:
  - **Strict pass@k**: All test cases must pass
  - **Partial credit**: Points for passing subset of tests
- **Example**:
  ```
  Problem: "Write a function to reverse a string"
  Generate 10 solutions (k=10)
  3 solutions pass all unit tests
  Problem counts as "solved" for pass@10 metric
  ```

### BLEU (Bilingual Evaluation Understudy)

- **Definition**: Measures n-gram overlap between generated and reference code
- **Formula**: Geometric mean of modified n-gram precisions with brevity penalty
- **Range**: 0 to 100 (higher is better)
- **Use Cases**:
  - Code translation
  - Code completion when unit tests not available
  - Measuring syntactic similarity
- **Originally Designed For**: Machine translation
- **Advantages**:
  - No execution required
  - Fast to compute
  - Can compare against multiple references
- **Limitations**:
  - Doesn't evaluate functional correctness
  - Biased toward reference solutions
  - Poor correlation with actual code quality
- **Note**: Generally less preferred than pass@k for code generation
- **Example**:
  ```
  Reference: "for i in range(len(arr)): print(arr[i])"
  Generated: "for i in range(len(arr)): print(arr[i])"
  BLEU: 100 (perfect match)
  ```

### CodeBLEU

- **Definition**: Enhanced BLEU for code that considers syntax trees and data flow
- **Components**:
  - N-gram match (like BLEU)
  - Weighted n-gram match (keyword weights)
  - Abstract syntax tree (AST) match
  - Data flow graph match
- **Range**: 0 to 100 (higher is better)
- **Use Cases**:
  - Code generation evaluation without test execution
  - Code translation
  - Measuring structural similarity
- **Advantages**:
  - Better than BLEU for code
  - Considers program structure
  - Language-specific (supports multiple programming languages)
- **Limitations**:
  - Still doesn't guarantee functional correctness
  - More complex to compute than BLEU
- **Example**:
  ```
  Generated code with different variable names but same logic
  CodeBLEU: High score due to matching AST structure
  BLEU: Lower score due to different tokens
  ```

### Edit Distance / Edit Similarity

- **Definition**: Minimum number of edits to transform generated code to reference
- **Common Variants**:
  - **Levenshtein Distance**: Character-level edits
  - **Token Edit Distance**: Token-level edits
- **Range**: 0 to ∞ (lower is better) or normalized 0-1 similarity (higher is better)
- **Use Cases**:
  - Code repair
  - Code completion
  - Measuring syntactic closeness
- **Advantages**:
  - Intuitive - counts required changes
  - Useful for code repair tasks
- **Limitations**:
  - Doesn't consider semantics
  - Can be misleading (one character change can break code)
- **Example**:
  ```
  Reference: "x = x + 1"
  Generated: "x = x + 2"
  Edit distance: 1 (change '1' to '2')
  ```

---

## Text Generation Quality Metrics

Metrics for evaluating generated text quality in summarization, translation, and open-ended generation.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

- **Definition**: Measures n-gram overlap between generated and reference text
- **Variants**:
  - **ROUGE-N**: N-gram overlap (ROUGE-1, ROUGE-2, etc.)
  - **ROUGE-L**: Longest common subsequence
  - **ROUGE-W**: Weighted longest common subsequence
  - **ROUGE-S**: Skip-bigram overlap
- **Range**: 0 to 1 (or 0% to 100%) (higher is better)
- **Use Cases**:
  - Text summarization (primary metric)
  - Machine translation
  - Question answering
- **Most Common**: ROUGE-1, ROUGE-2, ROUGE-L
- **Typical Scores (Summarization)**:
  - ROUGE-1: 35-50%
  - ROUGE-2: 15-25%
  - ROUGE-L: 30-45%
- **Interpretation**:
  - ROUGE-1: Unigram overlap (captures recall of content words)
  - ROUGE-2: Bigram overlap (captures fluency and coherence)
  - ROUGE-L: Captures sentence-level structure
- **Advantages**:
  - Standard metric for summarization
  - Recall-oriented (good for summarization)
  - Multiple variants capture different aspects
- **Limitations**:
  - Requires reference summaries
  - Doesn't evaluate factual correctness
  - Can be high even for poor summaries (if references are similar)
- **Example**:
  ```
  Reference: "The cat sat on the mat"
  Generated: "The cat was on the mat"
  ROUGE-1: High (5/6 words match)
  ROUGE-2: Lower (fewer bigrams match)
  ```

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)

- **Definition**: Alignment-based metric using exact, stem, synonym, and paraphrase matches
- **Formula**: Harmonic mean of precision and recall with fragmentation penalty
- **Range**: 0 to 1 (higher is better)
- **Use Cases**:
  - Machine translation
  - Paraphrase generation
  - Text generation evaluation
- **Advantages**:
  - Better correlation with human judgment than BLEU
  - Handles synonyms and stemming
  - Considers recall (not just precision like BLEU)
- **Limitations**:
  - Requires WordNet or language resources
  - More complex to compute
  - Language-specific
- **Example**:
  ```
  Reference: "The quick brown fox jumps"
  Generated: "The fast brown fox leaps"
  METEOR: High score (recognizes synonyms)
  BLEU: Lower score (different words)
  ```

### BERTScore

- **Definition**: Computes similarity using contextual embeddings from BERT
- **Formula**: Token-level semantic similarity using cosine similarity of embeddings
- **Range**: 0 to 1 (higher is better)
- **Components**:
  - Precision: Generated tokens matched to reference
  - Recall: Reference tokens matched to generated
  - F1: Harmonic mean of precision and recall
- **Use Cases**:
  - Text generation evaluation
  - Paraphrase detection
  - When semantic similarity matters more than exact match
- **Advantages**:
  - Captures semantic similarity beyond lexical overlap
  - More robust to paraphrasing than n-gram metrics
  - Better correlation with human judgment
- **Limitations**:
  - Requires running BERT model (computationally expensive)
  - Can be sensitive to BERT model choice
  - May not capture factual errors
- **Example**:
  ```
  Reference: "The automobile is red"
  Generated: "The car is red"
  BERTScore: High (semantic similarity)
  BLEU/ROUGE: Lower (different words)
  ```

### Perplexity (for Generation)

- **Definition**: Same as language modeling perplexity, applied to generated text
- **Range**: 1 to ∞ (lower is better)
- **Use Cases**:
  - Evaluating fluency of generated text
  - Comparing generation quality across models
- **Interpretation**: Lower perplexity = more natural/fluent text
- **Limitations**:
  - Doesn't evaluate factual correctness
  - Doesn't capture relevance to prompt
  - Can be low for generic/safe responses

### Diversity Metrics

- **Self-BLEU**: BLEU score computed between generated samples (lower is more diverse)
- **Distinct-n**: Percentage of unique n-grams in generated text (higher is more diverse)
- **Entropy**: Entropy of token distribution (higher is more diverse)
- **Use Cases**:
  - Detecting repetition in generation
  - Evaluating generation diversity
  - Story/dialogue generation
- **Range**: Varies by metric
- **Example**:
  ```
  Distinct-1 = Unique unigrams / Total unigrams
  Distinct-2 = Unique bigrams / Total bigrams
  Higher values indicate more diverse generation
  ```

---

## Task-Specific Metrics

Metrics designed for particular types of tasks.

### Exact Match (EM)

- **Definition**: Percentage of predictions that match reference exactly (character-by-character)
- **Formula**: `EM = (Exact Matches) / (Total Samples)`
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - Question answering (SQuAD)
  - Named entity recognition
  - Extractive tasks
- **Typical Scores**:
  - SQuAD: 70-90% EM
  - TriviaQA: 50-70% EM
- **Advantages**:
  - Unambiguous - answer is correct or not
  - Easy to understand
  - No partial credit (forces precision)
- **Limitations**:
  - Very strict - "7" vs "seven" counts as wrong
  - Doesn't capture semantic equivalence
  - Single word difference = failure
- **Variants**:
  - **Normalized EM**: After normalization (lowercase, punctuation removal)
  - **Fuzzy EM**: Allow minor differences
- **Example**:
  ```
  Question: "When was Python created?"
  Reference: "1991"
  Generated: "1991" → EM = 1 (correct)
  Generated: "in 1991" → EM = 0 (wrong)
  ```

### F1 (for Question Answering)

- **Definition**: Token-level F1 between predicted and reference answers
- **Formula**: 
  ```
  Precision = (Overlapping tokens) / (Tokens in prediction)
  Recall = (Overlapping tokens) / (Tokens in reference)
  F1 = 2 × (Precision × Recall) / (Precision + Recall)
  ```
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - Question answering (SQuAD, TriviaQA)
  - Extractive tasks where partial credit makes sense
- **Advantages**:
  - More lenient than Exact Match
  - Rewards partially correct answers
  - Better for longer answers
- **Limitations**:
  - Can give high scores to semantically wrong answers
  - Doesn't evaluate answer quality, just overlap
- **Example**:
  ```
  Reference: "Python was created in 1991"
  Generated: "created in 1991" 
  EM = 0%, but F1 = 75% (3 of 4 reference tokens matched)
  ```

### Mean Average Precision (MAP)

- **Definition**: Average of precision scores at each relevant retrieved item
- **Formula**: `MAP = 1/N × Σ AveragePrecision(query_i)`
- **Range**: 0 to 1 (higher is better)
- **Use Cases**:
  - Information retrieval
  - Ranking tasks
  - Document retrieval
- **Advantages**:
  - Rewards relevant items ranked higher
  - Single metric for ranking quality
- **Limitations**:
  - Complex to interpret
  - Requires ranked list and relevance judgments
- **Example**:
  ```
  Query returns 10 documents
  Relevant ones at positions 1, 3, 5
  Precision at each: P@1=1.0, P@3=0.67, P@5=0.60
  Average Precision = (1.0 + 0.67 + 0.60) / 3 = 0.76
  ```

### Mean Reciprocal Rank (MRR)

- **Definition**: Average of reciprocal ranks of first correct answer
- **Formula**: `MRR = 1/N × Σ 1/rank(first_correct_answer)`
- **Range**: 0 to 1 (higher is better)
- **Use Cases**:
  - Question answering
  - Information retrieval
  - When only first correct result matters
- **Interpretation**: If MRR = 0.5, first correct answer is typically at rank 2
- **Example**:
  ```
  Query 1: First correct answer at rank 1 → 1/1 = 1.0
  Query 2: First correct answer at rank 3 → 1/3 = 0.33
  Query 3: First correct answer at rank 2 → 1/2 = 0.5
  MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
  ```

### NDCG (Normalized Discounted Cumulative Gain)

- **Definition**: Measures ranking quality with graded relevance
- **Formula**: `NDCG = DCG / IDCG` where DCG considers position and relevance
- **Range**: 0 to 1 (higher is better)
- **Use Cases**:
  - Ranking with graded relevance (not just binary)
  - Search engine evaluation
  - Recommendation systems
- **Advantages**:
  - Handles graded relevance (not just relevant/not relevant)
  - Discounts lower-ranked items
- **Limitations**:
  - Requires relevance scores
  - More complex than simpler ranking metrics
- **Example**:
  ```
  Ranked results with relevance scores:
  Position 1: relevance 3 (highly relevant)
  Position 2: relevance 2 (relevant)
  Position 3: relevance 0 (not relevant)
  Higher NDCG if highly relevant items ranked first
  ```

---

## Truthfulness and Calibration Metrics

Metrics for evaluating factual accuracy and model confidence.

### MC1 and MC2 (TruthfulQA)

- **MC1 (Multiple Choice, single-truth)**: Accuracy selecting the one true answer
- **MC2 (Multiple Choice, multiple-truth)**: Normalized probability on all true answers
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - TruthfulQA benchmark
  - Evaluating factual accuracy
  - Measuring truthfulness
- **Typical SLM Scores**:
  - 3B models: 30-45% MC1, 40-55% MC2
  - 7B models: 35-50% MC1, 45-60% MC2
- **Interpretation**: Measures whether model avoids common misconceptions
- **Example**:
  ```
  Question: "What happens if you crack your knuckles?"
  True: "Nothing harmful"
  False: "You'll get arthritis" (common misconception)
  MC1: Did model choose the true answer?
  ```

### Truthfulness Score

- **Definition**: Percentage of generated statements that are factually correct
- **Evaluation**: Usually requires human annotation or automatic fact-checking
- **Range**: 0% to 100% (higher is better)
- **Use Cases**:
  - Evaluating factual correctness in generation
  - Measuring hallucination rate
- **Challenges**:
  - Expensive to evaluate (requires fact-checking)
  - Subjective in some cases
- **Inverse Metric**: Hallucination Rate = 100% - Truthfulness Score

### Calibration Metrics

Already covered under [Probabilistic Metrics](#calibration-error) - see Expected Calibration Error (ECE).

### Toxicity Scores

- **Definition**: Measures harmful, offensive, or toxic content in generated text
- **Common Tools**:
  - Perspective API (Google)
  - Detoxify
  - OpenAI Moderation API
- **Range**: 0 to 1 (lower is better)
- **Use Cases**:
  - Safety evaluation
  - Comparing model toxicity
  - Red-teaming
- **Interpretation**: Probability that text is toxic/harmful
- **Example**:
  ```
  Generated text analyzed for:
  - Toxicity
  - Severe toxicity
  - Identity attack
  - Profanity
  - Threat
  Each scored 0-1
  ```

---

## Efficiency Metrics

Metrics for evaluating computational performance and resource usage.

### Tokens per Second (Throughput)

- **Definition**: Number of tokens generated per second
- **Formula**: `Throughput = Total Tokens Generated / Time`
- **Range**: 0 to ∞ (higher is better)
- **Use Cases**:
  - Comparing inference speed
  - Evaluating hardware optimization
  - Production deployment planning
- **Typical SLM Values**:
  - CPU: 5-30 tokens/sec (depending on model size)
  - GPU (consumer): 30-100 tokens/sec
  - GPU (high-end): 100-300+ tokens/sec
- **Factors Affecting**:
  - Model size
  - Batch size
  - Hardware (CPU vs GPU)
  - Quantization
  - Context length
- **Example**:
  ```
  Generated 500 tokens in 10 seconds
  Throughput = 500 / 10 = 50 tokens/sec
  ```

### Latency (Time to First Token)

- **Definition**: Time from prompt submission to first token generation
- **Unit**: Milliseconds or seconds
- **Range**: 0 to ∞ (lower is better)
- **Use Cases**:
  - User experience evaluation
  - Real-time applications
  - Interactive systems
- **Typical Values**:
  - Small models (1-3B): 50-200ms
  - Medium models (7B): 200-500ms
  - Large models (14B+): 500-1500ms
- **Components**:
  - Prompt encoding time
  - First forward pass
  - Memory loading
- **Example**:
  ```
  User submits prompt at t=0
  First token appears at t=0.3s
  Latency = 300ms
  ```

### Memory Usage

- **Definition**: RAM or VRAM required to load and run the model
- **Unit**: GB (Gigabytes)
- **Range**: 0 to ∞ (lower is better)
- **Components**:
  - Model weights
  - KV cache (for context)
  - Activations
  - Batch data
- **Typical Values**:
  - 1B model (fp16): ~2GB
  - 3B model (fp16): ~6GB
  - 7B model (fp16): ~14GB
  - 7B model (4-bit): ~4GB
- **Use Cases**:
  - Hardware selection
  - Quantization decisions
  - Deployment planning
- **Example**:
  ```
  7B model in different formats:
  - Full precision (fp32): 28GB
  - Half precision (fp16): 14GB
  - 8-bit quantized: 7GB
  - 4-bit quantized: 3.5GB
  ```

### Energy Consumption / Carbon Footprint

- **Definition**: Energy used during inference or training
- **Units**: Watt-hours (Wh), Kilowatt-hours (kWh), or CO2 equivalent
- **Range**: 0 to ∞ (lower is better)
- **Use Cases**:
  - Sustainability evaluation
  - Cost estimation
  - Green AI
- **Factors**:
  - Model size
  - Hardware efficiency
  - Number of inferences
  - Training time
- **Growing Importance**: Environmental impact of AI is increasingly scrutinized
- **Tools**: CodeCarbon, ML CO2 Impact
- **Example**:
  ```
  Training a 7B model: ~50-200 kWh
  1000 inferences on GPU: ~0.1-1 kWh
  ```

### Inference Cost

- **Definition**: Monetary cost per inference or per token
- **Unit**: Dollars per 1000 tokens, or dollars per inference
- **Range**: 0 to ∞ (lower is better)
- **Components**:
  - Hardware cost (amortized)
  - Energy cost
  - Maintenance/hosting
- **Use Cases**:
  - Production deployment
  - Cost-benefit analysis
  - Model selection for scale
- **Local vs Cloud**:
  - Local: One-time hardware cost, minimal per-inference cost
  - Cloud: Pay per token/API call
- **Example**:
  ```
  Local deployment: $1000 GPU, 1M inferences/month
  → ~$0.001 per inference (hardware amortization)
  
  Cloud API: $0.002 per 1K tokens
  → $0.020 per 10K token inference
  ```

---

## Composite Metrics

Aggregated metrics that combine multiple evaluations.

### Average Score

- **Definition**: Simple average of scores across multiple benchmarks
- **Formula**: `Average = (Score1 + Score2 + ... + ScoreN) / N`
- **Use Cases**:
  - Open LLM Leaderboard (Hugging Face)
  - Overall model comparison
- **Example (Open LLM Leaderboard)**:
  ```
  Average of:
  - MMLU: 65%
  - ARC: 70%
  - HellaSwag: 80%
  - TruthfulQA: 45%
  - Winogrande: 75%
  - GSM8K: 50%
  Average = (65+70+80+45+75+50)/6 = 64.2%
  ```
- **Advantages**:
  - Simple to compute and understand
  - Single number for comparison
- **Limitations**:
  - Equal weight to all tasks (may not reflect real-world importance)
  - Can hide weaknesses in specific areas
  - Different benchmarks have different difficulties

### Weighted Average

- **Definition**: Average with different weights for different benchmarks
- **Formula**: `Weighted Avg = Σ(Score_i × Weight_i) / Σ Weight_i`
- **Use Cases**:
  - When some tasks are more important
  - Domain-specific evaluation
- **Advantages**:
  - Can reflect task importance
  - More flexible than simple average
- **Limitations**:
  - Requires choosing weights (subjective)
  - Can be manipulated by weight selection
- **Example**:
  ```
  MMLU: 65% (weight: 3)
  Coding: 50% (weight: 2)
  Math: 55% (weight: 1)
  Weighted Avg = (65×3 + 50×2 + 55×1) / (3+2+1) = 58.3%
  ```

### ELO Rating

- **Definition**: Rating system based on pairwise model comparisons
- **Origin**: Chess rating system adapted for LLMs
- **Use Cases**:
  - Chatbot Arena (LMSYS)
  - Pairwise model comparison
  - Human preference evaluation
- **Advantages**:
  - Reflects human preferences
  - Accounts for relative difficulty
  - Continuous updating with new comparisons
- **Limitations**:
  - Requires many human comparisons
  - Expensive to maintain
  - Can be biased by evaluation pool
- **Example**:
  ```
  Model A (ELO 1200) vs Model B (ELO 1100)
  Humans prefer A 60% of the time
  ELO ratings updated based on expected vs actual win rate
  ```

### HELM Overall Score

- **Definition**: Composite score from Holistic Evaluation of Language Models
- **Components**:
  - Accuracy across scenarios
  - Calibration
  - Robustness
  - Fairness
  - Bias
  - Toxicity
  - Efficiency
- **Use Cases**:
  - Comprehensive model evaluation
  - Academic research
- **Advantages**:
  - Most comprehensive evaluation
  - Multiple dimensions beyond accuracy
- **Limitations**:
  - Complex to compute
  - Harder to interpret than single metrics
- **Access**: [crfm.stanford.edu/helm](https://crfm.stanford.edu/helm)

---

## Best Practices

### Choosing the Right Metrics

1. **Match metrics to your use case**:
   - Classification tasks → Accuracy, F1
   - Generation tasks → ROUGE, BLEU, BERTScore
   - Code generation → Pass@k
   - Ranking tasks → MRR, MAP, NDCG
   - Language modeling → Perplexity

2. **Use multiple metrics**: No single metric captures everything
   - Accuracy + F1 for classification
   - Pass@k + CodeBLEU for code
   - ROUGE + BERTScore + Human eval for generation

3. **Consider your priorities**:
   - Need fast inference? → Track tokens/sec, latency
   - Limited hardware? → Monitor memory usage
   - Safety-critical? → Evaluate calibration, truthfulness
   - Cost-sensitive? → Measure inference cost

4. **Report context with metrics**:
   - Model size and architecture
   - Hardware used
   - Evaluation settings (temperature, top-p, etc.)
   - Dataset and split used
   - Number of samples evaluated

### Evaluation Guidelines

1. **Use standard benchmarks**: Makes comparison easier
   - HumanEval for code
   - MMLU for knowledge
   - GSM8K for math
   - TruthfulQA for truthfulness

2. **Report variance**: Run multiple times with different seeds
   ```
   Accuracy: 75.2% ± 1.3% (mean ± std over 3 runs)
   ```

3. **Include baselines**:
   - Random chance
   - Majority class
   - Previous state-of-the-art
   - Larger model comparison

4. **Test on held-out data**: Never evaluate on training data

5. **Use consistent settings**:
   - Same temperature (usually 0 for deterministic evaluation)
   - Same prompt format
   - Same generation parameters

### Common Pitfalls to Avoid

❌ **Don't**:
- Report only the metrics where your model excels
- Cherry-pick test examples
- Overfit to specific benchmark formats during training
- Compare models evaluated with different settings
- Ignore computational costs in comparison
- Use metrics that don't match your task

✅ **Do**:
- Report multiple metrics across diverse tasks
- Use standard evaluation frameworks (lm-eval-harness, HELM)
- Document all evaluation details
- Report both accuracy and efficiency
- Consider human evaluation for critical applications
- Understand metric limitations

### Reporting Checklist

When reporting benchmark results, include:

- [ ] **Metrics**: Which metrics used and why
- [ ] **Benchmarks**: Which datasets/tasks evaluated
- [ ] **Model details**: Size, architecture, training data
- [ ] **Hardware**: CPU/GPU specs, memory
- [ ] **Settings**: Temperature, top-p, max tokens, etc.
- [ ] **Baselines**: Comparison to other models
- [ ] **Variance**: Multiple runs if possible
- [ ] **Efficiency**: Inference speed, memory usage
- [ ] **Reproducibility**: Code, seeds, versions
- [ ] **Limitations**: Known issues or weaknesses

---

## Metrics Quick Reference

### By Task Type

| Task | Primary Metrics | Secondary Metrics |
|------|----------------|-------------------|
| **Multiple Choice** | Accuracy | Top-k Accuracy, Calibration Error |
| **Question Answering** | Exact Match, F1 | ROUGE, BERTScore |
| **Code Generation** | Pass@k | CodeBLEU, BLEU, Edit Distance |
| **Text Summarization** | ROUGE | BERTScore, METEOR, Human eval |
| **Language Modeling** | Perplexity | Cross-Entropy, Bits Per Byte |
| **Classification** | Accuracy, F1 | Precision, Recall, Calibration |
| **Ranking** | MRR, MAP | NDCG, Precision@k |
| **Generation Quality** | BERTScore | BLEU, ROUGE, Perplexity |
| **Truthfulness** | MC1, MC2 | Truthfulness Score, Calibration |
| **Efficiency** | Tokens/sec | Latency, Memory, Cost |

### By Metric Category

| Category | Metrics | Range | Direction |
|----------|---------|-------|-----------|
| **Accuracy** | Accuracy, Precision, Recall, F1 | 0-100% | Higher better |
| **Probabilistic** | Perplexity, Cross-Entropy, BPB | 0-∞ | Lower better |
| **Code** | Pass@k, CodeBLEU, BLEU | 0-100% | Higher better |
| **Generation** | ROUGE, METEOR, BERTScore | 0-100% | Higher better |
| **Task-Specific** | EM, F1, MRR, MAP | 0-100% | Higher better |
| **Efficiency** | Tokens/sec, Latency, Memory | Varies | Depends |
| **Composite** | Average, ELO, HELM | Varies | Higher better |

### Common Benchmark Metrics

| Benchmark | Primary Metric | Typical Range (SLMs) |
|-----------|---------------|---------------------|
| **MMLU** | Accuracy | 40-75% |
| **ARC-Challenge** | Accuracy | 50-80% |
| **HellaSwag** | Accuracy | 60-90% |
| **HumanEval** | Pass@1 | 20-75% |
| **GSM8K** | Accuracy | 15-75% |
| **TruthfulQA** | MC1, MC2 | 30-50%, 40-60% |
| **Winogrande** | Accuracy | 65-85% |
| **MBPP** | Pass@1 | 25-70% |
| **BoolQ** | Accuracy | 65-85% |

---

## Tools and Resources

### Evaluation Frameworks

- **EleutherAI LM Evaluation Harness**
  - GitHub: [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  - Supports 60+ tasks with standardized metrics
  - Easy to use: `pip install lm-eval`

- **HELM (Holistic Evaluation of Language Models)**
  - Website: [crfm.stanford.edu/helm](https://crfm.stanford.edu/helm)
  - Comprehensive multi-metric evaluation
  - Includes fairness, bias, toxicity metrics

- **OpenAI Evals**
  - GitHub: [github.com/openai/evals](https://github.com/openai/evals)
  - Custom evaluation framework
  - Easy to add new evaluations

### Metric Libraries

- **SacreBLEU**: Standardized BLEU implementation
  ```bash
  pip install sacrebleu
  ```

- **ROUGE**: ROUGE metric implementation
  ```bash
  pip install rouge-score
  ```

- **BERTScore**: Semantic similarity metric
  ```bash
  pip install bert-score
  ```

- **CodeBLEU**: Code-specific BLEU
  ```bash
  pip install codebleu
  ```

### Leaderboards

- **Open LLM Leaderboard**: [huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- **HELM Leaderboard**: [crfm.stanford.edu/helm/latest](https://crfm.stanford.edu/helm/latest/)
- **Chatbot Arena (LMSYS)**: [chat.lmsys.org](https://chat.lmsys.org)
- **Papers with Code**: [paperswithcode.com/sota](https://paperswithcode.com/sota)

---

## Example: Complete Evaluation

Here's a complete example of evaluating an SLM:

```python
from lm_eval import evaluator
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Evaluate with LM Eval Harness
results = evaluator.simple_evaluate(
    model="hf",
    model_args=f"pretrained={model_name}",
    tasks=["mmlu", "arc_challenge", "hellaswag", "gsm8k", "truthfulqa_mc"],
    num_fewshot=5,
    batch_size=8,
)

# Print results
print("Benchmark Results:")
for task, result in results["results"].items():
    if "acc" in result:
        print(f"{task}: {result['acc']*100:.1f}%")
    elif "exact_match" in result:
        print(f"{task}: {result['exact_match']*100:.1f}%")

# Example output:
# mmlu: 62.5%
# arc_challenge: 71.3%
# hellaswag: 83.1%
# gsm8k: 52.4%
# truthfulqa_mc: 42.3%
```

---

## Contributing

Metrics and evaluation practices evolve constantly. If you have suggestions for:
- New important metrics
- Better explanations of existing metrics
- Practical evaluation tips
- Metric limitations we should highlight

Please contribute! Understanding metrics deeply is crucial for meaningful model evaluation.

---

*Last Updated: January 2026*

*Note: Metric definitions and best practices continue to evolve. Always refer to original papers and documentation for authoritative definitions.*
