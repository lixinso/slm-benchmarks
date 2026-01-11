# Popular Open-Source Small Language Models (SLMs)

This document lists popular open-source Small Language Models (SLMs) that can be downloaded and run locally on consumer hardware. These models are designed to be efficient and practical for local deployment.

## Table of Contents
- [What are Small Language Models?](#what-are-small-language-models)
- [Ultra-Small Models (< 2B parameters)](#ultra-small-models--2b-parameters)
- [Small Models (2B - 8B parameters)](#small-models-2b---8b-parameters)
- [Medium Models (8B - 15B parameters)](#medium-models-8b---15b-parameters)
- [Hardware Requirements](#hardware-requirements)
- [How to Download and Run](#how-to-download-and-run)

---

## What are Small Language Models?

Small Language Models (SLMs) are compact, efficient language models optimized for local deployment. They offer:
- **Lower resource requirements** - Can run on consumer hardware (laptops, desktops)
- **Faster inference** - Quick response times due to smaller size
- **Privacy** - No data sent to cloud services
- **Cost-effective** - No API fees, one-time download
- **Offline capability** - Work without internet connection

---

## Ultra-Small Models (< 2B parameters)

### TinyLlama 1.1B
- **Organization**: Open source community
- **Parameters**: 1.1B
- **Context Length**: 2K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 2-4GB minimum
  - GPU: Optional (CPU-only works)
  - Storage: ~2GB
- **Key Features**: 
  - Smallest Llama-architecture model
  - Great for learning and experimentation
  - Very fast inference on CPU
- **Download**: 
  - Hugging Face: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
  - Ollama: `ollama pull tinyllama`

### Phi-1.5
- **Organization**: Microsoft Research
- **Parameters**: 1.3B
- **Context Length**: 2K tokens
- **License**: MIT
- **Hardware Requirements**: 
  - RAM: 2-4GB minimum
  - GPU: Optional
  - Storage: ~2.5GB
- **Key Features**: 
  - Strong common sense reasoning for its size
  - Trained on textbook-quality data
  - Excellent code generation
- **Download**: 
  - Hugging Face: `microsoft/phi-1_5`

### Qwen2-0.5B
- **Organization**: Alibaba Cloud
- **Parameters**: 0.5B
- **Context Length**: 32K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 1-2GB minimum
  - GPU: Optional
  - Storage: ~1GB
- **Key Features**: 
  - Extremely lightweight
  - Long context support
  - Multilingual (29 languages)
- **Download**: 
  - Hugging Face: `Qwen/Qwen2-0.5B-Instruct`
  - Ollama: `ollama pull qwen2:0.5b`

### Qwen2-1.5B
- **Organization**: Alibaba Cloud
- **Parameters**: 1.5B
- **Context Length**: 32K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 2-3GB minimum
  - GPU: Optional
  - Storage: ~3GB
- **Key Features**: 
  - Excellent multilingual capabilities
  - Strong coding performance
  - Long context understanding
- **Download**: 
  - Hugging Face: `Qwen/Qwen2-1.5B-Instruct`
  - Ollama: `ollama pull qwen2:1.5b`

### StableLM 2 1.6B
- **Organization**: Stability AI
- **Parameters**: 1.6B
- **Context Length**: 4K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 2-4GB minimum
  - GPU: Optional
  - Storage: ~3GB
- **Key Features**: 
  - Trained on diverse datasets
  - Good general-purpose performance
  - Commercial-friendly license
- **Download**: 
  - Hugging Face: `stabilityai/stablelm-2-1_6b-chat`

---

## Small Models (2B - 8B parameters)

### Phi-2
- **Organization**: Microsoft Research
- **Parameters**: 2.7B
- **Context Length**: 2K tokens
- **License**: MIT
- **Hardware Requirements**: 
  - RAM: 4-6GB minimum
  - GPU: Optional (4GB VRAM recommended)
  - Storage: ~5GB
- **Key Features**: 
  - Outperforms models 25x its size on some benchmarks
  - Excellent reasoning capabilities
  - Strong coding abilities
- **Download**: 
  - Hugging Face: `microsoft/phi-2`
  - Ollama: `ollama pull phi`

### Gemma 2B
- **Organization**: Google DeepMind
- **Parameters**: 2B
- **Context Length**: 8K tokens
- **License**: Gemma License (open weights, commercial use allowed)
- **Hardware Requirements**: 
  - RAM: 4-6GB minimum
  - GPU: Optional (4GB VRAM recommended)
  - Storage: ~4GB
- **Key Features**: 
  - Based on Gemini research
  - Strong instruction following
  - Good multilingual support
- **Download**: 
  - Hugging Face: `google/gemma-2b-it`
  - Ollama: `ollama pull gemma:2b`

### Phi-3-mini (3.8B)
- **Organization**: Microsoft
- **Parameters**: 3.8B
- **Context Length**: 128K tokens
- **License**: MIT
- **Hardware Requirements**: 
  - RAM: 6-8GB minimum
  - GPU: Optional (4-6GB VRAM recommended)
  - Storage: ~7GB
- **Key Features**: 
  - Exceptional long context (128K tokens)
  - State-of-the-art performance for size
  - Optimized for mobile and edge
- **Download**: 
  - Hugging Face: `microsoft/Phi-3-mini-128k-instruct`
  - Ollama: `ollama pull phi3`

### Qwen2-3B
- **Organization**: Alibaba Cloud
- **Parameters**: 3B
- **Context Length**: 32K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 6-8GB minimum
  - GPU: Recommended (4-6GB VRAM)
  - Storage: ~6GB
- **Key Features**: 
  - Excellent multilingual (29 languages)
  - Strong coding capabilities
  - Long context support
- **Download**: 
  - Hugging Face: `Qwen/Qwen2-3B-Instruct`

### StableLM 2 3B
- **Organization**: Stability AI
- **Parameters**: 3B
- **Context Length**: 4K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 6-8GB minimum
  - GPU: Recommended (4GB VRAM)
  - Storage: ~6GB
- **Key Features**: 
  - Balanced performance
  - Commercial-friendly
  - Good for general tasks
- **Download**: 
  - Hugging Face: `stabilityai/stablelm-2-3b-chat`

### Gemma 7B
- **Organization**: Google DeepMind
- **Parameters**: 7B
- **Context Length**: 8K tokens
- **License**: Gemma License (commercial use allowed)
- **Hardware Requirements**: 
  - RAM: 12-16GB minimum
  - GPU: Recommended (8GB VRAM)
  - Storage: ~14GB
- **Key Features**: 
  - Excellent performance for 7B class
  - Based on Gemini technology
  - Strong reasoning and instruction following
- **Download**: 
  - Hugging Face: `google/gemma-7b-it`
  - Ollama: `ollama pull gemma:7b`

### Mistral 7B
- **Organization**: Mistral AI
- **Parameters**: 7B
- **Context Length**: 32K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 12-16GB minimum
  - GPU: Recommended (8GB VRAM)
  - Storage: ~14GB
- **Key Features**: 
  - Best-in-class 7B model
  - Exceptional reasoning
  - Long context support
  - Excellent for fine-tuning
- **Download**: 
  - Hugging Face: `mistralai/Mistral-7B-Instruct-v0.3`
  - Ollama: `ollama pull mistral`

### Qwen2-7B
- **Organization**: Alibaba Cloud
- **Parameters**: 7B
- **Context Length**: 128K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 12-16GB minimum
  - GPU: Recommended (8GB VRAM)
  - Storage: ~14GB
- **Key Features**: 
  - Very long context (128K tokens)
  - Outstanding multilingual support
  - Excellent coding performance
- **Download**: 
  - Hugging Face: `Qwen/Qwen2-7B-Instruct`
  - Ollama: `ollama pull qwen2:7b`

### Llama 3.2 3B
- **Organization**: Meta AI
- **Parameters**: 3B
- **Context Length**: 128K tokens
- **License**: Llama 3.2 Community License
- **Hardware Requirements**: 
  - RAM: 6-8GB minimum
  - GPU: Recommended (4-6GB VRAM)
  - Storage: ~6GB
- **Key Features**: 
  - Latest Llama architecture
  - Long context support (128K)
  - Strong general performance
- **Download**: 
  - Hugging Face: `meta-llama/Llama-3.2-3B-Instruct`
  - Ollama: `ollama pull llama3.2:3b`

---

## Medium Models (8B - 15B parameters)

### Llama 3.1 8B
- **Organization**: Meta AI
- **Parameters**: 8B
- **Context Length**: 128K tokens
- **License**: Llama 3.1 Community License (commercial use allowed)
- **Hardware Requirements**: 
  - RAM: 16GB minimum
  - GPU: Recommended (10GB VRAM)
  - Storage: ~16GB
- **Key Features**: 
  - State-of-the-art 8B model
  - Long context (128K tokens)
  - Multilingual support
  - Tool use capabilities
- **Download**: 
  - Hugging Face: `meta-llama/Meta-Llama-3.1-8B-Instruct`
  - Ollama: `ollama pull llama3.1:8b`

### Gemma 2 9B
- **Organization**: Google DeepMind
- **Parameters**: 9B
- **Context Length**: 8K tokens
- **License**: Gemma License (commercial use allowed)
- **Hardware Requirements**: 
  - RAM: 16-20GB minimum
  - GPU: Recommended (12GB VRAM)
  - Storage: ~18GB
- **Key Features**: 
  - Improved architecture over Gemma 1
  - Excellent performance
  - Knowledge distillation from Gemini
- **Download**: 
  - Hugging Face: `google/gemma-2-9b-it`
  - Ollama: `ollama pull gemma2:9b`

### Phi-3-medium (14B)
- **Organization**: Microsoft
- **Parameters**: 14B
- **Context Length**: 128K tokens
- **License**: MIT
- **Hardware Requirements**: 
  - RAM: 24-32GB minimum
  - GPU: Recommended (16GB VRAM)
  - Storage: ~28GB
- **Key Features**: 
  - Exceptional performance for size
  - Very long context (128K)
  - Advanced reasoning capabilities
- **Download**: 
  - Hugging Face: `microsoft/Phi-3-medium-128k-instruct`
  - Ollama: `ollama pull phi3:medium`

### Qwen2-14B
- **Organization**: Alibaba Cloud
- **Parameters**: 14B
- **Context Length**: 128K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 24-32GB minimum
  - GPU: Recommended (16GB VRAM)
  - Storage: ~28GB
- **Key Features**: 
  - Excellent multilingual capabilities
  - Strong coding and reasoning
  - Long context support
- **Download**: 
  - Hugging Face: `Qwen/Qwen2-14B-Instruct`

### OLMo 7B
- **Organization**: Allen Institute for AI
- **Parameters**: 7B
- **Context Length**: 2K tokens
- **License**: Apache 2.0
- **Hardware Requirements**: 
  - RAM: 12-16GB minimum
  - GPU: Recommended (8GB VRAM)
  - Storage: ~14GB
- **Key Features**: 
  - Fully open (code, data, weights, training details)
  - Transparent research model
  - Good for academic use
- **Download**: 
  - Hugging Face: `allenai/OLMo-7B-Instruct`

---

## Hardware Requirements

### Minimum Requirements by Model Size

| Model Size | RAM | GPU (Optional) | Storage | CPU |
|------------|-----|----------------|---------|-----|
| < 1B | 2GB | None | 2GB | 2 cores |
| 1-2B | 4GB | 2GB VRAM | 4GB | 4 cores |
| 2-4B | 6GB | 4GB VRAM | 8GB | 4 cores |
| 4-8B | 12GB | 8GB VRAM | 16GB | 8 cores |
| 8-15B | 24GB | 16GB VRAM | 32GB | 8+ cores |

### Recommended Hardware for Different Use Cases

**Budget/Learning Setup:**
- CPU: Any modern quad-core
- RAM: 8GB
- GPU: None or integrated
- Models: TinyLlama, Phi-1.5, Qwen2-0.5B

**Mid-Range Setup:**
- CPU: 6-8 core processor
- RAM: 16GB
- GPU: 8GB VRAM (RTX 3060, RTX 4060, etc.)
- Models: Phi-3, Mistral 7B, Gemma 7B, Llama 3.2 3B

**High-End Setup:**
- CPU: 8+ core processor
- RAM: 32GB+
- GPU: 16GB+ VRAM (RTX 4080, RTX 4090, etc.)
- Models: Llama 3.1 8B, Gemma 2 9B, Phi-3 medium, Qwen2-14B

### Quantization Options

Most models support quantization to reduce memory requirements:
- **Q4_0**: 4-bit quantization (~1/4 original size, slight quality loss)
- **Q5_0**: 5-bit quantization (better quality, slightly larger)
- **Q8_0**: 8-bit quantization (minimal quality loss, ~1/2 original size)

Example: Mistral 7B
- Full precision (fp16): ~14GB
- 8-bit (Q8): ~7GB
- 4-bit (Q4): ~3.5GB

---

## How to Download and Run

### Using Ollama (Easiest)

Ollama is the simplest way to run local models:

1. **Install Ollama**: Visit [ollama.com](https://ollama.com) and download for your OS

2. **Download a model**:
```bash
ollama pull tinyllama    # 1.1B model
ollama pull phi3         # 3.8B model
ollama pull mistral      # 7B model
ollama pull llama3.1:8b  # 8B model
```

3. **Run the model**:
```bash
ollama run tinyllama
```

4. **Use in your code**:
```python
import ollama

response = ollama.chat(model='mistral', messages=[
  {'role': 'user', 'content': 'Why is the sky blue?'}
])
print(response['message']['content'])
```

### Using Hugging Face Transformers

For more control and customization:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Use fp16 for efficiency
    device_map="auto"           # Automatically use GPU if available
)

# Generate text
prompt = "What is machine learning?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Using llama.cpp (C++ Implementation)

For maximum performance on CPU:

1. **Install llama.cpp**:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

2. **Download GGUF model**:
Visit Hugging Face and search for GGUF versions (e.g., "Mistral-7B-GGUF")

3. **Run**:
```bash
./main -m models/mistral-7b-q4_0.gguf -p "What is AI?" -n 128
```

### Using text-generation-webui

For a user-friendly web interface:

1. **Install**: Follow instructions at [github.com/oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui)

2. **Download models** through the web interface

3. **Access** via browser at `http://localhost:7860`

### Using LM Studio

GUI application for running models locally:

1. **Download**: Visit [lmstudio.ai](https://lmstudio.ai)
2. **Browse and download models** within the app
3. **Chat** with models through the interface

---

## Model Selection Guide

### By Use Case

**General Chat & Questions:**
- Best: Mistral 7B, Llama 3.1 8B, Phi-3
- Budget: Phi-2, Gemma 2B, TinyLlama

**Coding Assistance:**
- Best: Qwen2-7B, Phi-3, Mistral 7B
- Budget: Phi-2, Qwen2-3B

**Multilingual:**
- Best: Qwen2 series (29 languages)
- Alternative: Gemma, Llama 3.1

**Long Documents/Context:**
- Best: Phi-3 (128K), Llama 3.1/3.2 (128K), Qwen2-7B (128K)
- Alternative: Mistral 7B (32K), Qwen2-3B (32K)

**Extremely Limited Hardware:**
- TinyLlama 1.1B
- Qwen2-0.5B
- Phi-1.5

**Best Overall Balance (Quality vs Size):**
- Mistral 7B
- Phi-3 mini
- Llama 3.1 8B

### By Hardware Available

**4GB RAM, No GPU:**
- TinyLlama, Qwen2-0.5B, Qwen2-1.5B

**8GB RAM, No GPU:**
- Phi-2, Gemma 2B, Qwen2-3B (quantized)

**16GB RAM, 8GB GPU:**
- Mistral 7B, Gemma 7B, Qwen2-7B, Llama 3.1 8B, Phi-3

**32GB RAM, 16GB GPU:**
- Gemma 2 9B, Phi-3 medium, Qwen2-14B

---

## Performance Benchmarks

### MMLU (General Knowledge)

| Model | Parameters | MMLU Score |
|-------|------------|------------|
| Llama 3.1 8B | 8B | 69.4 |
| Gemma 2 9B | 9B | 71.3 |
| Mistral 7B | 7B | 62.5 |
| Qwen2-7B | 7B | 70.3 |
| Phi-3 mini | 3.8B | 69.0 |
| Gemma 7B | 7B | 64.3 |
| Phi-2 | 2.7B | 56.3 |

### HumanEval (Coding)

| Model | Parameters | HumanEval |
|-------|------------|-----------|
| Qwen2-7B | 7B | 64.6 |
| Llama 3.1 8B | 8B | 62.2 |
| Phi-3 mini | 3.8B | 58.8 |
| Mistral 7B | 7B | 40.2 |
| Gemma 2 9B | 9B | 61.0 |

*Note: Scores vary by version and fine-tuning. Always test models for your specific use case.*

---

## Tips for Local Deployment

1. **Start Small**: Begin with TinyLlama or Phi-2 to test your setup
2. **Use Quantization**: 4-bit or 8-bit models save significant RAM/VRAM
3. **GPU Acceleration**: Even a modest GPU dramatically improves speed
4. **Batch Processing**: Process multiple queries together when possible
5. **Context Length**: Longer contexts use more memory - trim when possible
6. **Local Caching**: Models download once and cache locally
7. **Fine-tuning**: Consider fine-tuning smaller models for specific tasks

---

## Resources

- **Ollama**: [ollama.com](https://ollama.com) - Easiest way to run local models
- **Hugging Face**: [huggingface.co/models](https://huggingface.co/models) - Model hub
- **llama.cpp**: [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) - Efficient C++ implementation
- **LM Studio**: [lmstudio.ai](https://lmstudio.ai) - GUI for local models
- **Text Generation WebUI**: [github.com/oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui) - Web interface

---

*Last Updated: December 2025*

*Note: Model capabilities and availability are subject to change. Hardware requirements are estimates and may vary based on quantization and implementation.*
