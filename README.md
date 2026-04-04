# QuantGuard-AI: LLM Routing, Safety, and Quantization Evaluation System

## Overview

This project implements a system for evaluating and managing large language models under real-world constraints. It focuses on three core challenges:

* Model performance (accuracy)
* System efficiency (latency and resource usage)
* Safety under adversarial inputs (jailbreak and prompt injection)

The system dynamically routes requests across multiple model variants and measures trade-offs between quantized and full-precision models using a real benchmark dataset.

---

## Key Features

* Dynamic model routing based on input complexity and risk
* Evaluation on a real dataset (TruthfulQA)
* Comparison of quantized (GGUF) and full-precision (FP16) models
* Guardrail layer to detect and block unsafe outputs
* End-to-end metrics collection (accuracy, latency, model usage)
* CSV-based logging for analysis

---

## System Architecture

The system processes each request through the following stages:

Input → Prompt Classification → Routing Engine → Model Selection → Guardrail → Output

### Components

* **Prompt Classification**

  * Identifies whether a request is simple, complex, or adversarial

* **Routing Engine**

  * Selects the appropriate model based on classification

* **Model Layer**

  * FP16 model for higher-quality responses
  * GGUF quantized model for faster inference

* **Guardrail Layer**

  * Filters unsafe or malicious outputs

* **Evaluation Layer**

  * Computes accuracy, latency, and safety metrics

---

## Models Used

* Full-precision model:

  * TinyLlama 1.1B (FP16 via Hugging Face)

* Quantized model:

  * TinyLlama GGUF (Q4_K_M via llama.cpp)

The quantized model runs significantly faster but may produce less reliable outputs for complex queries.

---

## Dataset

The system uses the TruthfulQA dataset from Hugging Face.

This dataset is designed to evaluate whether models produce truthful answers or repeat common misconceptions.

---

## Evaluation Metrics

### Accuracy

A keyword-based matching approach is used to evaluate correctness.

This avoids strict string matching and better captures semantic correctness in model responses.

---

### Latency

Measured per request:

* GGUF model: ~0.25 seconds
* FP16 model: ~1.7 seconds

---

### Safety

Two types of adversarial behavior are evaluated:

* Jailbreak attempts
* Prompt injection attacks

The guardrail layer detects and blocks unsafe outputs.

---

## Results and Insights

### Accuracy

The system achieved approximately 72% accuracy on a subset of the TruthfulQA dataset.

This demonstrates:

* Small models can answer simple questions correctly
* Complex or misleading questions still lead to incorrect or hallucinated responses

---

### Quantization Trade-offs

* GGUF model provides low latency (~0.25s)
* FP16 model provides higher-quality responses (~1.7s)

Quantization improves performance but reduces factual reliability.

---

### Routing Behavior

Model usage distribution:

* FP16: ~22 requests
* GGUF: ~28 requests

This shows that the system dynamically selects models based on input characteristics.

---

### Latency vs Quality Trade-off

* Routing average latency: ~0.88s
* GGUF-only baseline: ~0.25s

Routing increases latency but improves response quality for complex queries.

---

### Safety Observations

* Both models are vulnerable to adversarial prompts
* Guardrails successfully block unsafe outputs

This highlights the need for safety layers in production systems.

---

### Key Takeaway

This project demonstrates that:

* Quantization improves speed but reduces reliability
* A single model is not sufficient for all tasks
* Intelligent routing is required to balance cost, latency, and quality
* Guardrails are essential for safe deployment of LLM systems

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/hemapradhiksha01/QuantGuard-AI.git
cd QuantGuard-AI
```

---

### 2. Install dependencies

```bash
pip3 install -r requirements.txt
```

---

### 3. Download GGUF model

The model is not included due to size limits.

Download from:

https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF

Place the file inside:

```
models/
```

---

### 4. Run the pipeline

```bash
python3 runner/run_pipeline.py
```

---

## Output

Results are saved in:

```
outputs/results.csv
```

This file includes:

* Prompt
* Model used
* Response
* Accuracy
* Latency
* Safety indicators

---

## Future Improvements

* Add GPTQ / AWQ model variants for deeper quantization comparison
* Use embedding-based similarity for more accurate evaluation
* Introduce confidence-based routing decisions
* Add batching and async inference for scalability
* Extend dataset coverage beyond TruthfulQA

---

## Summary

This project simulates how production LLM systems operate under real constraints by combining:

* Model routing
* Quantization-aware inference
* Safety filtering
* Benchmark-driven evaluation

It highlights the practical trade-offs required to build reliable and efficient AI systems.

