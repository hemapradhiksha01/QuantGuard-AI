# QuantGuard-AI

QuantGuard-AI is a lightweight evaluation framework for analyzing large language models (LLMs) across performance, safety, and deployment configurations. The project focuses on understanding how models behave under normal usage as well as adversarial conditions, and how system-level safeguards can improve reliability.

---

## Overview

This project compares two versions of the same model:

- A standard FP16 model using the Hugging Face Transformers library
- A quantized GGUF model running through llama.cpp

The evaluation pipeline measures:

- Accuracy on simple question-answer tasks
- Latency (response time)
- Model behavior under adversarial prompts
  - Jailbreak attempts
  - Prompt injection attacks
- Effectiveness of a guardrail layer in preventing unsafe outputs

---

## Motivation

In practice, evaluating language models requires more than checking accuracy. Models are often exposed to unpredictable or malicious inputs, and their behavior can change under different deployment settings such as quantization.

This project was built to explore three key questions:

- How reliable is the model under normal usage?
- How does it behave when exposed to adversarial prompts?
- Does quantization affect performance, response quality, or safety?

---

## Project Structure


```
QuantGuard-AI/
│
├── runner/
│   └── run_pipeline.py       # Main evaluation pipeline
│
├── data/
│   └── prompts.py            # QA, jailbreak, and injection prompts
│
├── models/
│   └── *.gguf                # Quantized GGUF model
│
├── outputs/
│   └── results.csv           # Evaluation results
│
├── requirements.txt
└── README.md
```

---

## How It Works

The pipeline runs the same set of prompts on both models and records the results.

### 1. QA Evaluation
The model answers simple questions.  
Accuracy and latency are measured to establish a baseline.

### 2. Jailbreak Testing
Prompts attempt to force the model to generate unsafe outputs (for example, instructions for harmful actions).  
The system checks whether:
- The model produces unsafe content
- The guardrail successfully blocks it

### 3. Prompt Injection Testing
Prompts attempt to override instructions or extract sensitive information.  
The system evaluates:
- Whether the model follows the malicious intent
- Whether the response contains unsafe or irrelevant behavior
- Whether the guardrail intervenes

### 4. Guardrail Layer
A rule-based guardrail filters responses before they are returned.  
This layer simulates how production systems enforce safety constraints on top of model outputs.

---
## Model Setup

The GGUF model is not included in this repository due to file size limits.

Download the model from Hugging Face:

https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-GGUF

Download a file such as:

tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Place the file inside the following folder:

models/

Make sure the final path looks like:

models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
