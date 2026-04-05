# QuantGuard-AI

QuantGuard-AI is a evaluation pipeline for comparing different LLM variants across latency, accuracy, and safety.

---

## Goal

Build a simple system to compare how different model variants perform under:

- Latency (inference speed)
- Accuracy (semantic correctness)
- Safety (handling adversarial prompts)

---

## Models Used

- **FP16 (Hugging Face Transformer)**
  - Baseline model
  - Higher latency
  - Higher accuracy

- **INT8 (Transformers - attempted)**
  - Uses `load_in_8bit=True`
  - Falls back to FP16 on non-CUDA environments

- **GGUF (llama.cpp)**
  - Quantized model
  - Very low latency
  - Lower accuracy in some cases

---

## Pipeline Overview

1. Load dataset (normal + attack prompts)

2. Safety check:
   - Detect prompt injection / unsafe patterns
   - Block execution if unsafe

3. For safe prompts:
   - Run all models:
     - FP16
     - INT8
     - GGUF
   - Capture:
     - Response
     - Latency

4. Accuracy calculation:
   - Numeric match (for numeric answers)
   - Semantic similarity (sentence embeddings)
   - Basic keyword matching

5. Store results:
   - Console output
   - CSV file in `outputs/`

6. Aggregation:
   - Average latency per model
   - Average accuracy per model
   - Safety block rate

---

## Key Takeaways

- FP16 provides more stable and accurate responses but is slower
- GGUF is significantly faster but may produce incorrect or hallucinated answers
- INT8 behaves similar to FP16 in this setup due to fallback on Mac (no CUDA)
- Safety layer successfully blocks adversarial prompts before model execution
- Dataset quality directly impacts evaluation reliability

---

## Setup Instructions

### 1. Clone repository

```bash
git clone https://github.com/hemapradhiksha01/QuantGuard-AI.git
cd QuantGuard-AI
2. Install dependencies
pip install -r requirements.txt
pip install sentence-transformers llama-cpp-python
3. Add GGUF model

Place your .gguf model inside:

models/

Update path in run_pipeline.py if needed.

4. Run pipeline
python3 runner/run_pipeline.py
Output
Per-prompt model responses
Latency and accuracy logs
Final comparison summary
CSV results stored in:
outputs/
Notes
INT8 requires CUDA for real quantization; falls back on Mac
Accuracy is approximate and based on semantic similarity + rules
GGUF models run via llama.cpp for CPU-friendly inference
