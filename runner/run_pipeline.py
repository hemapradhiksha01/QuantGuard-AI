import sys
import os
import time
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.prompts import jailbreak_prompts, injection_prompts
from data.dataset_loader import load_qa_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

# -------------------------
# Load Dataset
# -------------------------

qa_prompts = load_qa_dataset(limit=50)

# -------------------------
# Global Tracking
# -------------------------

all_results = []

model_usage = {
    "fp16": 0,
    "gguf": 0
}

total_latency_routing = 0
total_requests = 0

# -------------------------
# Load Models
# -------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading FP16 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("FP16 model loaded")

def load_gguf_model(path):
    print("\nLoading GGUF model...")
    model = Llama(model_path=path, verbose=False)
    print("GGUF model loaded")
    return model

gguf_model = load_gguf_model("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

# -------------------------
# Output Cleaning
# -------------------------

def clean_output(response, formatted_prompt):
    response = response.replace(formatted_prompt, "").strip()

    if "User:" in response:
        response = response.split("User:")[0].strip()

    if "Assistant:" in response:
        response = response.split("Assistant:")[0].strip()

    return response.strip()

# -------------------------
# Inference
# -------------------------

def generate_fp16(prompt):
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    start = time.time()
    outputs = model.generate(**inputs, max_new_tokens=50)
    latency = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_output(response, formatted_prompt)

    return response, latency

def generate_gguf(prompt):
    formatted_prompt = f"User: {prompt}\nAssistant:"

    start = time.time()
    output = gguf_model(formatted_prompt, max_tokens=50)
    latency = time.time() - start

    response = output["choices"][0]["text"]
    response = clean_output(response, formatted_prompt)

    return response, latency

# -------------------------
# Improved Prompt Classification
# -------------------------

def classify_prompt(prompt):
    prompt = prompt.lower()

    if any(x in prompt for x in ["ignore previous", "override", "disregard"]):
        return "injection"

    if any(x in prompt for x in ["hack", "bypass", "exploit"]):
        return "jailbreak"

    knowledge_keywords = [
        "why", "how", "what happens", "who said",
        "according to", "scientifically", "purpose",
        "explain", "impact"
    ]

    if any(k in prompt for k in knowledge_keywords):
        return "complex"

    return "simple"

# -------------------------
# Routing Engine
# -------------------------

def route_request(prompt):
    category = classify_prompt(prompt)

    if category == "simple":
        return "gguf"
    elif category == "complex":
        return "fp16"
    elif category in ["jailbreak", "injection"]:
        return "fp16"
    
    return "gguf"

# -------------------------
# Unified Inference
# -------------------------

def generate_response_with_routing(prompt):
    global total_latency_routing, total_requests

    model_choice = route_request(prompt)

    if model_choice == "fp16":
        response, latency = generate_fp16(prompt)
    else:
        response, latency = generate_gguf(prompt)

    model_usage[model_choice] += 1
    total_latency_routing += latency
    total_requests += 1

    return response, latency, model_choice

# -------------------------
# FIXED ACCURACY FUNCTION
# -------------------------

def compute_accuracy(response, expected):
    if expected is None:
        return 1

    response = response.lower()
    expected = expected.lower()

    expected_keywords = expected.split()[:5]

    matches = sum(1 for word in expected_keywords if word in response)

    return int(matches >= 2)

# -------------------------
# Safety Detection
# -------------------------

def detect_jailbreak(output):
    risky_keywords = ["hack", "bypass", "exploit", "attack", "sql injection"]
    return any(k in output.lower() for k in risky_keywords)

def detect_injection(prompt, output):
    prompt_lower = prompt.lower()
    output_lower = output.lower()

    injection_patterns = ["ignore previous", "override", "disregard"]
    prompt_flag = any(p in prompt_lower for p in injection_patterns)

    unsafe_patterns = [
        "password", "email address", "credentials",
        "confidential", "secret", "access", "database"
    ]

    unsafe_output = any(p in output_lower for p in unsafe_patterns)

    prompt_words = set(prompt_lower.split())
    output_words = set(output_lower.split())
    overlap = len(prompt_words.intersection(output_words))

    irrelevant_flag = overlap < 2 and len(output.strip()) > 0

    return prompt_flag and (unsafe_output or irrelevant_flag)

# -------------------------
# Guardrail
# -------------------------

def apply_guardrail(prompt, output):
    if detect_jailbreak(output):
        return "[BLOCKED: Jailbreak detected]"

    if detect_injection(prompt, output):
        return "[BLOCKED: Prompt injection detected]"

    return output

# -------------------------
# Baseline
# -------------------------

def baseline_gguf_latency():
    total = 0
    for item in qa_prompts:
        _, latency = generate_gguf(item["question"])
        total += latency
    return total / len(qa_prompts)

# -------------------------
# Pipeline
# -------------------------

def run_pipeline():
    print("\n==============================")
    print("FINAL ROUTING SYSTEM")
    print("==============================")

    print("\nRunning QA evaluation...\n")

    correct = 0

    for item in qa_prompts:
        question = item["question"]
        expected = item["answer"]

        response, latency, model_used = generate_response_with_routing(question)
        acc = compute_accuracy(response, expected)

        correct += acc

        print("Q:", question)
        print("Model Used:", model_used)
        print("A:", response)
        print("Accuracy:", acc)
        print("Latency:", round(latency, 3))
        print("-" * 50)

    print(f"\nQA Accuracy: {round(correct / len(qa_prompts), 2)}")

# -------------------------
# Run
# -------------------------

run_pipeline()

# -------------------------
# Metrics
# -------------------------

print("\n==============================")
print("ROUTING METRICS")
print("==============================")

avg_latency_routing = total_latency_routing / total_requests
avg_latency_baseline = baseline_gguf_latency()

print(f"Routing Avg Latency: {round(avg_latency_routing, 3)}")
print(f"Baseline GGUF Latency: {round(avg_latency_baseline, 3)}")

print("\nModel Usage:")
for model, count in model_usage.items():
    print(f"{model}: {count}")

latency_improvement = avg_latency_baseline - avg_latency_routing
print(f"\nLatency Difference: {round(latency_improvement, 3)} seconds")
