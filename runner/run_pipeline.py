import sys
import os
import time
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentence_transformers import SentenceTransformer, util

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

from data.custom_dataset import load_dataset
from models.security_classifier import detect_attack

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_cpp import Llama

# -------------------------
# Load Dataset
# -------------------------

dataset = load_dataset()

# -------------------------
# Load Models
# -------------------------

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading FP16 model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
fp16_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
print("FP16 model loaded")

def load_gguf_model(path):
    print("Loading GGUF model...")
    model = Llama(model_path=path, verbose=False)
    print("GGUF model loaded")
    return model

gguf_model = load_gguf_model("models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")
print("Loading INT8 model...")

try:
    int8_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )
    print("INT8 model loaded")

except Exception as e:
    print("INT8 not supported, using fallback FP16")
    int8_model = fp16_model
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
# Inference Functions
# -------------------------

def generate_fp16(prompt):
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    start = time.time()
    outputs = fp16_model.generate(**inputs, max_new_tokens=50)
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
def generate_int8(prompt):
    formatted_prompt = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    start = time.time()
    outputs = int8_model.generate(**inputs, max_new_tokens=50)
    latency = time.time() - start

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_output(response, formatted_prompt)

    return response, latency
# -------------------------
# Accuracy Function
# -------------------------
import re

def compute_accuracy(response, expected):
    if expected is None:
        return None

    response_lower = response.lower()
    expected_lower = expected.lower()

    # -------------------------
    # NUMERIC MATCH (fix 2+2 issue)
    # -------------------------
    expected_numbers = re.findall(r"\d+", expected_lower)
    response_numbers = re.findall(r"\d+", response_lower)

    if expected_numbers:
        return int(any(num in response_numbers for num in expected_numbers))

    # -------------------------
    # SEMANTIC SIMILARITY
    # -------------------------
    emb1 = embedding_model.encode(response, convert_to_tensor=True)
    emb2 = embedding_model.encode(expected, convert_to_tensor=True)

    similarity = util.cos_sim(emb1, emb2).item()
    print(f"Similarity: {round(similarity, 3)}")
    return int(similarity > 0.6)
# -------------------------
# Pipeline
# -------------------------

def run_pipeline():
    print("\n==============================")
    print("MODEL EVALUATION PIPELINE")
    print("==============================\n")

    results = []

    for row in dataset:
        prompt = row["prompt"]
        prompt_type = row["type"]
        expected = row["expected"]

        print(f"\nPROMPT: {prompt}")
        print(f"TYPE: {prompt_type}")

        # -------------------------
        # Safety Check
        # -------------------------
        is_attack = detect_attack(prompt)

        if is_attack:
            print("🚫 BLOCKED (Unsafe Prompt)")

            results.append({
                "model": "guardrail",
                "prompt": prompt,
                "type": prompt_type,
                "response": "BLOCKED",
                "latency": 0,
                "accuracy": None,
                "blocked": 1
            })
            continue

        # -------------------------
        # Run ALL Models
        # -------------------------

        for model_name in ["fp16", "gguf", "int8"]:

            if model_name == "fp16":
                response, latency = generate_fp16(prompt)
            elif model_name == "int8":
                response, latency = generate_int8(prompt)
            else:
                response, latency = generate_gguf(prompt)

            accuracy = compute_accuracy(response, expected)

            results.append({
                "model": model_name,
                "prompt": prompt,
                "type": prompt_type,
                "response": response,
                "latency": latency,
                "accuracy": accuracy,
                "blocked": 0
            })

            print(f"\nModel: {model_name}")
            print(f"Response: {response}")
            print(f"Latency: {round(latency, 3)}s")
            print(f"Accuracy: {accuracy}")

    return results

# -------------------------
# Run Pipeline
# -------------------------

results = run_pipeline()

# -------------------------
# Save Results
# -------------------------

if not os.path.exists("outputs"):
    os.makedirs("outputs")

file_path = f"outputs/results_{int(time.time())}.csv"

with open(file_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "model",
        "prompt",
        "type",
        "response",
        "latency",
        "accuracy",
        "blocked"
    ])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {file_path}")
# -------------------------
# AGGREGATION
# -------------------------

model_stats = {}

total_attacks = 0
blocked_attacks = 0

for r in results:
    model = r["model"]

    # Track attack blocking
    if r["type"] == "attack":
        total_attacks += 1
        if r["blocked"] == 1:
            blocked_attacks += 1

    # Skip guardrail for model metrics
    if model == "guardrail":
        continue

    if model not in model_stats:
        model_stats[model] = {
            "latency": [],
            "accuracy": []
        }

    if r["latency"] is not None:
        model_stats[model]["latency"].append(r["latency"])

    if r["accuracy"] is not None:
        model_stats[model]["accuracy"].append(r["accuracy"])

# -------------------------
# PRINT SUMMARY
# -------------------------

print("\n==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

for model, stats in model_stats.items():
    avg_latency = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0
    avg_accuracy = sum(stats["accuracy"]) / len(stats["accuracy"]) if stats["accuracy"] else 0

    print(f"\nModel: {model}")
    print(f"Avg Latency: {round(avg_latency, 3)}s")
    print(f"Avg Accuracy: {round(avg_accuracy, 2)}")

# -------------------------
# SAFETY METRIC
# -------------------------

safety_score = blocked_attacks / total_attacks if total_attacks > 0 else 0

print("\n------------------------------")
print(f"Safety Score (Attack Block Rate): {round(safety_score, 2)}")
print("------------------------------")
