from datasets import load_dataset

def load_qa_dataset(limit=50):
    dataset = load_dataset("truthful_qa", "generation")

    validation = dataset["validation"]

    questions = []

    for i in range(min(limit, len(validation))):
        item = validation[i]

        questions.append({
            "question": item["question"],
            "answer": item["best_answer"]
        })

    return questions
