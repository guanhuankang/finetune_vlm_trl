from datasets import load_dataset

def format_data(sample):
    system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images. Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase. The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text. Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample["query"],
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"][0]}],
        },
    ]

def load_chartqa_dataset():
    dataset_id = "HuggingFaceM4/ChartQA"
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=["train[:10%]", "val[:10%]", "test[:10%]"])
    
    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    return train_dataset, eval_dataset, test_dataset

if __name__ == "__main__":
    train_dataset, eval_dataset, test_dataset = load_chartqa_dataset()
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Example of accessing a sample
    print("Example train sample:", train_dataset[0])  # Print the first training sample
    # print("Example eval sample:", eval_dataset[0])    # Print the first evaluation sample