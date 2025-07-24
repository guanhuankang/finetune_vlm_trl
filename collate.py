from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def collate_fn(samples, processor):
    # chat_contents = [sample["chat_content"] for sample in samples]

    texts = [
        processor.apply_chat_template(
            sample["chat_content"], tokenize=False, add_generation_prompt=sample["add_generation_prompt"]
        )
        for sample in samples
    ]

    image_inputs = [process_vision_info(sample["chat_content"])[0] for sample in samples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    # ---- Pad & Image Tokens ---- #
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    # ---- ------------------ ---- #

    batch["labels"] = labels  # Add labels to the batch
    
    batch["names"] = [sample["name"] for sample in samples]
    batch["widths"] = [sample["width"] for sample in samples]
    batch["heights"] = [sample["height"] for sample in samples]

    return batch
