from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import torch

def resize_tensor(t, h, w):
    t = torch.nn.functional.interpolate(t[None, None, :, :], size=(h, w), mode="bilinear")
    return t[0, 0, :, :]

def collate_fn(samples, processor):
    # chat_contents = [sample["chat_content"] for sample in samples]

    texts = [
        processor.apply_chat_template(
            sample["chat_content"],
            tokenize=False,
            add_generation_prompt=sample["add_generation_prompt"],
        )
        for sample in samples
    ]

    image_inputs = [
        process_vision_info(sample["chat_content"])[0] for sample in samples
    ]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )

    # ---- Pad & Image Tokens ---- #
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    # image_pad_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_tokens = [vision_start_id, vision_end_id, image_pad_id]
    # image_tokens = [151652, 151653, 151655]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    # ---- ------------------ ---- #

    batch["labels"] = labels  # Add labels to the batch

    batch["names"] = [sample["name"] for sample in samples]
    batch["widths"] = [sample["width"] for sample in samples]
    batch["heights"] = [sample["height"] for sample in samples]
    batch["masks"] = [
        torch.stack([resize_tensor(torch.tensor(m), h=480, w=640) for m in sample["masks"]], dim=0)
        for sample in samples
    ]
    batch["images"] = [sample["image"] for sample in samples]

    return batch
