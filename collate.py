from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info


def collate_fn(samples, processor):
    chat_contents = [sample["chat_content"] for sample in samples]

    texts = [
        processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=False) for chat in chat_contents
    ]

    text_val = [
        processor.apply_chat_template(chat[0:-1], tokenize=False, add_generation_prompt=True) for chat in chat_contents
    ]  # remove assistant answer for val purpose

    image_inputs = [process_vision_info(chat)[0] for chat in chat_contents]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    batch_val = processor(
        text=text_val, images=image_inputs, return_tensors="pt", padding=True
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    if isinstance(processor, Qwen2VLProcessor):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(
            processor.image_token)]

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100

    batch["labels"] = labels  # Add labels to the batch
    batch["batch_val"] = batch_val

    info_keys = ['name', 'width', 'height', 'input_width', 'input_height']
    batch["info"] = [dict((k, v) for k, v in sample.items() if k in info_keys)
                     for sample in samples]

    return batch  # Return the prepared batch
