
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info

def collate_fn(examples, processor):
    texts = [
        processor.apply_chat_template(example, tokenize=False,add_generation_prompt=False) for example in examples
    ]
    text_val = [
        processor.apply_chat_template(example[0:-1], tokenize=False,add_generation_prompt=True) for example in examples
    ]  ## remove assistant answer for val purpose
    image_inputs = [process_vision_info(example)[0] for example in examples]

    batch = processor(
        text=texts, images=image_inputs, return_tensors="pt", padding=True
    )
    batch_val = processor(
        text=text_val, images=image_inputs, return_tensors="pt", padding=True
    )
    
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch
    batch["batch_val"] = batch_val

    return batch  # Return the prepared batch
