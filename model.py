import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import BitsAndBytesConfig

def get_model(cfg):
    model_id = cfg.model_id

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id, use_fast=True)

    adapter_path = os.path.join(cfg.runs_dir, cfg.run_name, cfg.checkpoint_name)
    if os.path.isdir(adapter_path):
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found. Load pretrained weights.", adapter_path)

    return model, processor
