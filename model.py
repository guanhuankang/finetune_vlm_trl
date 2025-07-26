import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig

from download_checkpoint import download_checkpoint

def get_model(cfg):
    model_id = cfg.model_id

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )

    download_checkpoint(cfg=cfg)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    )

    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    adapter_path = os.path.join(cfg.runs_dir, cfg.run_name, cfg.checkpoint_name)
    if os.path.isdir(adapter_path):
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found in {adapter_path}. Load pretrained weights.")

    return model, processor
