import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from functools import partial
import tqdm
from llm_json import json
import wandb
from PIL import Image

from dataset import load_psor_dataset, format_data
from config import get_config
from collate import collate_fn
from callbacks import GenerationEvaluation
from utils import init_wandb
from evaluator import Evaluator
import time

def parse(s):
    try:
        results = json.loads(s)
        assert isinstance(results["results"], list)
        return results
    except:
        return {"results": []}

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(sample)
    
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )

    start_time = time.time()
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens) ## num_beams=1
    print("Generation time:", time.time() - start_time)

    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

if __name__=="__main__":
    cfg = get_config(["--evaluation"])
    
    init_wandb(cfg=cfg, training_args=cfg)

    # Model & Processor
    model_id = cfg.model_id
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id)

    if os.path.isdir(cfg.runs_dir):
        adapter_path = cfg.runs_dir
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found. Load pretrained weights.")

    # func = partial(collate_fn, processor=processor, add_generation_prompt=True)
    evaluator = Evaluator(cfg=cfg)

    while True:
        image_path = input("Image path:")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert('RGB')
        width, height = image.size
        input_image = image.resize((cfg.input_width, cfg.input_height))
        name = image_path.replace("\\", "/").split("/")[-1][0:-4]
        
        sample = format_data({
            "image": input_image,
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "label": "",
        })
        generated_text = generate_text_from_sample(model=model, processor=processor, sample=sample)
        results = parse(generated_text)

        evaluator.init()
        evaluator.update({
            "name": name,
            "width": width,
            "height": height,
            "input_width": cfg.input_width,
            "input_height": cfg.input_height,
            "results": results["results"]
        })
        print(evaluator.average())


