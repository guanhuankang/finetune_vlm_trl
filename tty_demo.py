import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader
from functools import partial
import tqdm
from llm_json import json
import wandb

from dataset import load_psor_dataset
from config import get_config
from evaluator import Evaluator
from collate import collate_fn

def parse(s):
    try:
        results = json.loads(s)
        assert isinstance(results["results"], list)
        return results
    except:
        return {"results": []}

def format_data(sample):
    system_message = """You are a Vision Language Model specialized in Salient Object Ranking. Detect all salient objects in the user's image and rank them from the most to least salient. Output results in this strict JSON format: {"results": [{"rank": 1,"category": "object_name", "bbox": {"x1": x1:int, "y1": y1:int, "x2": x2:int, "y2": y2:int}}, ..., {"rank": N, "category": "background","bbox": {"x1": 0, "y1": 0, "x2": width, "y2": height}}]}
    Requirements:
    1. Final entry must be background object with its bounding box covering the full image (x1=0, y1=0, x2=width, y2=height).
    2. Bounding boxes use absolute pixel coordinates (x1,y1 = top-left, x2,y2 = bottom-right).
    3. Images typically contain only a few salient objects, with a maximum limit of 10 per image.
    4. Output must be pure JSON with no additional text."""
    # The maximum number of salient objects per image is limited to 10.
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
                    "text": f"This is the input image with height = {sample['input_height']} and width = {sample['input_width']}."
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["label"]}],
        },
    ]


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(
        sample[0:-1], tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(sample)
    
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens) ## num_beams=1

    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    input_height = int((model_inputs['image_grid_thw'][0][1]*14).cpu())
    input_width = int((model_inputs['image_grid_thw'][0][2]*14).cpu())

    return output_text[0], (input_height, input_width) 

if __name__=="__main__":
    cfg = get_config()
    
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    wandb.init(
        project=cfg.project,
        name="train_"+cfg.run_name,
        config=cfg,
        mode=cfg.wandb_mode,
    )

    # Model & Processor
    model_id = cfg.model_id
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id)

    if os.path.isdir(cfg.output_dir):
        adapter_path = cfg.output_dir
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found. Load pretrained weights.")

    ## Data
    eval_dataset, test_dataset, train_dataset = load_psor_dataset(cfg)

    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=False,
        drop_last=False,
    )
    
    ## Evaluator
    evaluator = Evaluator(cfg=cfg)
    evaluator.init()

    print("Generation Start:::")
    import time

    trim = lambda input_ids, output_ids: [ out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
    for batch in eval_dataloader:
        batch_info = batch.info
        batch.pop("info")
        batch = batch.to("cuda")
        
        start_time = time.time_ns()

        generated_ids = model.generate(**batch, max_new_tokens = 1024, num_beams=1)
        # generated_ids = trim(batch.input_ids, generated_ids)
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        end_time = time.time_ns()
        print("generation time:", (end_time - start_time) / 1e9, generated_texts)
        
        for text, info in zip(generated_texts, batch_info):
            evaluator.update(info | parse(text))
        print(evaluator.average())
