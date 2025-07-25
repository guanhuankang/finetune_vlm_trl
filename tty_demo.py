import os
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
import time

from dataset import EvalImageHandler
from config import get_config
from collate import collate_fn
from evaluator import Evaluator
from generation import Generation
from visualization import visualize

if __name__ == "__main__":
    cfg = get_config(["--evaluation"])

    # Model & Processor
    model_id = cfg.model_id
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    processor = Qwen2VLProcessor.from_pretrained(model_id)

    adapter_path = os.path.join(cfg.runs_dir, cfg.run_name, cfg.checkpoint_name)
    if os.path.isdir(adapter_path):
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found. Load pretrained weights.", adapter_path)

    evaluator = Evaluator(cfg=cfg)
    generation = Generation(cfg=cfg)
    eval_image_handler = EvalImageHandler(cfg=cfg)

    while True:
        image_path = input("Image path:") or "assets/dataset/images/000000386912.jpg"

        sample = eval_image_handler.handle(image_path=image_path)

        batch = collate_fn(samples=[sample], processor=processor)

        start_time = time.time()
        outputs = generation.generate(model=model, processor=processor, batch=batch)
        end_time = time.time()

        output = outputs[0]
        evaluator.init()
        generated_lst = evaluator.update(
            name=sample["name"],
            width=sample["width"],
            height=sample["height"],
            input_width=sample["input_width"],
            input_height=sample["input_height"],
            results=output["results"],
        )

        # visualization
        image = sample["image"]
        vis = {
            "image_path": image_path,
            "image": visualize(image=image, generated_lst=generated_lst),
            "generated_text": output["generated_text"],
            "results": output["results"],
            "evaluation": evaluator.average(),
            "generated_lst": [(k, str(v)) for obj in generated_lst for k, v in obj.items()],
            "runtime": f"{end_time - start_time} sec",
        }

        for k, v in vis.items():
            print(k, v)

        save_dir = os.path.join(cfg.output_dir, cfg.run_name)
        name = sample["name"]
        os.makedirs(save_dir, exist_ok=True)

        vis["image"].save(os.path.join(save_dir, f"tty_demo_output_{name}.png"))
