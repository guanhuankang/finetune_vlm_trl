import os
import time

from dataset import EvalImageHandler
from collate import collate_fn
from evaluator import Evaluator
from generation import Generation
from visualization import visualize
from model import PSORModel
from config import PSORConfig

if __name__ == "__main__":
    config = PSORConfig.from_args_and_file()
    config.evaluation = True
    
    model = PSORModel(config=config)

    if os.path.isdir(config.adapter_path):
        print(f"Loading adapter from {config.adapter_path}")
        model.load_adapter(config.adapter_path)
    else:
        print(
            f"No adapter path is found in {config.adapter_path}. Load pretrained weights."
        )

    processor = model.get_processor()

    evaluator = Evaluator(config=config)
    generation = Generation(config=config)
    eval_image_handler = EvalImageHandler(config=config)

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

        save_dir = os.path.join(config.sft_output_dir, "tty_demo")
        os.makedirs(save_dir, exist_ok=True)

        name = sample["name"]
        vis["image"].save(os.path.join(save_dir, f"{name}.png"))
