from transformers import TrainerCallback
import wandb
from tqdm import tqdm
import os
import torch

from evaluator import Evaluator
from generation import Generation
from visualization import visualize


class PSORCallback(TrainerCallback):
    def __init__(self, config):
        super().__init__()
        self.evaluator = Evaluator(config=config)
        self.generation = Generation(config=config)
        self.config = config

        self.data = []
        self.round = 0

    def evaluate(self, model, processor, eval_dataloader, state=None):
        self.round += 1
        global_step = state.global_step if state is not None else self.round

        save_results = []
        wandb_table_data = []

        with tqdm(total=len(eval_dataloader), desc="Evaluation") as bar:
            self.evaluator.init()
            for index, batch in enumerate(eval_dataloader):

                outputs = self.generation.generate(
                    model=model, processor=processor, batch=batch
                )

                for out in outputs:
                    name = out["name"]
                    self.evaluator.update(name=name, results=out["records"])

                    image = self.evaluator.get_image(name=name)
                    image = visualize(image=image, generated_lst=out["records"])

                    for i in range(len(out["records"])):
                        out["records"][i]["mask"] = None
                        out["records"][i]["bbox"] = out["records"][i]["bbox"].todict()
                    save_results.append(out)

                    if index < self.config.n_image_visualization:
                        wandb_table_data.append([wandb.Image(image, caption=name), str(out)])

                bar.update()
            wandb.log({
                f"Table-{global_step}": wandb.Table(columns=["image", "data"], data=wandb_table_data)
            })
            log_metrics = self.evaluator.average()
            print(log_metrics)
            wandb.log(log_metrics)
            
            self.data = save_results

    def on_evaluate(self, args, state, control, **kwargs):

        if not state.is_local_process_zero:
            return control
        else:
            model = kwargs["model"]
            processor = kwargs["processing_class"]
            eval_dataloader = kwargs["eval_dataloader"]

            try:
                self.evaluate(model, processor, eval_dataloader, state=state)
            except Exception as e:
                print(f"Evaluation process error: {e}")
                import traceback
                traceback.print_exc()

            return control

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        
        model.seg_model.save_pretrained(os.path.join(self.config.sft_output_dir, f"checkpoint-{state.global_step}"))

        torch.save(self.data, os.path.join(self.config.sft_output_dir, f"checkpoint-{state.global_step}/evaluation.results.pth"))

        return super().on_save(args, state, control, **kwargs)