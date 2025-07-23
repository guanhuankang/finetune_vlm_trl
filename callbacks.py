from transformers import TrainerCallback
import wandb
from tqdm import tqdm

from evaluator import Evaluator
from generation import Generation
from visualization import visualize
from utils import clear_memory


class GenerationEvaluationCallback(TrainerCallback):
    def __init__(self, cfg):
        super().__init__()
        self.evaluator = Evaluator(cfg=cfg)
        self.generation = Generation(cfg=cfg)
        self.cfg = cfg

    def evaluate(self, model, processor, eval_dataloader):
        input_width = self.cfg.input_width
        input_height = self.cfg.input_height

        with tqdm(total=len(eval_dataloader), desc="Evaluation") as bar:
            self.evaluator.init()
            for index, batch in enumerate(eval_dataloader):

                outputs = self.generation.generate(
                    model=model, processor=processor, batch=batch
                )
                for name, width, height, out in zip(
                    batch.names, batch.widths, batch.heights, outputs
                ):
                    generated_lst = self.evaluator.update(
                        name=name,
                        width=width,
                        height=height,
                        input_width=input_width,
                        input_height=input_height,
                        results=out["results"],
                    )

                    image = self.evaluator.get_image(name=name)
                    image = wandb.Image(
                        visualize(image=image, generated_lst=generated_lst),
                        caption=name,
                    )
                    wandb.log({"image_"+name: image})
                    wandb.log(
                        {
                            "table_"+name: wandb.Table(
                                columns=["generated_text", "results"],
                                data=[[str(out["generated_text"]), str(out["results"])]],
                            )
                        }
                    )

                bar.update()

            log_metrics = self.evaluator.average()

            wandb.log(log_metrics)

            print(log_metrics)

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return control
        else:
            model = kwargs["model"]
            processor = kwargs["processing_class"]
            eval_dataloader = kwargs["eval_dataloader"]

            self.evaluate(model, processor, eval_dataloader)

            return control
