from transformers import TrainerCallback
import wandb
from tqdm import tqdm
import os

from evaluator import Evaluator
from generation import Generation
from visualization import visualize


class PSORCallback(TrainerCallback):
    def __init__(self, config):
        super().__init__()
        self.evaluator = Evaluator(config=config)
        self.generation = Generation(config=config)
        self.config = config

    def evaluate(self, model, processor, eval_dataloader):
        input_width = self.config.input_width
        input_height = self.config.input_height
        n_image_visualization = self.config.n_image_visualization

        with tqdm(total=len(eval_dataloader), desc="Evaluation") as bar:
            self.evaluator.init()
            for index, batch in enumerate(eval_dataloader):

                outputs = self.generation.generate(
                    model=model, processor=processor, batch=batch
                )

                print(outputs)

                for name, width, height, out in zip(
                    batch.names, batch.widths, batch.heights, outputs
                ):
                    self.evaluator.update(name=name, results=out["results"])

                    image = self.evaluator.get_image(name=name)
                    image = visualize(image=image, generated_lst=out["results"])
                    if index < n_image_visualization:
                        wandb.log({"image_" + name: wandb.Image(image, caption=name)})
                        wandb.log(
                            {
                                "table_"
                                + name: wandb.Table(
                                    columns=["generated_text", "results"],
                                    data=[
                                        [
                                            str(out["generated_text"]),
                                            "\n".join(
                                                [
                                                    ",".join(
                                                        [
                                                            f"{k}:{x[k]}"
                                                            for k in ["rank", "category", "bbox"]
                                                        ] + [
                                                            f"mask:{x['mask'] is None}"
                                                        ]
                                                    )
                                                    for x in out["results"]
                                                ]
                                            ),
                                        ]
                                    ],
                                )
                            }
                        )

                bar.update()

            log_metrics = self.evaluator.average()

            wandb.log(log_metrics)

            print(log_metrics)

            return log_metrics

    def on_evaluate(self, args, state, control, **kwargs):

        if not state.is_local_process_zero:
            return control
        else:
            model = kwargs["model"]
            processor = kwargs["processing_class"]
            eval_dataloader = kwargs["eval_dataloader"]

            try:
                self.evaluate(model, processor, eval_dataloader)
            except Exception as e:
                print(f"Evaluation process error: {e}")
                import traceback
                traceback.print_exc()

            return control

    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.seg_model.save_pretrained(os.path.join(self.config.sft_output_dir, f"checkpoint-{state.global_step}"))
        return super().on_save(args, state, control, **kwargs)