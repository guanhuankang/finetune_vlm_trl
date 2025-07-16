from transformers import TrainerCallback
from copy import deepcopy

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, processor, gen_kwargs=None):
        self.processor = processor
        self.gen_kwargs = gen_kwargs or {"max_new_tokens": 1024, "num_beams": 4}

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        trainer = kwargs["trainer"]
        gens, refs = [], []

        for batch in trainer.get_eval_dataloader():
            import pickle, time
            with open("output/"+str(time.time())+".pkl", "wb") as f:
                pickle.dump(batch, f)
            print(batch)
            
            batch = {k: v.to(trainer.model.device) for k, v in batch.items()}
            gen_ids = trainer.model.generate(**batch, **self.gen_kwargs)
            gens.extend(self.processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True))

            label_ids = batch["labels"]
            refs.extend(self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True))

        metrics = self.compute_metrics_fn({"predictions": gens, "label_ids": refs})
        metrics = {f"eval_gen_{k}": v for k, v in metrics.items()}

        trainer.log(metrics)
        return control
