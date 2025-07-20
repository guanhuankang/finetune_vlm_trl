from transformers import TrainerCallback, AutoProcessor
import wandb
import tqdm
from llm_json import json
from utils import clear_memory
from evaluator import Evaluator


class GenerationEvaluation(TrainerCallback):
    def __init__(self, cfg):
        super().__init__()
        self.evaluator = Evaluator(cfg=cfg)

    def evaluate(self, model, processor, eval_dataloader):
        trim = lambda input_ids, output_ids: [ out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        def parse(s):
            try:
                results = json.loads(s)
                assert isinstance(results["results"], list)
                return results
            except:
                return {"results": []}
        
        self.evaluator.init()
        for index, batch in tqdm.tqdm(enumerate(eval_dataloader)):
            model_inputs = batch.batch_val.to(model.device)

            generated_ids = model.generate(**model_inputs, max_new_tokens=1024, num_beams=4)

            generated_ids = trim(model_inputs.input_ids, generated_ids)
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            
            for text, info in zip(generated_texts, batch.info):
                self.evaluator.update(info | parse(text))

        return self.evaluator.average()
    
    def on_evaluate(self, args, state, control, **kwargs):
        clear_memory()

        if not state.is_local_process_zero:
            return control
        else:
            model = kwargs["model"]
            processor = kwargs["processing_class"]
            eval_dataloader = kwargs["eval_dataloader"]

            log_metrics = self.evaluate(model, processor, eval_dataloader)
            wandb.log(log_metrics)
            print(log_metrics)
            
            return control
