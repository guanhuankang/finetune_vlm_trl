from transformers import TrainerCallback, AutoProcessor
import wandb
from llm_json import json
from utils import clear_memory

class GenerationEvaluation(TrainerCallback):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model
        self.processor = processor

    def evaluate(self, eval_dataloader):
        trim = lambda input_ids, output_ids: [ out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        for batch in eval_dataloader:
            model_inputs = batch.batch_val.to(self.model.device)
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=1024, num_beams=4)
            
            generated_ids = trim(model_inputs.input_ids, generated_ids)
            generated_texts = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print({"generated_texts": [json.loads(x) for x in generated_texts]})
            

    def on_evaluate(self, args, state, control, **kwargs):
        clear_memory()

        if not state.is_local_process_zero:
            return
        else:
            # model = kwargs["model"]
            # processor = kwargs["processing_class"]
            eval_dataloader = kwargs["eval_dataloader"]

            self.evaluate(eval_dataloader)
