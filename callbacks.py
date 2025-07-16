from transformers import TrainerCallback, AutoProcessor
from utils import clear_memory

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def on_evaluate(self, args, state, control, **kwargs):
        clear_memory()

        if not state.is_local_process_zero:
            return
        
        model = kwargs["model"]
        processor = kwargs["processing_class"]
        eval_dataloader = kwargs["eval_dataloader"]

        # trim = lambda input_ids, output_ids: [ out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        def trim(input_ids, output_ids):
            output_ids[output_ids==-100] = 0
            return output_ids
        
        for batch in eval_dataloader:
            model_inputs = batch.batch_val
            generated_ids = model.generate(**model_inputs.to(model.device), max_new_tokens=4096)
            generated_ids = trim(model_inputs.input_ids, generated_ids)

            generated_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=False
            )
            print("generated_text:", generated_text)

            label_ids = trim(model_inputs.input_ids, batch.labels)
            label_text = processor.batch_decode(
                label_ids,
                skip_special_tokens=False
            )
            print("label_text:", label_text)


        print(len(eval_dataloader))
        return 