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

        for batch in eval_dataloader:
            model_inputs = batch.batch_val.to(batch.device)
            generated_ids = model.generate(**model_inputs, max_new_tokens=1024)
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print("output_text:", output_text)

            label_text = processor.batch_decode(
                batch.input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print("label_text:", label_text)


        print(len(eval_dataloader))
        return 