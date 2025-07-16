from transformers import TrainerCallback, AutoProcessor
from copy import deepcopy

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        print(args)
        print(state)
        print(control)
        print(kwargs)
        
        model = kwargs["model"]
        processor = kwargs["processing_class"]
        eval_dataloader = kwargs["eval_dataloader"]

        gens, refs = [], []
        for batch in eval_dataloader:
            print(batch)
            try:
                print("OK", batch.input_ids)
                print("OK", batch.input_ids.shape, batch.labels.shape)
                import pickle
                with open("output/callback.pkl", "wb") as f:
                    pickle.dump(batch, f)
            except:
                pass
            generated_ids = model.generate(**batch, max_new_tokens=1024)
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            print(output_text)


        print(len(len(eval_dataloader)))
        return ""