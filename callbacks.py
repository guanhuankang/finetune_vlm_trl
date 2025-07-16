from transformers import TrainerCallback, AutoProcessor
from copy import deepcopy

class GenerationEvalCallback(TrainerCallback):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_local_process_zero:
            return

        import pickle, time

        print(args)
        print(state)
        print(control)
        print(kwargs)
        
        model = kwargs["model"]
        processor = kwargs["processing_class"]
        eval_dataloader = kwargs["eval_dataloader"]

        gens, refs = [], []

        for batch in eval_dataloader():
            import pickle, time
            with open("output/evaluate_batch_"+str(time.time())+".pkl", "wb") as f:
                pickle.dump(batch, f)
            print(batch)
            