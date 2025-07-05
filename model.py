import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def GPU_monitor():
    """
    Monitor GPU usage.
    """
    import torch
    from torch.cuda import memory_allocated, memory_reserved, max_memory_allocated, max_memory_reserved

    print(f"GPU Memory Allocated: {memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {memory_reserved() / 1e9:.2f} GB")
    print(f"Max GPU Memory Allocated: {max_memory_allocated() / 1e9:.2f} GB")
    print(f"Max GPU Memory Reserved: {max_memory_reserved() / 1e9:.2f} GB")

class VLModel(torch.nn.Module):
    def __init__(self, model_id: str = "Qwen/Qwen2-VL-7B-Instruct"):
        super().__init__()
        if "Qwen" in model_id:
            from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=bnb_config
            )
            processor = Qwen2VLProcessor.from_pretrained(model_id)
        else:
            pass
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        self.model = model
        self.processor = processor
        
    def forward(self, inputs):
        return self.model(**inputs)

    def generate(self, inputs, **kwargs):
        return self.model.generate(**inputs, **kwargs)

