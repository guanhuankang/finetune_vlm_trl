import torch
from transformers import PreTrainedModel, GenerationMixin
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration

from config import MODEL_TYPE, PSORConfig
from download_checkpoint import download_checkpoint

class PSORModel(PreTrainedModel, GenerationMixin):
    config_class = PSORConfig

    def __init__(self, config: PSORConfig):
        super().__init__(config)

        download_checkpoint(config)

        self.config = config
        self.model = self._load_base_model(config)
        
    def _load_base_model(self, config):
        base_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if "Qwen2.5-" in config.base_model_id:
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.base_model_id, **base_kwargs
            )
        elif "Qwen2-" in config.base_model_id:
            return Qwen2VLForConditionalGeneration.from_pretrained(
                config.base_model_id, **base_kwargs
            )
        else:
            raise ValueError(f"Unsupported model ID: {config.base_model_id}")

    def get_processor(self):
        return AutoProcessor.from_pretrained(self.config.base_model_id, use_fast=True)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

AutoConfig.register(MODEL_TYPE, PSORConfig)
AutoModel.register(PSORConfig, PSORModel)
