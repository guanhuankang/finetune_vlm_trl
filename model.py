import torch
from transformers import PreTrainedModel, GenerationMixin
from transformers import AutoConfig, AutoModel, AutoProcessor
import os
from transformers import Qwen2_5_VLForConditionalGeneration

from config import MODEL_TYPE, PSORConfig

class PSORModel(PreTrainedModel, GenerationMixin):
    config_class = PSORConfig

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.base_model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            # quantization_config=bnb_config,
        )
        
    def get_processor(self):
        return AutoProcessor.from_pretrained(self.config.base_model_id, use_fast=True)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)

    def _set_gradient_checkpointing(self, *args, **kwargs):
        self.model._set_gradient_checkpointing(*args, **kwargs)


AutoConfig.register(MODEL_TYPE, PSORConfig)
AutoModel.register(PSORConfig, PSORModel)
