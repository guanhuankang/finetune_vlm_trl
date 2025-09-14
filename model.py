import torch
from transformers import PreTrainedModel, GenerationMixin
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2VLForConditionalGeneration
from transformers.generation.utils import GenerateOutput
from peft import LoraConfig, get_peft_model
from einops import rearrange

from config import MODEL_TYPE, PSORConfig
from download_checkpoint import download_checkpoint
from mask_decoder.mask_decoder import MaskDecoder

class PSORModel(PreTrainedModel, GenerationMixin):
    config_class = PSORConfig

    def __init__(self, config: PSORConfig):
        super().__init__(config)

        download_checkpoint(config)

        self.config = config

        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id, use_fast=True)

        self.model = self._load_base_model(config)

        self._add_special_tokens()

        self.seg_model = MaskDecoder(config)

    def _add_special_tokens(self):
        tokenizer = self.processor.tokenizer
        special_tokens = [
            "<|mask_start|>",
            "<|mask_end|>",
            "<|prediction_start|>",
            "<|prediction_end|>",
        ] + [f"<|mask_{i}|>" for i in range(32)]
        assert tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        ) == len(special_tokens)
        self.model.resize_token_embeddings(len(tokenizer))
        self.model.config.vocab_size = len(tokenizer)

        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Model config vocab size: {self.model.config.vocab_size}")
        print(
            f"Embedding weight shape: {self.model.get_input_embeddings().weight.shape}"
        )

    def _load_base_model(self, config):
        base_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
        }
        if "Qwen2.5-" in config.base_model_id:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.base_model_id, **base_kwargs
            )
        elif "Qwen2-" in config.base_model_id:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                config.base_model_id, **base_kwargs
            )
        else:
            raise ValueError(f"Unsupported model ID: {config.base_model_id}")

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        # model = get_peft_model(model=model, peft_config=peft_config)
    
        return model
    
    def get_processor(self):
        return self.processor

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        out = self.model.forward(*args, **kwargs)

        # Mask Branch
        prediction_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|prediction_start|>")
        prediction_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|prediction_end|>")
        mask_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|mask_start|>")
        vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        image_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

        batch_size = len(kwargs["input_ids"])
        layer_idx = self.config.base_model_layer_idx_for_mask
        
        out.mask_predictions = []

        for bs in range(batch_size):
            ids = kwargs["input_ids"][bs]
            attn = kwargs["attention_mask"][bs]
            # image_tokens_indices = ((ids == image_pad_id) * attn).nonzero()[:, 0]
            pred_start_index = ((ids == prediction_start_id) * attn).nonzero()[:, 0]
            pred_end_index = ((ids == prediction_end_id) * attn).nonzero()[:, 0]
            if len(pred_start_index) > 0 and len(pred_end_index) > 0:
                attn = torch.zeros_like(attn)
                attn[torch.arange(pred_start_index[0],pred_end_index[0] + 1)] = 1

                image = kwargs["images"][bs]  # PIL.Image (H, W)
                
                masks = kwargs["masks"][bs].unsqueeze(1) if "masks" in kwargs else None  # k,1,H,W

                mask_indices = ((ids == mask_start_id) * attn).nonzero() + torch.arange(
                    self.config.n_mask_tokens + 2
                )[None, :].to(ids.device)

                mask_tokens = out.hidden_states[layer_idx][bs, mask_indices - 1, :]  # k, n, C
                
                # image_features = out.hidden_states[layer_idx][bs, image_tokens_indices - 1, :]
                # image_features = rearrange(image_features, "(k h w) c -> k c h w", \
                #                          k=1, h=kwargs["image_grid_thw"][0, 1]//2)

                mask_branch_out = self.seg_model(mask_tokens, None, masks, image)
                if mask_branch_out["loss"] != None:
                    out.loss += mask_branch_out["loss"] / batch_size * 1.0
                gen_masks = mask_branch_out["masks"]
            else:
                gen_masks = None
            out.mask_predictions.append(gen_masks)
        return out

    def generate(self, *args, **kwargs):
        kwargs.setdefault("output_hidden_states", True)
        kwargs.setdefault("return_dict_in_generate", True)

        gen_out = self.model.generate(*args, **kwargs)

        inputs = kwargs | {
            "input_ids": gen_out.sequences,
            "attention_mask": (gen_out.sequences != -1).long(),
        }

        with torch.no_grad():
            out = self.forward(**inputs)

        gen_out.mask_predictions = out.mask_predictions
        return gen_out

AutoConfig.register(MODEL_TYPE, PSORConfig)
AutoModel.register(PSORConfig, PSORModel)


if __name__=="__main__":
    model = PSORModel(PSORConfig())

    for name, param in model.named_parameters():
    # if "lora" in name:
    #     print(f"✅ LoRA 参数: {name} | requires_grad={param.requires_grad}")
        if param.requires_grad:
            print(f"✅ trainable parameters: {name}")
