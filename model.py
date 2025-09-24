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
from utils import BBox

class PSORModel(PreTrainedModel, GenerationMixin):
    config_class = PSORConfig

    def __init__(self, config: PSORConfig):
        super().__init__(config)

        download_checkpoint(config)

        self.config = config

        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id, use_fast=True)

        self.model = self._load_base_model(config)

        self.seg_model = MaskDecoder(config)

    @property
    def device(self):
        return self.model.device
    
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

        # peft_config = LoraConfig(
        #     lora_alpha=16,
        #     lora_dropout=0.05,
        #     r=8,
        #     bias="none",
        #     target_modules=["q_proj", "v_proj"],
        #     task_type="CAUSAL_LM",
        # )
        # model = get_peft_model(model=model, peft_config=peft_config)
    
        return model
    
    def get_processor(self):
        return self.processor

    def mask_prediction(self, input_ids, images, batch_masks):
        ## Mask Branch
        mask_predictions = []
        loss = 0.0
        for ids, image, masks in zip(input_ids, images, batch_masks):
            if len(masks) > 0:
                ## k, 1, H, W
                masks = torch.stack([torch.tensor(m) for m in masks], dim=0).unsqueeze(1).to(input_ids)
            else:
                masks = None

            records = self.ids_to_records(ids=ids, width=64, height=64)
            
            if len(records["records"]) <= 0:
                mask_predictions.append(None)
                continue

            seg_out = self.seg_model(records["records"], image=image, masks=masks)

            mask_predictions.append(seg_out["masks"])
            if seg_out["loss"] != None:
                loss += seg_out["loss"] / len(input_ids) * 1.0

        return {
            "loss": loss,
            "mask_predictions": mask_predictions,
        }

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = True
        out = self.model.forward(*args, **kwargs)

        mask_out = self.mask_prediction(
            input_ids=kwargs["input_ids"],
            images=kwargs["images"],
            batch_masks=kwargs["masks"] if "masks" in kwargs else [[]]*len(kwargs["images"]),
        )
        
        out.loss += mask_out["loss"]
        out.mask_predictions = mask_out["mask_predictions"]

        return out

    def generate(self, *args, **kwargs):
        kwargs.setdefault("output_hidden_states", False)
        kwargs.setdefault("return_dict_in_generate", True)
        images = kwargs["images"]
        kwargs.pop("images")

        out = self.model.generate(*args, **kwargs)

        with torch.no_grad():
            mask_out = self.mask_prediction(
                input_ids=out.sequences,
                images=images,
                batch_masks=kwargs["masks"] if "masks" in kwargs else [[]]*len(images),
            )
        out.mask_predictions = mask_out["mask_predictions"]

        return out

    def ids_to_records(self, ids, width=None, height=None):
        vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        prediction_start_idx = ((ids == vision_end_id).nonzero()[-1::, 0].tolist() + [0])[0]

        text = self.processor.decode(
            ids[prediction_start_idx::],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        records = []
        end_of_detection = False
        input_width = self.config.input_width
        input_height = self.config.input_height
        width = width or input_width
        height = height or input_height
        clamp = lambda x: min(1.0, max(0.0, x))

        for s in text.split("\n"):
            end_of_detection = end_of_detection or ("|END|" in s)
            try:
                "|{rank}|{category}|{x1},{y1},{x2},{y2}|"
                rank = int(s.split("|")[1])
                category = str(s.split("|")[2]).strip()
                bbox = tuple(map(int, s.split("|")[3].split(",")))
                records.append({
                    "rank": rank,
                    "category": category,
                    "bbox": BBox({
                        "x1": clamp(bbox[0] / input_width) * width,
                        "y1": clamp(bbox[1] / input_height) * height,
                        "x2": clamp(bbox[2] / input_width) * width,
                        "y2": clamp(bbox[3] / input_height) * height,
                    }),
                })
            except:
                continue
        records.sort(key=lambda x: x["rank"])
        return {
            "records": records,
            "end_of_detection": end_of_detection,
            "output_text": self.processor.decode(
                ids[prediction_start_idx::],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
        }
    
AutoConfig.register(MODEL_TYPE, PSORConfig)
AutoModel.register(PSORConfig, PSORModel)

if __name__=="__main__":
    model = PSORModel(PSORConfig())

    for name, param in model.named_parameters():
    # if "lora" in name:
    #     print(f"✅ LoRA 参数: {name} | requires_grad={param.requires_grad}")
        if param.requires_grad:
            print(f"✅ trainable parameters: {name}")
