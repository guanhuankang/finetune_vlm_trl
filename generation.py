import torch.nn.functional as F
from transformers import AutoProcessor
import numpy as np
from llm_json import json
from pycocotools.mask import encode as coco_mask_encode
from utils import BBox

class Generation:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = AutoProcessor.from_pretrained(self.config.base_model_id, use_fast=True)

    def generate(self, model, processor, batch):
        def trim(input_ids, output_ids): return [
            out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        model_inputs = {
            "input_ids": batch.input_ids.to('cuda'),
            "attention_mask": batch.attention_mask.to('cuda'),
            "pixel_values": batch.pixel_values.to('cuda'),
            "image_grid_thw": batch.image_grid_thw.to("cuda"),
            "images": batch.images,
        }
        generated_output = model.generate(**model_inputs, max_new_tokens=1024)
        # generated_ids = trim(model_inputs['input_ids'], generated_output.sequences)
        generated_ids = generated_output.sequences
        generated_masks = generated_output.mask_predictions

        outputs = []
        for name, width, height, ids, masks in zip(
            batch.names, 
            batch.widths,
            batch.heights,
            generated_ids,
            generated_masks
        ):
            records = model.ids_to_records(ids, width=width, height=height)
            for i in range(len(records["records"])):
                records["records"][i]["mask"] = (masks[i, 0].detach().cpu().numpy() > 0.5) * 1.0
                records["records"][i]["mask_rle"] = coco_mask_encode(
                    np.asfortranarray(
                        records["records"][i]["mask"].astype(np.uint8)
                ))
                
            outputs.append(
                {"name": name, "width": width, "height": height} | records
            )

        return outputs
