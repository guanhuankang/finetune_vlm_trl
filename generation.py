import torch.nn.functional as F
from llm_json import json
from utils import BBox


class Generation:
    def __init__(self, config):
        super().__init__()
        self.config = config

    def try_parses(self, name, width, height, text, masks=None):
        out = []
        input_width = self.config.input_width
        input_height = self.config.input_height

        if masks is not None:
            masks = F.interpolate(masks, size=(
                height, width), mode="bilinear").detach().cpu()

        for s in text.split("\n"):
            try:
                rank = int(s.split("]")[0].split("[")[-1])
                category = str(s.split("]")[1].split("[")[-1]).strip()
                bbox = tuple(
                    map(int, s.split("(")[-1].split(")")[0].split(",")))
                record = {
                    "name": name,
                    "width": width,
                    "height": height,
                    "rank": rank,
                    "category": category,
                    "bbox": BBox({
                        "x1": bbox[0] / input_width * width,
                        "y1": bbox[1] / input_height * height,
                        "x2": bbox[2] / input_width * width,
                        "y2": bbox[3] / input_height * height,
                    }),
                    "mask": None,
                }
                if (masks is not None) and rank >= 0 and rank <= len(masks):
                    record["mask"] = masks[rank-1, 0].numpy()
                out.append(record)
            except:
                continue
        out.sort(key=lambda x: x["rank"])
        return out

    def generate(self, model, processor, batch):
        def trim(input_ids, output_ids): return [
            out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        model_inputs = {
            "input_ids": batch.input_ids.to('cuda'),
            "attention_mask": batch.attention_mask.to('cuda'),
            "pixel_values": batch.pixel_values.to('cuda'),
            "image_grid_thw": batch.image_grid_thw.to("cuda"),
        }
        generated_output = model.generate(**model_inputs, max_new_tokens=2048)
        generated_ids = trim(
            model_inputs['input_ids'], generated_output.sequences)

        print_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        generated_texts = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        outputs = [
            {
                "generated_text": print_text,
                "results": self.try_parses(
                    name=name,
                    width=width,
                    height=height,
                    text=text,
                    masks=masks,
                ),
            }
            for name, width, height, print_text, text, masks in zip(
                batch.names, batch.widths, batch.heights, print_texts, generated_texts, generated_output.mask_predictions
            )
        ]

        return outputs
