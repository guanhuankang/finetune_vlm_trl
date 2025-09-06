from llm_json import json
from utils import BBox

class Generation:
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def try_parses(self, text):
        out = []
        for s in text.split("\n"):
            try:
                rank = int(s.split("]")[0].split("[")[-1])
                category = str(s.split("]")[1].split("[")[-1]).strip()
                bbox = tuple(map(int, s.split("(")[-1].split(")")[0].split(",")))
                out.append({
                    "rank": rank,
                    "category": category,
                    "bbox": BBox({
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3],
                    })
                })
            except:
                continue
        return out

    def generate(self, model, processor, batch):
        trim = lambda input_ids, output_ids: [ out_ids[len(in_ids)::] for in_ids, out_ids in zip(input_ids, output_ids)]
        model_inputs = {
            "input_ids": batch.input_ids.to('cuda'),
            "attention_mask": batch.attention_mask.to('cuda'),
            "pixel_values": batch.pixel_values.to('cuda'),
            "image_grid_thw": batch.image_grid_thw.to("cuda"),
        }
        generated_ids = model.generate(**model_inputs, max_new_tokens=2048)
        generated_ids = trim(model_inputs['input_ids'], generated_ids)
        
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

        outputs = []
        for print_text, text in zip(print_texts, generated_texts):
            outputs.append({
                "generated_text": print_text,
                "results": self.try_parses(text)
            })
        return outputs
    