from llm_json import json

class Generation:
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def try_parse_json(self, s):
        try:
            results = json.loads(s)
            assert isinstance(results["results"], list)
            return results
        except:
            return {"results": []}

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
                "results": self.try_parse_json(text)
            })
        return outputs
    