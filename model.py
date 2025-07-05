import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

from .utils import *

class VLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        model_id = "Qwen/Qwen2-VL-7B-Instruct"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        processor = Qwen2VLProcessor.from_pretrained(model_id)

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

    def collate_fn(self, examples):
        texts = [
            self.processor.apply_chat_template(example, tokenize=False)
            for example in examples
        ]  # Prepare texts for processing
        image_inputs = [
            process_vision_info(example)[0] for example in examples
        ]  # Process the images to extract inputs

        batch = processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )  # Encode texts and images into tensors

        labels = batch["input_ids"].clone()  # Clone input IDs for labels
        labels[labels == processor.tokenizer.pad_token_id] = (
            -100
        )  # Mask padding tokens in labels

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(
            processor, Qwen2VLProcessor
        ):  # Check if the processor is Qwen2VLProcessor
            image_tokens = [
                151652,
                151653,
                151655,
            ]  # Specific image token IDs for Qwen2VLProcessor
        else:
            image_tokens = [
                processor.tokenizer.convert_tokens_to_ids(processor.image_token)
            ]  # Convert image token to ID

        # Mask image token IDs in the labels
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100  # Mask image token IDs in labels

        batch["labels"] = labels  # Add labels to the batch

        return batch  # Return the prepared batch
    
    def format_data(self, sample):
        system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
        Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
        The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
        Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""
        
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                    {
                        "type": "text",
                        "text": sample["query"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["label"][0]}],
            },
        ]