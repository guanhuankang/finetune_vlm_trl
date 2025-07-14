import os
import torch
from trl import SFTConfig, SFTTrainer
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from transformers import BitsAndBytesConfig
import wandb
from peft import LoraConfig, get_peft_model
from qwen_vl_utils import process_vision_info
from functools import partial

from dataset import load_psor_dataset
from utils import clear_memory, GPU_monitor
from collate import collate_fn

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    text_input = processor.apply_chat_template(
        sample[:2], tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )
    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    )

    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]

def get_model(cfg):
    model_id = cfg.model_id
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        # quantization_config=bnb_config,
    )
    
    processor = Qwen2VLProcessor.from_pretrained(model_id, use_fast=True)
    
    return model, processor


def train(cfg):
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.output_dir,  # Directory to save the model
        num_train_epochs=cfg.num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=cfg.per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=1,  # Batch size for evaluation
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,  # Steps to accumulate gradients
        gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=cfg.learning_rate,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=cfg.logging_steps,  # Steps interval for logging
        eval_steps=cfg.eval_steps,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=cfg.save_steps,  # Steps interval for saving
        metric_for_best_model="eval_loss",  # Metric to evaluate the best model
        greater_is_better=False,  # Whether higher metric values are better
        load_best_model_at_end=True,  # Load the best model after training
        # Mixed precision and gradient settings
        bf16=True,  # Use bfloat16 precision
        tf32=True,  # Use TensorFloat-32 precision
        max_grad_norm=0.3,  # Maximum norm for gradient clipping
        warmup_ratio=0.03,  # Ratio of total steps for warmup
        # Hub and reporting
        push_to_hub=False,  # Whether to push model to Hugging Face Hub
        report_to="wandb",  # Reporting tool for tracking metrics
        # Gradient checkpointing settings
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input
    )

    training_args.remove_unused_columns = False  # Keep unused columns in dataset

    os.environ["WANDB_MODE"] = cfg.wandb_mode  
    wandb.init(
        project=cfg.project,
        name=cfg.run_name,
        config=training_args,
        mode=cfg.wandb_mode
    )
    
    train_dataset, eval_dataset, test_dataset = load_psor_dataset(cfg=cfg)
    
    model, processor = get_model(cfg=cfg)
    
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=partial(collate_fn, processor=processor),
        peft_config=peft_config,
        processing_class=processor.tokenizer,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

def test(cfg):
    clear_memory()
    
    model, processor = get_model(cfg=cfg)

    adapter_path = cfg.output_dir
    model.load_adapter(adapter_path)
    
    _, eval_dataset, _ = load_psor_dataset()
    
    inputs = eval_dataset[0]
    outputs = generate_text_from_sample(model, processor, inputs)
    print("inputs", inputs[:2])
    print("outputs:", outputs)
    print("labels:", inputs[2::])
    GPU_monitor()

if __name__=="__main__":
    from config import get_config
    cfg = get_config()

    train(cfg=cfg)
    test(cfg=cfg)
