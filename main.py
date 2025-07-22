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
from utils import clear_memory, GPU_monitor, init_wandb
from collate import collate_fn
from callbacks import GenerationEvaluationCallback

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
        output_dir=os.path.join(cfg.output_dir, cfg.run_name),  # Directory to save the model
        num_train_epochs=cfg.num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=cfg.per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,  # Batch size for evaluation
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
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # Options for gradient checkpointing
        # Dataset configuration
        dataset_text_field="",  # Text field in dataset
        dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
        # max_seq_length=1024  # Maximum sequence length for input

        # --- Multi-GPU Specific ---
        dataloader_num_workers=cfg.num_gpus,  # Optimize data loading
        ddp_find_unused_parameters=False,  # Critical for DDP
        ddp_timeout=1800,  # Prevent timeouts
        # -------------------------
        
        remove_unused_columns=False,
    )

    init_wandb(cfg, training_args=training_args)

    eval_dataset, test_dataset, train_dataset = load_psor_dataset(cfg=cfg)

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
        data_collator=partial(collate_fn, processor=processor, add_generation_prompt=False),
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        # compute_metrics=None,
        callbacks=[] if cfg.quick_eval else [GenerationEvaluationCallback(cfg=cfg)],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


def test(cfg):
    from torch.utils.data import DataLoader

    clear_memory()

    model, processor = get_model(cfg=cfg)

    adapter_path = os.path.join(cfg.runs_dir, cfg.run_name)
    if os.path.isdir(adapter_path):
        model.load_adapter(adapter_path)
        print(f"Load adapter from {adapter_path}")
    else:
        print(f"No adapter path is found. Load pretrained weights.")
    
    if wandb.run == None:
        init_wandb(cfg, training_args=cfg)

    _, test_dataset, _ = load_psor_dataset(cfg=cfg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.per_device_eval_batch_size,
        collate_fn=partial(collate_fn, processor=processor, add_generation_prompt=True),
        shuffle=False,
        drop_last=False,
    )

    gen_eval = GenerationEvaluationCallback(cfg=cfg)

    gen_eval.evaluate(model, processor, test_dataloader)


if __name__ == "__main__":
    from config import get_config

    cfg = get_config()

    if cfg.evaluation:
        test(cfg=cfg)
    else:
        train(cfg=cfg)
