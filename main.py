import os
from trl import SFTConfig, SFTTrainer
import wandb
from peft import LoraConfig
from functools import partial

from dataset import load_psor_dataset
from utils import clear_memory, init_wandb
from collate import collate_fn
from callbacks import GenerationEvaluationCallback
from model import PSORModel
from config import PSORConfig

def train(config: PSORConfig):
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=config.sft_output_dir,  # Directory to save the model
        num_train_epochs=config.num_train_epochs,  # Number of training epochs
        per_device_train_batch_size=config.per_device_train_batch_size,  # Batch size for training
        per_device_eval_batch_size=config.per_device_eval_batch_size,  # Batch size for evaluation
        gradient_accumulation_steps=config.gradient_accumulation_steps,  # Steps to accumulate gradients
        gradient_checkpointing=False,  # Enable gradient checkpointing for memory efficiency
        # Optimizer and scheduler settings
        optim="adamw_torch_fused",  # Optimizer type
        learning_rate=config.learning_rate,  # Learning rate for training
        lr_scheduler_type="constant",  # Type of learning rate scheduler
        # Logging and evaluation
        logging_steps=config.logging_steps,  # Steps interval for logging
        eval_steps=config.eval_steps,  # Steps interval for evaluation
        eval_strategy="steps",  # Strategy for evaluation
        save_strategy="steps",  # Strategy for saving the model
        save_steps=config.save_steps,  # Steps interval for saving
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
        dataloader_num_workers=config.num_gpus,  # Optimize data loading
        ddp_find_unused_parameters=False,  # Critical for DDP
        ddp_timeout=1800,  # Prevent timeouts
        # -------------------------
        
        remove_unused_columns=False,
    )

    init_wandb(config, training_args=config)

    config.save_pretrained(training_args.output_dir)

    eval_dataset, _, train_dataset = load_psor_dataset(config=config)

    model = PSORModel(config=config)

    processor = model.get_processor()

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
        # compute_metrics=None,
        callbacks=[] if config.quick_eval else [GenerationEvaluationCallback(config=config)],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)

def test(config):
    from torch.utils.data import DataLoader

    model = PSORModel(config=config)

    if os.path.isdir(config.adapter_path):
        print(f"Loading adapter from {config.adapter_path}")
        model.load_adapter(config.adapter_path)
    else:
        print(
            f"No adapter path is found in {config.adapter_path}. Load pretrained weights."
        )

    processor = model.get_processor()
    
    init_wandb(config, training_args=config)

    _, test_dataset, _ = load_psor_dataset(config=config)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.per_device_eval_batch_size,
        collate_fn=partial(collate_fn, processor=processor),
        shuffle=False,
        drop_last=False,
    )

    gen_eval = GenerationEvaluationCallback(config=config)

    gen_eval.evaluate(model, processor, test_dataloader)

if __name__ == "__main__":
    config = PSORConfig.from_args_and_file()
    print(config)

    if not config.evaluation:
        train(config=config)
    else:
        test(config=config)