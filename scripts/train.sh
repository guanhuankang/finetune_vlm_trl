#!/bin/bash

cd /home/ghk/mnt/homespace/finetune_vlm_trl
source .venv/bin/activate

torchrun --nproc_per_node=2 --master_port=29501 main.py \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --wandb_mode offline \
 --num_gpus 2 \
 --input_width 448 \
 --input_height 448 \
 --base_model_id assets/Qwen/Qwen2.5-VL-3B-Instruct