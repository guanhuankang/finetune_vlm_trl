#!/bin/bash

cd /home/ghk/mnt/homespace/finetune_vlm_trl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python main.py \
 --per_device_train_batch_size 1 \
 --gradient_accumulation_steps 8 \
 --wandb_mode offline \
 --num_gpus 1 \
 --input_width 336 \
 --input_height 336 \
 --base_model_id assets/Qwen/Qwen2.5-VL-3B-Instruct \
 