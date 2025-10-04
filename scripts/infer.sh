#!/bin/bash

cd /home/ghk/mnt/homespace/finetune_vlm_trl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=1 python main.py \
 --evaluation \
 --pretrained_path output/20250924-235834-d36f\
 --ckp -1 \
 --test_split 0,10 \
 --wandb_mode online 