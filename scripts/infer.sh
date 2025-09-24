#!/bin/bash

cd /home/ghk/mnt/homespace/finetune_vlm_trl
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 python main.py \
 --evaluation \
 --pretrained_path output/20250924-223454-db7b \
 --test_split 0,2 \