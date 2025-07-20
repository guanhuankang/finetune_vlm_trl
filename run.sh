#!/bin/bash

source .venv/bin/activate

python main.py \
    --val_test_train_split "0,10;0,10;10,100" \
    --num_train_epochs 1 \
    --num_gpus 1\
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_steps 50 \
    --save_steps 100 \
    --wandb_mode offline 
    
    
    