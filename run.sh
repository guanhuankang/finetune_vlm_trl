#!/bin/bash

source .venv/bin/activate

python main.py \
    --val_test_train_split "0,8;0,16;4,10" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_steps 4 \
    --logging_steps 1 \
    --num_train_epochs 1 \
    
    
    