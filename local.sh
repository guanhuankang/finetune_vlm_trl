#!/bin/bash

source .venv/bin/activate

python main.py \
    --val_split "0,2" \
    --train_split "0,10" \
    --test_split "0,10" \
    --num_train_epochs 1 \
    --num_gpus 1\
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_steps 5 \
    --save_steps 10 \
    --wandb_mode online \
    --project PSORLocal \
    --evaluation 
    # --run_name run_20250719_230201
    
    
    
    