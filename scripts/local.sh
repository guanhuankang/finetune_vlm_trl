#!/bin/bash

source .venv/bin/activate

python main.py \
    --val_split "0,2" \
    --test_split "0,5" \
    --train_split "10,10" \
    --num_train_epochs 1 \
    --num_gpus 1\
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --eval_steps 5 \
    --save_steps 5 \
    --wandb_mode offline \
    --project PSORLocal \
    --learning_rate 1e-9
    # --quick_eval
    # --evaluation 
    # --run_name run_20250719_230201
    
    
    
    