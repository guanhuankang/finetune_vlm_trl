#!/bin/bash

source .venv/bin/activate

python tty_demo.py \
    --val_test_train_split "0,2;0,2;100,200" \
    --runs runs/run_20250719_230201 \
    --run_id "checkpoint-2000" \
    --wandb_mode "offline"
