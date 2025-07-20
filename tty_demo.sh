#!/bin/bash

source .venv/bin/activate

python tty_demo.py \
    --val_test_train_split "0,100;0,100;100,200" \
    --wandb_mode "offline"