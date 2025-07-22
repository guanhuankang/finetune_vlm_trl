#!/bin/bash

source .venv/bin/activate

python tty_demo.py \
    --run_name run_20250719_230201 \
    --wandb_mode "offline"
