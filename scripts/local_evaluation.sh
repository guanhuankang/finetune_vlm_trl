#!/bin/bash

source .venv/bin/activate

python main.py --evaluation  --project PSORLocal --run_name da5235e7 --checkpoint_name checkpoint-625 --test_split "0,10" --wandb_mode online