# PSOR


## Installation
We use `uv` to manage the environment.

```shell
uv venv
source .venv/bin/activate
uv pip install torch torchvision trl transformers pycocotools wandb qwen_vl_utils pillow peft llm-json tqdm
```

## Training
`python main.py`

`CUDA_LAUNCH_BLOCKING=1 python main.py`

## Running log
1. PID=
