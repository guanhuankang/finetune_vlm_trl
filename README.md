# PSOR


## Installation
We use `uv` to manage the environment.

```shell
uv venv
source .venv/bin/activate
uv pip install torch torchvision trl transformers pycocotools wandb qwen_vl_utils pillow peft llm-json tqdm
```

## Training
Run scripts to train the model and test model.
```shell
source .venv/bin/activate && python main.py
```
We use wandb to record training log.


Evaluation Mode:
```shell
source .venv/bin/activate && python main.py --evaluation --run_id RUN_ID
```
