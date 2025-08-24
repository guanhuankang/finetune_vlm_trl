# PSOR


## Installation
We use `uv` to manage the environment.

```shell
uv venv
source .venv/bin/activate
uv pip install torch torchvision trl transformers pycocotools wandb qwen_vl_utils pillow peft llm-json tqdm einops omegaconf

git clone https://github.com/bytedance/1d-tokenizer.git
mv "1d-tokenizer" bytedance_1d_tokenizer
```

## Training
Run scripts to train the model and test model.
```shell
source .venv/bin/activate && python main.py
```
We use wandb to record training log.


## Evaluation

Evaluation Mode:
```shell
source .venv/bin/activate && python main.py --evaluation --config_file CONFIG_FILE
```

## Inference
```shell
source .venv/bin/activate && python tty_demo.py --run_name run_20250719_230201
```
image_path = `assets/dataset/images/000000386912.jpg`, `assets/dataset/images/000000087038.jpg`
## Tech Pannel

PSORDataset / EvalImageHandler -> collate_fn -> Model (Generation) -> generated_text/results -> Evaluator, Visualization
