source .venv/bin/activate
uv pip install llm-json

python main.py --val_test_train_split "0,1;0,2;0,10"