from huggingface_hub import snapshot_download

# model_id = "Qwen/Qwen-VL-Chat"
model_id = "Qwen/Qwen2-VL-7B-Instruct"
local_dir = model_id  # adjust as needed
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
