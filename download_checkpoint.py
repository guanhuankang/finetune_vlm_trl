import os
from huggingface_hub import snapshot_download

def download_checkpoint(cfg):
    repo_id = "/".join(cfg.model_id.split("/")[-2::])
    local_dir = cfg.model_id
    if not os.path.isdir(local_dir):
        print(f"Downloading snapshot from {repo_id} to {local_dir}.")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        