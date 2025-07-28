import os
from huggingface_hub import snapshot_download

def download_checkpoint(config):
    repo_id = "/".join(config.base_model_id.split("/")[-2::])
    local_dir = config.base_model_id
    if not os.path.isdir(local_dir):
        print(f"Downloading snapshot from {repo_id} to {local_dir}.")
        snapshot_download(repo_id=repo_id, local_dir=local_dir)
        