def GPU_monitor():
    """
    Monitor GPU usage.
    """
    from torch.cuda import (
        memory_allocated,
        memory_reserved,
        max_memory_allocated,
        max_memory_reserved,
    )

    print(f"GPU Memory Allocated: {memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Reserved: {memory_reserved() / 1e9:.2f} GB")
    print(f"Max GPU Memory Allocated: {max_memory_allocated() / 1e9:.2f} GB")
    print(f"Max GPU Memory Reserved: {max_memory_reserved() / 1e9:.2f} GB")


def clear_memory():
    import gc
    import time
    import torch

    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)
    GPU_monitor()


def init_wandb(cfg, training_args):
    import os
    import wandb
    
    os.environ["WANDB_MODE"] = cfg.wandb_mode
    wandb.init(
        project=cfg.project,
        id=cfg.run_id,
        name=cfg.run_name,
        config=training_args,
        mode=cfg.wandb_mode,
    )