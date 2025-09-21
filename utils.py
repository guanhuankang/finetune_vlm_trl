import math

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


def init_wandb(config, training_args):
    import os
    import wandb
    
    os.environ["WANDB_MODE"] = config.wandb_mode
    wandb.init(
        project=config.project,
        id=config.run_id,
        name=config.run_name,
        config=training_args,
        mode=config.wandb_mode,
    )

class BBox:
    def __init__(self, bbox: dict):
        self.x1 = bbox["x1"]
        self.y1 = bbox["y1"]
        self.x2 = bbox["x2"]
        self.y2 = bbox["y2"]
    
    def toint(self):
        x1 = int(math.floor(self.x1))
        y1 = int(math.floor(self.y1))
        x2 = int(math.ceil(self.x2))
        y2 = int(math.ceil(self.y2))
        return x1, y1, x2, y2

    def intersection(self, bbox):
        x1 = max(self.x1, bbox.x1)
        y1 = max(self.y1, bbox.y1)
        x2 = min(self.x2, bbox.x2)
        y2 = min(self.y2, bbox.y2)
        return BBox({"x1": x1, "y1": y1, "x2": x2, "y2": y2})

    def area(self):
        w = max(self.x2 - self.x1, 0)
        h = max(self.y2 - self.y1, 0)
        return h * w

    def iou(self, bbox):
        s = self.intersection(bbox).area()
        u = self.area() + bbox.area() - s
        return s / u

    def scale(self, r_x, r_y):
        self.x1 = self.x1 * r_x
        self.y1 = self.y1 * r_y
        self.x2 = self.x2 * r_x
        self.y2 = self.y2 * r_y

    def todict(self):
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}
    
    def __str__(self):
        return str(self.todict())
    
    def __repr__(self):
        return self.__str__() 
    