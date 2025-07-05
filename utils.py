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

