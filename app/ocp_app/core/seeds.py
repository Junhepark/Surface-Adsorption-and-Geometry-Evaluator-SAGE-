import numpy as np
import random

try:
    import torch
except Exception:  # torch not available?
    torch = None


def fix_all(seed: int = 0) -> None:
    """numpy / random (/ torch) fixing seed."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
