import torch
import numpy as np
import random


def fix(seed):
    # Torch
    torch.manual_seed(seed)

    # CuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # NumPy
    np.random.seed(seed)

    # Python
    random.seed(seed)

    print("\n==============================")
    print("Random seed has been fixed.")
    print(f"Seed: {seed}")
    print("==============================\n")