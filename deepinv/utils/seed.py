import numpy as np
import torch
import random


def seed_all(seed):
    """Seed all relevant libraries with random processes to ensure reproducibility.

    :param int seed: the given seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
