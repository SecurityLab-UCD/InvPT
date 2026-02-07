import os
import random

import numpy as np
import torch

MAX_NUM_PROC = 80


def default_num_proc() -> int:
    """Return the default number of parallel workers, capped at available CPUs."""
    return os.cpu_count() or 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
