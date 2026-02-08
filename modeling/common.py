import os
import random

import numpy as np
import torch


def default_num_proc() -> int:
    """Return the default number of parallel workers.

    HuggingFace Datasets preprocessing (e.g., ``Dataset.map(num_proc=...)``) and
    tokenizers can both parallelize. Using very high ``num_proc`` values can
    oversubscribe CPU and/or hammer the datasets cache on disk, often making
    preprocessing *slower*.
    """
    return os.cpu_count() or 1


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
