from dataclasses import dataclass
from enum import Enum
import fire
import json
import os
from multiprocessing import Pool

from torch.utils.data import DataLoader, IterableDataset
from transformers import DataCollatorForLanguageModeling


@dataclass
class CodeSearchNetExample:
    repo: str
    func_name: str
    original_string: str
    code: str
    language: str
    docstring: str


def contra_data_collator(mlm_collator, features):
    # Separate features for original code and augmented code
    code_features = [
        {
            "input_ids": f["code_input_ids"],
            "attention_mask": f["code_attention_mask"],
            "special_tokens_mask": f["code_special_tokens_mask"],
        }
        for f in features
    ]
    aug_features = [
        {
            "input_ids": f["aug_input_ids"],
            "attention_mask": f["aug_attention_mask"],
            "special_tokens_mask": f["aug_special_tokens_mask"],
        }
        for f in features
    ]
    # Apply MLM collator
    code_batch = mlm_collator(code_features)
    aug_batch = mlm_collator(aug_features)
    # Combine batches
    batch = {
        "code_input_ids": code_batch["input_ids"],
        "code_attention_mask": code_batch["attention_mask"],
        "code_labels": code_batch["labels"],
        "aug_input_ids": aug_batch["input_ids"],
        "aug_attention_mask": aug_batch["attention_mask"],
        "aug_labels": aug_batch["labels"],
    }
    return batch
