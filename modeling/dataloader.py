from dataclasses import dataclass
from enum import Enum

import torch


class AugType(str, Enum):
    LOCALVARRENAMING = "LocalVarRenaming"
    FOR2WHILE = "For2While"
    WHILE2FOR = "While2For"
    PP2ADDASSIGNMENT = "PP2AddAssignment"
    ADDASSIGNMENT2EQUALASSIGNMENT = "AddAssignment2EqualAssignment"
    REVERSEIFELSE = "ReverseIfElse"


@dataclass
class CodeSearchNetExample:
    repo: str
    func_name: str
    language: str
    code: str
    docstring: str
    transformed: str | None = None  # This is added after code transformation
    aug_type: AugType | None = None
    function_id: int | None = None  # Deterministic hash of canonical code string


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

    # Pad to same seq_len (each batch is independently padded to its own max)
    code_seq_len = code_batch["input_ids"].size(1)
    aug_seq_len = aug_batch["input_ids"].size(1)
    if code_seq_len != aug_seq_len:
        pad_token_id = mlm_collator.tokenizer.pad_token_id
        target_len = max(code_seq_len, aug_seq_len)
        if code_seq_len < target_len:
            pad = target_len - code_seq_len
            code_batch["input_ids"] = torch.nn.functional.pad(
                code_batch["input_ids"], (0, pad), value=pad_token_id
            )
            code_batch["attention_mask"] = torch.nn.functional.pad(
                code_batch["attention_mask"], (0, pad), value=0
            )
            code_batch["labels"] = torch.nn.functional.pad(
                code_batch["labels"], (0, pad), value=-100
            )
        else:
            pad = target_len - aug_seq_len
            aug_batch["input_ids"] = torch.nn.functional.pad(
                aug_batch["input_ids"], (0, pad), value=pad_token_id
            )
            aug_batch["attention_mask"] = torch.nn.functional.pad(
                aug_batch["attention_mask"], (0, pad), value=0
            )
            aug_batch["labels"] = torch.nn.functional.pad(
                aug_batch["labels"], (0, pad), value=-100
            )

    # Combine batches
    batch = {
        "code_input_ids": code_batch["input_ids"],
        "code_attention_mask": code_batch["attention_mask"],
        "code_labels": code_batch["labels"],
        "aug_input_ids": aug_batch["input_ids"],
        "aug_attention_mask": aug_batch["attention_mask"],
        "aug_labels": aug_batch["labels"],
    }

    # Pass through function_ids for SupCon loss (when present)
    if "function_id" in features[0]:
        batch["function_id"] = torch.tensor(
            [f["function_id"] for f in features], dtype=torch.long
        )

    return batch
