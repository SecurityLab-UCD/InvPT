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


def grouped_contra_data_collator(mlm_collator, max_num_augs, features):
    """Collate grouped samples where each item has 1 anchor + variable-count augmentations.

    Each feature dict contains:
      - code_input_ids, code_attention_mask, code_special_tokens_mask  (anchor)
      - aug_input_ids_list: list of K token-id lists (variable K per sample)
      - aug_attention_mask_list, aug_special_tokens_mask_list: same structure
      - function_id: int

    Returns a batch dict with:
      - code_input_ids: [B, seq_len]
      - code_attention_mask: [B, seq_len]
      - code_labels: [B, seq_len]           (MLM labels for anchors)
      - aug_input_ids: [B * max_K, seq_len] (flattened augmentations)
      - aug_attention_mask: [B * max_K, seq_len]
      - aug_labels: [B * max_K, seq_len]    (MLM labels; padding has -100)
      - group_sizes: [B]                    (real aug count per anchor)
      - function_id: [B]
    """
    # --- Anchor collation (same as existing) ---
    code_features = [
        {
            "input_ids": f["code_input_ids"],
            "attention_mask": f["code_attention_mask"],
            "special_tokens_mask": f["code_special_tokens_mask"],
        }
        for f in features
    ]
    code_batch = mlm_collator(code_features)

    # --- Augmentation collation ---
    seq_len = len(features[0]["code_input_ids"])

    # Determine max_K for this batch (capped by max_num_augs)
    max_K = min(
        max(len(f["aug_input_ids_list"]) for f in features),
        max_num_augs,
    )
    # Ensure at least 1 slot to avoid empty tensors
    max_K = max(max_K, 1)

    group_sizes = []
    aug_features_flat = []

    for f in features:
        aug_ids = f["aug_input_ids_list"][:max_K]
        aug_masks = f["aug_attention_mask_list"][:max_K]
        aug_special = f["aug_special_tokens_mask_list"][:max_K]
        real_K = len(aug_ids)
        group_sizes.append(real_K)

        for k in range(real_K):
            aug_features_flat.append(
                {
                    "input_ids": aug_ids[k],
                    "attention_mask": aug_masks[k],
                    "special_tokens_mask": aug_special[k],
                }
            )

        # Pad remaining slots with zeros (special_tokens_mask=1 → no MLM)
        for _ in range(max_K - real_K):
            aug_features_flat.append(
                {
                    "input_ids": [0] * seq_len,
                    "attention_mask": [0] * seq_len,
                    "special_tokens_mask": [1] * seq_len,
                }
            )

    aug_batch = mlm_collator(aug_features_flat)

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

    batch = {
        "code_input_ids": code_batch["input_ids"],
        "code_attention_mask": code_batch["attention_mask"],
        "code_labels": code_batch["labels"],
        "aug_input_ids": aug_batch["input_ids"],  # [B * max_K, seq_len]
        "aug_attention_mask": aug_batch["attention_mask"],
        "aug_labels": aug_batch["labels"],
        "group_sizes": torch.tensor(group_sizes, dtype=torch.long),
    }

    if "function_id" in features[0]:
        batch["function_id"] = torch.tensor(
            [f["function_id"] for f in features], dtype=torch.long
        )

    return batch
