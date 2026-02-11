# type: ignore

import hashlib
import os
from collections import defaultdict
from functools import partial

from accelerate import PartialState
from datasets import Dataset, Features, Value, load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    TrainingArguments,
)

from ._types import ContraMode
from .common import default_num_proc, set_seed
from .dataloader import contra_data_collator, grouped_contra_data_collator
from .model import ContrastiveTrainer, SplitHeadWrapper


def _get_world_size() -> int:
    """Return world size via Accelerate (handles both distributed and single-process)."""
    return PartialState().num_processes


def compute_function_id(code: str) -> int:
    """Deterministic 63-bit hash of the code string (positive, fits int64)."""
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def tokenize(tokenizer, example, max_seq_length=256):
    code_inputs = tokenizer(
        example["code"],
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )
    aug_inputs = tokenizer(
        example["transformed"],
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )
    result = {
        "code_input_ids": code_inputs["input_ids"],
        "code_attention_mask": code_inputs["attention_mask"],
        "code_special_tokens_mask": code_inputs["special_tokens_mask"],
        "aug_input_ids": aug_inputs["input_ids"],
        "aug_attention_mask": aug_inputs["attention_mask"],
        "aug_special_tokens_mask": aug_inputs["special_tokens_mask"],
    }
    # Compute function_id from original code for SupCon multi-positive masking
    codes = example["code"]
    if isinstance(codes, list):
        result["function_id"] = [compute_function_id(c) for c in codes]
    else:
        result["function_id"] = compute_function_id(codes)
    return result


def regroup_dataset(dataset, max_num_augs: int = 6) -> Dataset:
    """Regroup flat (code, transformed) rows by function_id into grouped records.

    Each output record contains one anchor code and a list of its augmentations,
    padded to ``max_num_augs`` with empty strings. Groups with zero successful
    augmentations are filtered out.

    Returns:
        A ``datasets.Dataset`` with columns: repo, func_name, language, code,
        docstring, transformed_list, aug_type_list, num_augs, function_id.
    """
    groups: dict[int, dict] = defaultdict(
        lambda: {
            "repo": None,
            "func_name": None,
            "language": None,
            "code": None,
            "docstring": None,
            "transformed_list": [],
            "aug_type_list": [],
        }
    )

    # Iterating a HF Dataset is much faster than random indexing (dataset[i]).
    for row in dataset:
        fid = compute_function_id(row["code"])
        g = groups[fid]
        if g["code"] is None:
            g["repo"] = row["repo"]
            g["func_name"] = row["func_name"]
            g["language"] = row["language"]
            g["code"] = row["code"]
            g["docstring"] = row["docstring"]
        g["transformed_list"].append(row["transformed"])
        g["aug_type_list"].append(row["aug_type"])

    # Build columnar dict, pad to max_num_augs
    result: dict[str, list] = {
        "repo": [],
        "func_name": [],
        "language": [],
        "code": [],
        "docstring": [],
        "transformed_list": [],
        "aug_type_list": [],
        "num_augs": [],
        "function_id": [],
    }
    for fid, g in groups.items():
        real_k = len(g["transformed_list"])
        if real_k == 0:
            continue
        truncated = g["transformed_list"][:max_num_augs]
        aug_types = g["aug_type_list"][:max_num_augs]
        num_augs = len(truncated)
        # Pad to max_num_augs
        padded_transforms = truncated + [""] * (max_num_augs - num_augs)
        padded_aug_types = aug_types + [""] * (max_num_augs - num_augs)

        result["repo"].append(g["repo"])
        result["func_name"].append(g["func_name"])
        result["language"].append(g["language"])
        result["code"].append(g["code"])
        result["docstring"].append(g["docstring"])
        result["transformed_list"].append(padded_transforms)
        result["aug_type_list"].append(padded_aug_types)
        result["num_augs"].append(num_augs)
        result["function_id"].append(fid)

    return Dataset.from_dict(result)


def tokenize_grouped(tokenizer, example, max_seq_length=256, max_num_augs=6):
    """Tokenize a grouped example: one anchor + list of augmentations.

    Works with both single examples and batched examples (HF ``.map(batched=True)``).
    """
    code_inputs = tokenizer(
        example["code"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )
    result = {
        "code_input_ids": code_inputs["input_ids"],
        "code_attention_mask": code_inputs["attention_mask"],
        "code_special_tokens_mask": code_inputs["special_tokens_mask"],
        "function_id": example["function_id"],
        "num_augs": example["num_augs"],
    }

    if isinstance(example["code"], list):
        # Batched mode: each element of transformed_list is a list of strings
        all_aug_ids = []
        all_aug_masks = []
        all_aug_special = []
        for i, transforms in enumerate(example["transformed_list"]):
            num = example["num_augs"][i]
            # Only tokenize real augmentations (non-empty)
            real_transforms = transforms[:num]
            if real_transforms:
                aug_inputs = tokenizer(
                    real_transforms,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )
                all_aug_ids.append(aug_inputs["input_ids"])
                all_aug_masks.append(aug_inputs["attention_mask"])
                all_aug_special.append(aug_inputs["special_tokens_mask"])
            else:
                all_aug_ids.append([])
                all_aug_masks.append([])
                all_aug_special.append([])
        result["aug_input_ids_list"] = all_aug_ids
        result["aug_attention_mask_list"] = all_aug_masks
        result["aug_special_tokens_mask_list"] = all_aug_special
    else:
        # Single mode
        num = example["num_augs"]
        real_transforms = example["transformed_list"][:num]
        if real_transforms:
            aug_inputs = tokenizer(
                real_transforms,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )
            result["aug_input_ids_list"] = aug_inputs["input_ids"]
            result["aug_attention_mask_list"] = aug_inputs["attention_mask"]
            result["aug_special_tokens_mask_list"] = aug_inputs["special_tokens_mask"]
        else:
            result["aug_input_ids_list"] = []
            result["aug_attention_mask_list"] = []
            result["aug_special_tokens_mask_list"] = []

    return result


def main(
    dataset_path: str,
    model_name: str,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int,
    num_proc: int,
    seed: int,
    run_name: str,
    learning_rate: float,
    resume: bool,
    alpha: float,
    temperature: float,
    max_seq_length: int,
    sample_rate: float,
    checkpoint: str | None = None,
    tokenizer_name: str | None = None,
    contra_mode: ContraMode = "info_nce",
    max_num_augs: int = 6,
    self_contrast: bool = True,
):
    set_seed(seed)

    # Cap num_proc to a sane default (see modeling.common.default_num_proc()).
    num_proc = min(num_proc, default_num_proc())
    # Each rank would otherwise spawn num_proc workers, quickly oversubscribing
    # CPUs (e.g., 4 ranks × 80 workers = 320).
    world_size = _get_world_size()
    if world_size > 1:
        num_proc = max(1, num_proc // world_size)
    if num_proc > 1:
        # Avoid oversubscription (processes × tokenizer threads).
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer_name = tokenizer_name or model_name
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    config = RobertaConfig.from_pretrained(tokenizer_name)
    # model = RobertaForMaskedLM.from_pretrained(model_name)

    roberta_mlm = RobertaForMaskedLM.from_pretrained(
        model_name if checkpoint is None else checkpoint,
        config=config,
    )  # load weights from stage 1

    # Wrap so the LM head is applied per-chunk (avoids [2B, seq, vocab] OOM).
    # DDP will wrap SplitHeadWrapper, keeping all params in one forward().
    model = SplitHeadWrapper(roberta_mlm)

    features = Features(
        {
            "repo": Value("string"),
            "func_name": Value("string"),
            "language": Value("string"),
            "code": Value("string"),
            "docstring": Value("string"),
            "transformed": Value("string"),
            "aug_type": Value("string"),
            "function_id": Value("int64"),
        }
    )
    dataset = load_dataset("json", data_files=dataset_path, features=features)["train"]
    if self_contrast:
        # For rows without a transformation, use the original code as the
        # augmentation (self-contrast: same code, different MLM masks).
        dataset = dataset.map(
            lambda transformed, code: {
                "transformed": transformed if transformed is not None else code
            },
            input_columns=["transformed", "code"],
            num_proc=num_proc,
        )
    else:
        # Drop rows that have no real augmentation.
        dataset = dataset.filter(
            lambda transformed: transformed is not None and transformed != "",
            input_columns=["transformed"],
            num_proc=num_proc,
        )

    if sample_rate < 1.0:
        dataset = dataset.shuffle(seed=seed).select(
            range(int(len(dataset) * sample_rate))
        )

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    if contra_mode == "grouped":
        # Regroup flat rows by function_id into {code, [aug_1, ..., aug_K]}
        grouped_dataset = regroup_dataset(dataset, max_num_augs=max_num_augs)
        tokenized_datasets = grouped_dataset.map(
            partial(
                tokenize_grouped,
                tokenizer,
                max_seq_length=max_seq_length,
                max_num_augs=max_num_augs,
            ),
            batched=True,
            num_proc=num_proc,
            remove_columns=grouped_dataset.column_names,
        ).shuffle(seed=seed)

        collator_fn = partial(grouped_contra_data_collator, mlm_collator, max_num_augs)
    else:
        tokenized_datasets = dataset.map(
            partial(tokenize, tokenizer, max_seq_length=max_seq_length),
            batched=True,
            num_proc=num_proc,
            remove_columns=dataset.column_names,
        ).shuffle(seed=seed)

        collator_fn = partial(contra_data_collator, mlm_collator)

    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run_name}",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size // _get_world_size(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        warmup_ratio=0.1,
        logging_steps=1000,
        eval_strategy="epoch",
        learning_rate=learning_rate,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run_name,
        save_total_limit=3,
        load_best_model_at_end=True,
        dataloader_num_workers=max(1, (os.cpu_count() or 1) // _get_world_size()),
        save_safetensors=False,  # SplitHeadWrapper has tied weights from RoBERTa (embeddings ↔ lm_head); safetensors rejects shared tensors
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_fn,
        processing_class=tokenizer,
        alpha=alpha,
        temperature=temperature,
        contra_mode=contra_mode,
    )

    trainer.train(resume_from_checkpoint=resume)

    # Save the inner RobertaForMaskedLM so downstream tasks can load it
    # directly with RobertaForMaskedLM.from_pretrained().
    save_path = f"saved_models/{run_name}/final"
    unwrapped = trainer.model
    if hasattr(unwrapped, "module"):  # DDP
        unwrapped = unwrapped.module
    unwrapped.roberta_mlm.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
