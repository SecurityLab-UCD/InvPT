# type: ignore

import argparse
import hashlib
import os
from collections import defaultdict
from typing import Literal

from datasets import Dataset, Features, Value, load_dataset
from torch.cuda import device_count
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    TrainingArguments,
)

from .common import DEVICE, set_seed
from .dataloader import contra_data_collator, grouped_contra_data_collator
from .model import ContrastiveTrainer

ContraMode = Literal["info_nce", "supcon", "grouped"]


def compute_function_id(code: str) -> int:
    """Deterministic 63-bit hash of the code string (positive, fits int64)."""
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def tokenize(tokenizer, example, max_seq_length=256):
    code_inputs = tokenizer(
        example["code"],
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_special_tokens_mask=True,
    )
    aug_inputs = tokenizer(
        example["transformed"],
        padding="max_length",
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

    for i in range(len(dataset)):
        row = dataset[i]
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
):
    set_seed(seed)

    tokenizer_name = tokenizer_name or model_name
    tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_name)
    config = RobertaConfig.from_pretrained(tokenizer_name)
    # model = RobertaForMaskedLM.from_pretrained(model_name)

    model = RobertaForMaskedLM.from_pretrained(
        model_name if checkpoint is None else checkpoint,
        config=config,
    )  # load weights from stage 1
    model.to(DEVICE)

    features = Features(
        {
            "repo": Value("string"),
            "func_name": Value("string"),
            "language": Value("string"),
            "code": Value("string"),
            "docstring": Value("string"),
            "transformed": Value("string"),
            "aug_type": Value("string"),
        }
    )
    dataset = load_dataset("json", data_files=dataset_path, features=features)["train"]
    dataset = dataset.filter(lambda x: x["transformed"] is not None)

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
        tokenized_datasets = grouped_dataset.shuffle(seed=seed).map(
            lambda example: tokenize_grouped(
                tokenizer, example, max_seq_length, max_num_augs
            ),
            batched=True,
            num_proc=num_proc,
        )
        collator_fn = lambda features: grouped_contra_data_collator(
            mlm_collator, features, max_num_augs
        )
    else:
        tokenized_datasets = dataset.shuffle(seed=seed).map(
            lambda example: tokenize(tokenizer, example, max_seq_length=max_seq_length),
            batched=True,
            num_proc=num_proc,
        )
        collator_fn = lambda features: contra_data_collator(mlm_collator, features)

    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run_name}",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size // device_count(),
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
        dataloader_num_workers=os.cpu_count(),
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator_fn,
        alpha=alpha,
        temperature=temperature,
        contra_mode=contra_mode,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(f"saved_models/{run_name}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/csn_jp.jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="InvarientBERT")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--sample_rate", type=float, default=1.0)
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint file to load model weights from. Use this to resume training from a previous state.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Tokenizer model name. Defaults to --model_name if not specified. Useful when the model only provides weights and reuses the tokenizer from its base model.",
    )
    parser.add_argument(
        "--contra_mode",
        type=str,
        default="info_nce",
        choices=["info_nce", "supcon", "grouped"],
        help="Contrastive loss mode: 'info_nce' (diagonal positives only), "
        "'supcon' (multi-positive by function_id mask), or "
        "'grouped' (grouped multi-key contrast with explicit aug grouping).",
    )
    parser.add_argument(
        "--max_num_augs",
        type=int,
        default=6,
        help="Maximum augmentations per anchor group (only used with --contra_mode grouped).",
    )

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        seed=args.seed,
        run_name=args.run_name,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        resume=args.resume,
        alpha=args.alpha,
        temperature=args.temperature,
        max_seq_length=args.max_seq_length,
        sample_rate=args.sample_rate,
        checkpoint=args.checkpoint,
        tokenizer_name=args.tokenizer_name,
        contra_mode=args.contra_mode,
        max_num_augs=args.max_num_augs,
    )
