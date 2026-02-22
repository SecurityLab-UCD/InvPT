# type: ignore

import hashlib
import os
from functools import partial

from accelerate import PartialState
from datasets import Features, Value, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)

from .common import default_num_proc, set_seed
from .dataloader import contra_data_collator
from .model import ContrastiveTrainer, SplitHeadWrapper


def _get_world_size() -> int:
    """Return world size via Accelerate (handles both distributed and single-process)."""
    return PartialState().num_processes


def compute_function_id(code: str) -> int:
    """Deterministic 63-bit hash of the code string (positive, fits int64)."""
    digest = hashlib.sha256(code.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def tokenize(tokenizer, example, max_seq_length=256, include_nl=False):
    if include_nl:
        # Bimodal: [CLS] docstring [SEP] code [EOS]
        code_inputs = tokenizer(
            example["docstring"],
            example["code"],
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
        aug_inputs = tokenizer(
            example["docstring"],
            example["transformed"],
            truncation=True,
            max_length=max_seq_length,
            return_special_tokens_mask=True,
        )
    else:
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
    self_contrast: bool = True,
    model_type: str = "roberta",
    pooling: str = "cls",
    mlm_weight: float = 1.0,
    include_nl: bool = False,
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
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    config = AutoConfig.from_pretrained(tokenizer_name)

    mlm_model = AutoModelForMaskedLM.from_pretrained(
        model_name if checkpoint is None else checkpoint,
        config=config,
    )

    # Wrap so the LM head is applied per-chunk (avoids [2B, seq, vocab] OOM).
    # DDP will wrap SplitHeadWrapper, keeping all params in one forward().
    model = SplitHeadWrapper(mlm_model)

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

    tokenized_datasets = dataset.map(
        partial(
            tokenize,
            tokenizer,
            max_seq_length=max_seq_length,
            include_nl=include_nl,
        ),
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
        mlm_weight=mlm_weight,
        temperature=temperature,
        pooling=pooling,
    )

    trainer.train(resume_from_checkpoint=resume)

    # Save the inner *ForMaskedLM so downstream tasks can load it
    # directly with AutoModelForMaskedLM.from_pretrained().
    save_path = f"saved_models/{run_name}/final"
    unwrapped = trainer.model
    if hasattr(unwrapped, "module"):  # DDP
        unwrapped = unwrapped.module
    unwrapped.mlm_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
