# type: ignore

from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    RobertaConfig,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import fire
from model import ContrastiveTrainer
from dataloader import contra_data_collator

from common import DEVICE, set_seed
import os
import argparse
from torch.cuda import device_count


def tokenize(tokenizer, example):
    code_inputs = tokenizer(
        example["code"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True,
    )
    aug_inputs = tokenizer(
        example["code"],  # NOTE: contrasting `code` with itself
        padding="max_length",
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True,
    )
    return {
        "code_input_ids": code_inputs["input_ids"],
        "code_attention_mask": code_inputs["attention_mask"],
        "code_special_tokens_mask": code_inputs["special_tokens_mask"],
        "aug_input_ids": aug_inputs["input_ids"],
        "aug_attention_mask": aug_inputs["attention_mask"],
        "aug_special_tokens_mask": aug_inputs["special_tokens_mask"],
    }


def main(
    dataset_path: str,
    model_name: str,
    batch_size: int,
    max_steps: int,
    gradient_accumulation_steps: int,
    num_proc: int,
    seed: int,
    run_name: str,
    learning_rate: float,
    resume: bool,
):

    set_seed(seed)

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
    model = RobertaForMaskedLM.from_pretrained(model_name)
    model.to(DEVICE)

    dataset = load_dataset("json", data_files=dataset_path)["train"]

    tokenized_datasets = dataset.shuffle(seed=seed).map(
        lambda example: tokenize(tokenizer, example),
        batched=True,
        num_proc=num_proc,
    )
    split_dataset = tokenized_datasets.train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run_name}/stage1",
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size // device_count(),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        save_steps=10000,
        warmup_steps=5000,
        logging_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=learning_rate,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=f"{run_name}_stage1",
        save_total_limit=3,
        load_best_model_at_end=True,
    )

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: contra_data_collator(mlm_collator, features),
        alpha=0.7,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(f"saved_models/{run_name}/stage1/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="data/raw_csn.jsonl")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_proc", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="InvarientBERT")
    parser.add_argument("--max_steps", type=int, default=200_000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=4e-4)
    parser.add_argument("--resume", default=False, action="store_true")

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        seed=args.seed,
        run_name=args.run_name,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        resume=args.resume,
    )
