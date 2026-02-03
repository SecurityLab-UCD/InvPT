# type: ignore

import argparse
import os

from datasets import Features, Value, load_dataset
from torch.cuda import device_count
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    TrainingArguments,
)

from .common import DEVICE, set_seed
from .dataloader import contra_data_collator
from .model import ContrastiveTrainer


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
):
    set_seed(seed)

    tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    config = RobertaConfig.from_pretrained(model_name)
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

    tokenized_datasets = dataset.shuffle(seed=seed).map(
        lambda example: tokenize(tokenizer, example, max_seq_length=max_seq_length),
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
        data_collator=lambda features: contra_data_collator(mlm_collator, features),
        alpha=alpha,
        temperature=temperature,
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
    )
