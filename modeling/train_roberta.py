# type: ignore

from transformers import (
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import fire
from model import ContraBERTTrainer
from dataloader import contra_data_collator

from common import tokenizer, config, DEVICE, set_seed
import os


def tokenize(example):
    code_inputs = tokenizer(
        example["code"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_special_tokens_mask=True,
    )
    aug_inputs = tokenizer(
        example["original_string"],
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
    dataset_path: str = "data/codesearchnet.jsonl",
    batch_size: int = 64,
    num_train_epochs: int = 3,
    num_proc: int = 80,
    seed: int = 0,
    wandb_project: str | None = "PIA",
    run_name: str = "ContraBERT",
    continue_from_released: bool = False,
    contra_type: str = "info_nce",
    resume_from: str | None = None,
):

    set_seed(seed)

    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    model_path = "microsoft/codebert-base" if resume_from is None else resume_from
    model = (
        RobertaForMaskedLM.from_pretrained(model_path)
        if continue_from_released
        else RobertaForMaskedLM(config)  # start pre-training from scratch
    )
    model.to(DEVICE)

    dataset = load_dataset("json", data_files=dataset_path)

    tokenized_datasets = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=num_proc,
    )
    split_dataset = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        logging_steps=1000,
        eval_strategy="steps",
        eval_steps=1000,
        learning_rate=5e-5,
        weight_decay=0.01,
        remove_unused_columns=False,
        report_to="wandb",
        run_name=run_name,
    )

    trainer = ContraBERTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda features: contra_data_collator(mlm_collator, features),
        alpha=0.7,
        contra_type=contra_type,
    )

    trainer.train(resume_from_checkpoint=(resume_from is not None))
    trainer.save_model(f"saved_models/{run_name}/final")


if __name__ == "__main__":
    fire.Fire(main)
