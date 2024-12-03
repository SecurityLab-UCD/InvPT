# type: ignore

from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset, DatasetDict
import fire


def tokenize(tokenizer, example):
    texts = [
        code + "\n" + doc for code, doc in zip(example["code"], example["docstring"])
    ]
    return tokenizer(texts, truncation=True, max_length=512)


def main(
    dataset_path: str = "data/codesearchnet.jsonl",
    run_name: str = "codebert",
    batch_size: int = 64,
    num_train_epochs: int = 3,
):

    tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
    config = RobertaConfig.from_pretrained("microsoft/codebert-base")
    model = RobertaForMaskedLM(config)

    dataset = load_dataset("json", data_files=dataset_path)

    tokenized_datasets = dataset.map(
        lambda examples: tokenize(tokenizer, examples),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=f"./saved_models/{run_name}",
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=1000,
        logging_steps=1000,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
    )

    trainer.train()
    trainer.save_model(f"saved_models/{run_name}/final")


if __name__ == "__main__":
    fire.Fire(main)
