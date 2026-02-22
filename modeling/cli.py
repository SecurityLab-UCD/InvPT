from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import typer

from modeling._types import ModelType
from modeling.common import default_num_proc
from modeling.config import load_config
from modeling.pretrain import main

app = typer.Typer(
    help="InvPT: Invariant Pre-training for Robust Code Representation Learning."
)


@app.command()
def run(
    config: Annotated[
        Path,
        typer.Argument(help="Path to a YAML experiment config file."),
    ],
    sample_rate: Annotated[
        Optional[float],
        typer.Option("--sample-rate", "-sr", help="Override sample rate from config."),
    ] = None,
    self_contrast: Annotated[
        Optional[bool],
        typer.Option(
            help="Override self-contrast setting from config. Use --no-self-contrast to disable.",
        ),
    ] = None,
) -> None:
    """Run pre-training from a YAML config file.

    All parameters are read from the config file to ensure full reproducibility.
    Use --sample-rate / --no-self-contrast to override specific settings.

    Example: python -m modeling run experiments/base.yaml --no-self-contrast
    """
    cfg = load_config(config)
    kwargs = asdict(cfg)
    if sample_rate is not None:
        kwargs["sample_rate"] = sample_rate
    if self_contrast is not None:
        kwargs["self_contrast"] = self_contrast
    main(**kwargs)


@app.command("run-all")
def run_all(
    config_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing YAML experiment config files."),
    ],
    sample_rate: Annotated[
        Optional[float],
        typer.Option("--sample-rate", "-sr", help="Override sample rate from config."),
    ] = None,
    self_contrast: Annotated[
        Optional[bool],
        typer.Option(
            help="Override self-contrast setting from config. Use --no-self-contrast to disable.",
        ),
    ] = None,
) -> None:
    """Run pre-training for every YAML config in a directory, sequentially.

    Example: python -m modeling run-all experiments/supcon/
    """
    configs = sorted(config_dir.glob("*.yaml"))
    if not configs:
        typer.echo(f"No .yaml files found in {config_dir}")
        raise typer.Exit(1)

    typer.echo(f"Found {len(configs)} config(s) in {config_dir}:")
    for c in configs:
        typer.echo(f"  - {c}")
    typer.echo("")

    for i, cfg_path in enumerate(configs, 1):
        typer.echo(f"{'=' * 42}")
        typer.echo(f"[{i}/{len(configs)}] Running: {cfg_path}")
        typer.echo(f"{'=' * 42}")
        cfg = load_config(cfg_path)
        kwargs = asdict(cfg)
        if sample_rate is not None:
            kwargs["sample_rate"] = sample_rate
        if self_contrast is not None:
            kwargs["self_contrast"] = self_contrast
        main(**kwargs)
        typer.echo("")

    typer.echo("All experiments complete.")


@app.command()
def pretrain(
    dataset_path: Annotated[
        str, typer.Option(help="Path to the JSONL dataset.")
    ] = "data/csn_jp.jsonl",
    model_name: Annotated[
        str, typer.Option(help="Model name or path.")
    ] = "microsoft/codebert-base",
    tokenizer_name: Annotated[
        Optional[str], typer.Option(help="Tokenizer name (defaults to model_name).")
    ] = None,
    batch_size: Annotated[
        int, typer.Option(help="Total batch size across GPUs.")
    ] = 256,
    num_epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = 10,
    gradient_accumulation_steps: Annotated[
        int, typer.Option(help="Gradient accumulation steps.")
    ] = 1,
    learning_rate: Annotated[float, typer.Option(help="Peak learning rate.")] = 2e-4,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 0,
    run_name: Annotated[
        str, typer.Option(help="W&B run name and output directory.")
    ] = "InvariantBERT",
    alpha: Annotated[float, typer.Option(help="Weight for contrastive loss.")] = 1.0,
    temperature: Annotated[float, typer.Option(help="Contrastive temperature.")] = 0.07,
    max_seq_length: Annotated[
        int, typer.Option(help="Max tokenizer sequence length.")
    ] = 256,
    sample_rate: Annotated[
        float, typer.Option(help="Fraction of dataset to sample.")
    ] = 1.0,
    num_proc: Annotated[
        int,
        typer.Option(
            help="Parallel workers for dataset preprocessing (datasets filter/map)."
        ),
    ] = default_num_proc(),
    resume: Annotated[
        bool, typer.Option(help="Resume from latest checkpoint.")
    ] = False,
    checkpoint: Annotated[
        Optional[str], typer.Option(help="Path to checkpoint for weight loading.")
    ] = None,
    self_contrast: Annotated[
        bool,
        typer.Option(
            help="Use self-contrast (same code, different MLM masks) for rows without augmentation. If disabled, rows without augmentation are dropped."
        ),
    ] = True,
    model_type: Annotated[
        ModelType, typer.Option(help="Model architecture type.")
    ] = ModelType.ROBERTA,
    pooling: Annotated[
        str,
        typer.Option(help="Pooling strategy for contrastive embeddings (cls or mean)."),
    ] = "cls",
    mlm_weight: Annotated[float, typer.Option(help="Weight for MLM loss.")] = 1.0,
    include_nl: Annotated[
        bool,
        typer.Option(help="Include NL docstrings in input (bimodal NL+PL training)."),
    ] = False,
) -> None:
    """Run pre-training with all parameters specified as CLI options.

    Example: python -m modeling pretrain --batch-size 64 --num-epochs 3
    """

    main(
        dataset_path=dataset_path,
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        batch_size=batch_size,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        run_name=run_name,
        alpha=alpha,
        temperature=temperature,
        max_seq_length=max_seq_length,
        sample_rate=sample_rate,
        num_proc=num_proc,
        resume=resume,
        checkpoint=checkpoint,
        self_contrast=self_contrast,
        model_type=model_type,
        pooling=pooling,
        mlm_weight=mlm_weight,
        include_nl=include_nl,
    )


if __name__ == "__main__":
    app()
