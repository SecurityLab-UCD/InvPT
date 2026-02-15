#!/usr/bin/env python3
"""Launch downstream tasks in parallel across 8 GPUs."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import typer


@dataclass(frozen=True)
class ModelSpec:
    model_path: str
    tokenizer_name: str
    model_type: str


@dataclass(frozen=True)
class RunHandle:
    label: str
    process: subprocess.Popen
    log_file: TextIO


MODELS: dict[tuple[str, str], ModelSpec] = {
    ("inv-codebert", "supcon"): ModelSpec(
        "./saved_models/InvCodeBERT-supcon/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    ("inv-graphcodebert", "supcon"): ModelSpec(
        "./saved_models/InvGraphCodeBERT-supcon/final",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    ("inv-contrabert_c", "supcon"): ModelSpec(
        "./saved_models/InvContraBERT_C-supcon/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    ("inv-contrabert_g", "supcon"): ModelSpec(
        "./saved_models/InvContraBERT_G-supcon/final",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    ("inv-modernbert", "supcon"): ModelSpec(
        "./saved_models/aug-only/InvModernBERT-supcon/final",
        "answerdotai/ModernBERT-base",
        "modernbert",
    ),
}


TASKS = [
    ("Clone-detection-POJ104", None),
    ("Clone-detection-CodeNet", "Java250"),
    ("Clone-detection-CodeNet", "Python800"),
    ("Clone-detection-CodeNet", "C++1400"),
    ("Code-classification-POJ104", "Cpp"),
    ("Code-classification-CodeNet", "Java250"),
    ("Code-classification-CodeNet", "Python800"),
    ("Code-classification-CodeNet", "C++1400"),
]


app = typer.Typer(add_completion=False, no_args_is_help=True)


def resolve_model(model_key: str, loss: str) -> ModelSpec:
    key = (model_key, loss)
    if key not in MODELS:
        available = ", ".join(
            f"{model}/{loss_key}"
            for model, loss_key in sorted(MODELS.keys(), key=lambda x: (x[0], x[1]))
        )
        raise typer.BadParameter(
            f"Unknown model/loss: {model_key}/{loss}. Available: {available}"
        )
    return MODELS[key]


def run_task(
    root: Path,
    model_key: str,
    spec: ModelSpec,
    task_dir: str,
    subset: str | None,
    gpu_id: str,
    results_root: Path,
    dry_run: bool,
) -> RunHandle | None:
    task_path = root / "downstream" / task_dir
    model_path = spec.model_path
    if model_path.startswith("./"):
        model_path = str((root / model_path[2:]).resolve())

    results_base = results_root / model_key / task_dir
    results_base_str = str(results_base.resolve())
    log_path = results_base / "run.log"

    cmd = ["./run.sh", model_path]
    if task_dir == "Clone-detection-POJ104":
        cmd.extend([results_base_str, spec.model_type, spec.tokenizer_name])
    elif task_dir == "Clone-detection-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Clone-detection-CodeNet")
        cmd.extend(
            [
                results_base_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    elif task_dir == "Code-classification-POJ104":
        if subset is None:
            raise ValueError("Subset is required for Code-classification-POJ104")
        cmd.extend(
            [
                results_base_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    elif task_dir == "Code-classification-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Code-classification-CodeNet")
        cmd.extend(
            [
                results_base_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    else:
        raise ValueError(f"Unsupported task: {task_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    label = f"{task_dir}{'/' + subset if subset else ''}"
    if dry_run:
        print(f"[launch] GPU {gpu_id}: {label}")
        print(f"         cwd: {task_path}")
        print(f"         cmd: {' '.join(cmd)}")
        return None

    results_base.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    log_file.write(f"[launch] GPU {gpu_id}: {label}\n")
    log_file.write(f"[launch] cwd: {task_path}\n")
    log_file.write(f"[launch] cmd: {' '.join(cmd)}\n\n")
    log_file.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(task_path),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return RunHandle(label=label, process=proc, log_file=log_file)


@app.command()
def main(
    loss: str = typer.Option(
        ..., "--loss", help="Training loss identifier (e.g., supcon)."
    ),
    model: str = typer.Option(
        ..., "--model", help="Pretrained model key (e.g., inv-codebert)."
    ),
    gpus: str = typer.Option(
        "0,1,2,3,4,5,6,7",
        "--gpus",
        help="Comma-separated GPU ids to use (must be 8).",
    ),
    results_root: str = typer.Option(
        "results", "--results-root", help="Base output directory for results."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing them."
    ),
) -> None:
    root = Path(__file__).resolve().parents[1]

    model_key = model.strip()
    loss_key = loss.strip()
    spec = resolve_model(model_key, loss_key)

    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if len(gpu_ids) != len(TASKS):
        raise typer.BadParameter(
            f"Expected {len(TASKS)} GPU ids, got {len(gpu_ids)}: {gpu_ids}"
        )

    results_root_path = Path(results_root).resolve()

    processes: list[RunHandle] = []
    for (task_dir, subset), gpu_id in zip(TASKS, gpu_ids, strict=True):
        proc = run_task(
            root,
            model_key,
            spec,
            task_dir,
            subset,
            gpu_id,
            results_root_path,
            dry_run,
        )
        if proc is not None:
            processes.append(proc)

    if dry_run:
        raise typer.Exit(0)

    failures: list[str] = []
    for handle in processes:
        code = handle.process.wait()
        handle.log_file.close()
        if code != 0:
            failures.append(f"{handle.label} (exit {code})")

    if failures:
        print("\n[error] Some tasks failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise typer.Exit(1)

    print("\n[done] All downstream tasks completed successfully.")


if __name__ == "__main__":
    app()
