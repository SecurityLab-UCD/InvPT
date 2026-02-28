#!/usr/bin/env python3
"""Launch downstream tasks across GPUs with a work-stealing pool."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import typer
from tqdm import tqdm


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
    # --- baselines ---
    ("codebert", "supcon"): ModelSpec(
        "microsoft/codebert-base",
        "microsoft/codebert-base",
        "roberta",
    ),
    ("graphcodebert", "supcon"): ModelSpec(
        "microsoft/graphcodebert-base",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    ("contrabert_c", "supcon"): ModelSpec(
        "./saved_models/ContraBERT_C",
        "microsoft/codebert-base",
        "roberta",
    ),
    ("contrabert_g", "supcon"): ModelSpec(
        "./saved_models/ContraBERT_G",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    ("codesage", "supcon"): ModelSpec(
        "codesage/codesage-small",
        "codesage/codesage-small",
        "codesage",
    ),
    # --- InvPT models ---
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

# Ablation models — all based on CodeBERT
ABLATION_MODELS: dict[str, ModelSpec] = {
    "contra-only": ModelSpec(
        "./saved_models/InvCodeBERT-ablation-contra-only/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    "mlm-only": ModelSpec(
        "./saved_models/InvCodeBERT-ablation-mlm-only/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    "no-self-contrast": ModelSpec(
        "./saved_models/InvCodeBERT-ablation-no-selfcon/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    "infonce": ModelSpec(
        "./saved_models/InvCodeBERT-ablation-infonce/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    "include-nl": ModelSpec(
        "./saved_models/InvCodeBERT-ablation-include-nl/final",
        "microsoft/codebert-base",
        "roberta",
    ),
}


TASKS = [
    ("Clone-detection-POJ104", None),
    ("Clone-detection-CodeNet", "Java250"),
    ("Clone-detection-CodeNet", "Python800"),
    ("Clone-detection-CodeNet", "C++1400"),
    ("Code-classification-POJ104", None),
    ("Code-classification-CodeNet", "Java250"),
    ("Code-classification-CodeNet", "Python800"),
    ("Code-classification-CodeNet", "C++1400"),
]

OPERATOR_KEYS = [
    "localvarrenaming",
    "for2while",
    "while2for",
    "pp2addassignment",
    "addassignment2equalassignment",
    "reverseifelse",
]

LANG_FOR_TASK: dict[tuple[str, str | None], str] = {
    ("Clone-detection-POJ104", None): "cpp",
    ("Clone-detection-CodeNet", "Java250"): "java",
    ("Clone-detection-CodeNet", "Python800"): "python",
    ("Clone-detection-CodeNet", "C++1400"): "cpp",
    ("Code-classification-POJ104", None): "cpp",
    ("Code-classification-CodeNet", "Java250"): "java",
    ("Code-classification-CodeNet", "Python800"): "python",
    ("Code-classification-CodeNet", "C++1400"): "cpp",
}

OPS_FOR_LANG: dict[str, list[str]] = {
    "cpp": OPERATOR_KEYS,
    "java": OPERATOR_KEYS,
    "python": [
        "localvarrenaming",
        "addassignment2equalassignment",
        "reverseifelse",
    ],
}


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
    results_dir = results_base / subset if subset else results_base
    results_dir_str = str(results_dir.resolve())
    log_path = results_dir / "run.log"

    cmd = ["./run.sh", model_path]
    if task_dir == "Clone-detection-POJ104":
        cmd.extend([results_dir_str, spec.model_type, spec.tokenizer_name])
    elif task_dir == "Clone-detection-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Clone-detection-CodeNet")
        cmd.extend(
            [
                results_dir_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    elif task_dir == "Code-classification-POJ104":
        subset_arg = subset or ""
        cmd.extend(
            [
                results_dir_str,
                subset_arg,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    elif task_dir == "Code-classification-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Code-classification-CodeNet")
        cmd.extend(
            [
                results_dir_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
            ]
        )
    else:
        raise ValueError(f"Unsupported task: {task_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    label = f"{model_key}/{task_dir}{'/' + subset if subset else ''}"
    if dry_run:
        print(f"[launch] GPU {gpu_id}: {label}")
        print(f"         cwd: {task_path}")
        print(f"         cmd: {' '.join(cmd)}")
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
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


def run_per_operator_task(
    root: Path,
    model_key: str,
    spec: ModelSpec,
    task_dir: str,
    subset: str | None,
    operator_key: str,
    gpu_id: str,
    results_root: Path,
    dry_run: bool,
) -> RunHandle | None:
    task_path = root / "downstream" / task_dir
    model_path = spec.model_path
    if model_path.startswith("./"):
        model_path = str((root / model_path[2:]).resolve())

    results_base = results_root / model_key / task_dir
    results_dir = results_base / subset if subset else results_base
    results_dir_str = str(results_dir.resolve())
    log_path = results_dir / f"run_aug_{operator_key}.log"

    cmd = ["./run_aug_test.sh", model_path]
    if task_dir == "Clone-detection-POJ104":
        cmd.extend(
            [results_dir_str, spec.model_type, spec.tokenizer_name, operator_key]
        )
    elif task_dir == "Clone-detection-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Clone-detection-CodeNet")
        cmd.extend(
            [
                results_dir_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
                operator_key,
            ]
        )
    elif task_dir == "Code-classification-POJ104":
        subset_arg = subset or ""
        cmd.extend(
            [
                results_dir_str,
                subset_arg,
                spec.model_type,
                spec.tokenizer_name,
                operator_key,
            ]
        )
    elif task_dir == "Code-classification-CodeNet":
        if subset is None:
            raise ValueError("Subset is required for Code-classification-CodeNet")
        cmd.extend(
            [
                results_dir_str,
                subset,
                spec.model_type,
                spec.tokenizer_name,
                operator_key,
            ]
        )
    else:
        raise ValueError(f"Unsupported task: {task_dir}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    label = f"{model_key}/{task_dir}{'/' + subset if subset else ''}/{operator_key}"
    if dry_run:
        print(f"[launch] GPU {gpu_id}: {label}")
        print(f"         cwd: {task_path}")
        print(f"         cmd: {' '.join(cmd)}")
        return None

    results_dir.mkdir(parents=True, exist_ok=True)
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


def _run_jobs(
    jobs: list[tuple[str, ModelSpec, str, str | None]],
    gpu_ids: list[str],
    results_root_path: Path,
    dry_run: bool,
    desc: str = "Downstream",
) -> None:
    """Execute *jobs* across *gpu_ids* with a work-stealing thread pool."""
    root = Path(__file__).resolve().parents[1]
    total = len(jobs)

    if dry_run:
        for i, (mk, sp, task_dir, subset) in enumerate(jobs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            run_task(root, mk, sp, task_dir, subset, gpu_id, results_root_path, True)
        print(f"\n[dry-run] {total} total jobs across {len(gpu_ids)} GPUs")
        raise typer.Exit(0)

    gpu_pool: queue.Queue[str] = queue.Queue()
    for gid in gpu_ids:
        gpu_pool.put(gid)

    running = 0
    failed = 0
    lock = threading.Lock()
    failures: list[str] = []
    pbar = tqdm(total=total, desc=desc, unit="task")

    def run_job(mk: str, sp: ModelSpec, task_dir: str, subset: str | None) -> None:
        nonlocal running, failed
        gpu_id = gpu_pool.get()
        label = f"{mk}/{task_dir}{'/' + subset if subset else ''}"
        try:
            with lock:
                running += 1
                pbar.set_postfix(running=running, failed=failed, refresh=True)
            handle = run_task(
                root, mk, sp, task_dir, subset, gpu_id, results_root_path, False
            )
            if handle is not None:
                code = handle.process.wait()
                handle.log_file.close()
                with lock:
                    running -= 1
                    if code != 0:
                        failed += 1
                        failures.append(f"{label} (exit {code})")
                        tqdm.write(f"[FAIL] GPU {gpu_id}: {label} (exit {code})")
                    else:
                        tqdm.write(f"[done] GPU {gpu_id}: {label}")
                    pbar.set_postfix(running=running, failed=failed, refresh=False)
                    pbar.update(1)
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(run_job, mk, sp, task_dir, subset)
            for mk, sp, task_dir, subset in jobs
        ]
        for f in futures:
            f.result()
    pbar.close()

    if failures:
        print(f"\n[error] {len(failures)}/{total} tasks failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise typer.Exit(1)

    print(f"\n[done] All {total} tasks completed successfully.")


def _parse_gpus(gpus: str) -> list[str]:
    gpu_ids = [gpu.strip() for gpu in gpus.split(",") if gpu.strip()]
    if not gpu_ids:
        raise typer.BadParameter("No GPU ids provided")
    return gpu_ids


@app.command()
def run(
    all_models: bool = typer.Option(
        False, "--all", help="Run all models in the registry."
    ),
    loss: str = typer.Option(
        None, "--loss", help="Training loss identifier (e.g., supcon)."
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Pretrained model key(s), comma-separated (e.g., inv-codebert,inv-graphcodebert).",
    ),
    gpus: str = typer.Option(
        "0,1,2,3,4,5,6,7",
        "--gpus",
        help="Comma-separated GPU ids to use.",
    ),
    results_root: str = typer.Option(
        "results", "--results-root", help="Base output directory for results."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing them."
    ),
) -> None:
    """Run downstream evaluation for pretrained / baseline models."""
    # Resolve which models to run
    if all_models:
        if model is not None:
            raise typer.BadParameter("Cannot use --model with --all")
        entries = list(MODELS.items())
        if loss is not None:
            loss_key = loss.strip()
            entries = [((m, lk), s) for (m, lk), s in entries if lk == loss_key]
        if not entries:
            raise typer.BadParameter(f"No models found for loss={loss}")
    else:
        if model is None or loss is None:
            raise typer.BadParameter(
                "Either --all or both --model and --loss are required"
            )
        loss_key = loss.strip()
        model_keys = [m.strip() for m in model.split(",") if m.strip()]
        if not model_keys:
            raise typer.BadParameter("No model keys provided")
        entries = []
        for model_key in model_keys:
            spec = resolve_model(model_key, loss_key)
            entries.append(((model_key, loss_key), spec))

    gpu_ids = _parse_gpus(gpus)

    jobs: list[tuple[str, ModelSpec, str, str | None]] = []
    for (mk, _lk), sp in entries:
        for task_dir, subset in TASKS:
            jobs.append((mk, sp, task_dir, subset))

    _run_jobs(jobs, gpu_ids, Path(results_root).resolve(), dry_run)


@app.command("per-operator")
def per_operator(
    all_models: bool = typer.Option(
        False, "--all", help="Run all models in the registry."
    ),
    loss: str = typer.Option(
        None, "--loss", help="Training loss identifier (e.g., supcon)."
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Pretrained model key(s), comma-separated.",
    ),
    gpus: str = typer.Option(
        "0,1,2,3,4,5,6,7",
        "--gpus",
        help="Comma-separated GPU ids to use.",
    ),
    results_root: str = typer.Option(
        "results", "--results-root", help="Base output directory for results."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing them."
    ),
) -> None:
    """Run per-operator robustness evaluation on augmented test sets."""
    if all_models:
        if model is not None:
            raise typer.BadParameter("Cannot use --model with --all")
        entries = list(MODELS.items())
        if loss is not None:
            loss_key = loss.strip()
            entries = [((m, lk), s) for (m, lk), s in entries if lk == loss_key]
        if not entries:
            raise typer.BadParameter(f"No models found for loss={loss}")
    else:
        if model is None or loss is None:
            raise typer.BadParameter(
                "Either --all or both --model and --loss are required"
            )
        loss_key = loss.strip()
        model_keys = [m.strip() for m in model.split(",") if m.strip()]
        if not model_keys:
            raise typer.BadParameter("No model keys provided")
        entries = []
        for model_key in model_keys:
            spec = resolve_model(model_key, loss_key)
            entries.append(((model_key, loss_key), spec))

    gpu_ids = _parse_gpus(gpus)
    root = Path(__file__).resolve().parents[1]
    results_root_path = Path(results_root).resolve()

    jobs: list[tuple[str, ModelSpec, str, str | None, str]] = []
    for (mk, _lk), sp in entries:
        for task_dir, subset in TASKS:
            lang = LANG_FOR_TASK[(task_dir, subset)]
            for operator_key in OPS_FOR_LANG[lang]:
                jobs.append((mk, sp, task_dir, subset, operator_key))

    total = len(jobs)
    if total == 0:
        print("No per-operator jobs to run.")
        return

    if dry_run:
        for i, (mk, sp, task_dir, subset, operator_key) in enumerate(jobs):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            run_per_operator_task(
                root,
                mk,
                sp,
                task_dir,
                subset,
                operator_key,
                gpu_id,
                results_root_path,
                True,
            )
        print(f"\n[dry-run] {total} total per-operator jobs across {len(gpu_ids)} GPUs")
        raise typer.Exit(0)

    gpu_pool: queue.Queue[str] = queue.Queue()
    for gid in gpu_ids:
        gpu_pool.put(gid)

    running = 0
    failed = 0
    lock = threading.Lock()
    failures: list[str] = []
    pbar = tqdm(total=total, desc="Per-operator", unit="task")

    def run_job(
        mk: str,
        sp: ModelSpec,
        task_dir: str,
        subset: str | None,
        operator_key: str,
    ) -> None:
        nonlocal running, failed
        gpu_id = gpu_pool.get()
        label = f"{mk}/{task_dir}{'/' + subset if subset else ''}/{operator_key}"
        try:
            with lock:
                running += 1
                pbar.set_postfix(running=running, failed=failed, refresh=True)
            handle = run_per_operator_task(
                root,
                mk,
                sp,
                task_dir,
                subset,
                operator_key,
                gpu_id,
                results_root_path,
                False,
            )
            if handle is not None:
                code = handle.process.wait()
                handle.log_file.close()
                with lock:
                    running -= 1
                    if code != 0:
                        failed += 1
                        failures.append(f"{label} (exit {code})")
                        tqdm.write(f"[FAIL] GPU {gpu_id}: {label} (exit {code})")
                    else:
                        tqdm.write(f"[done] GPU {gpu_id}: {label}")
                    pbar.set_postfix(running=running, failed=failed, refresh=False)
                    pbar.update(1)
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [
            executor.submit(run_job, mk, sp, task_dir, subset, operator_key)
            for mk, sp, task_dir, subset, operator_key in jobs
        ]
        for f in futures:
            f.result()
    pbar.close()

    if failures:
        print(f"\n[error] {len(failures)}/{total} tasks failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise typer.Exit(1)

    print(f"\n[done] All {total} per-operator tasks completed successfully.")


@app.command()
def ablation(
    all_models: bool = typer.Option(False, "--all", help="Run all ablation models."),
    model: str = typer.Option(
        None,
        "--model",
        help=(
            "Ablation model key(s), comma-separated. "
            f"Available: {', '.join(sorted(ABLATION_MODELS))}."
        ),
    ),
    gpus: str = typer.Option(
        "0,1,2,3,4,5,6,7",
        "--gpus",
        help="Comma-separated GPU ids to use.",
    ),
    results_root: str = typer.Option(
        "results", "--results-root", help="Base output directory for results."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing them."
    ),
) -> None:
    """Run downstream evaluation for ablation models."""
    if all_models:
        if model is not None:
            raise typer.BadParameter("Cannot use --model with --all")
        selected = list(ABLATION_MODELS.items())
    else:
        if model is None:
            raise typer.BadParameter("Either --all or --model is required")
        keys = [k.strip() for k in model.split(",") if k.strip()]
        if not keys:
            raise typer.BadParameter("No model keys provided")
        selected = []
        for k in keys:
            if k not in ABLATION_MODELS:
                available = ", ".join(sorted(ABLATION_MODELS))
                raise typer.BadParameter(
                    f"Unknown ablation model: {k}. Available: {available}"
                )
            selected.append((k, ABLATION_MODELS[k]))

    gpu_ids = _parse_gpus(gpus)

    jobs: list[tuple[str, ModelSpec, str, str | None]] = []
    for mk, sp in selected:
        for task_dir, subset in TASKS:
            jobs.append((mk, sp, task_dir, subset))

    _run_jobs(jobs, gpu_ids, Path(results_root).resolve(), dry_run, desc="Ablation")


if __name__ == "__main__":
    app()
