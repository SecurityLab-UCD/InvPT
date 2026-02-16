#!/usr/bin/env python3
"""Run all models for a given task+dataset combination.

Usage examples:
    # Run all models for Code-classification-POJ104 on GPUs 0-7
    python run_task.py Code-classification-POJ104

    # Run all models for Clone-detection-CodeNet on GPUs 0,1,2,3
    python run_task.py Clone-detection-CodeNet --gpus 0,1,2,3

    # Only run scripts matching a subset (e.g., Java250)
    python run_task.py Clone-detection-CodeNet --subset Java250

    # Dry run to see what would execute
    python run_task.py Defect-detection --dry-run

    # List all available tasks
    python run_task.py --list
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import typer
from tqdm import tqdm

app = typer.Typer(add_completion=False, no_args_is_help=True)

SCRIPT_DIR = Path(__file__).resolve().parent

# Models to skip (not yet ready / under development)
SKIP_MODELS = {"modernbert"}


def _is_skipped(script_name: str) -> bool:
    """Return True if the script belongs to a skipped model."""
    return any(m in script_name for m in SKIP_MODELS)


def discover_tasks() -> list[str]:
    """Return sorted list of task directories (those containing .sh files)."""
    return sorted(
        d.name for d in SCRIPT_DIR.iterdir() if d.is_dir() and any(d.glob("*.sh"))
    )


def discover_scripts(task: str, subset: str | None = None) -> list[Path]:
    """Return all .sh scripts for a task, optionally filtered by subset."""
    task_path = SCRIPT_DIR / task
    if not task_path.is_dir():
        raise typer.BadParameter(
            f"Task directory not found: {task}\n"
            f"Available tasks: {', '.join(discover_tasks())}"
        )
    scripts = sorted(s for s in task_path.glob("*.sh") if not _is_skipped(s.name))
    if not scripts:
        raise typer.BadParameter(f"No .sh scripts found in {task}")
    if subset:
        filtered = [s for s in scripts if subset in s.name]
        if not filtered:
            all_names = [s.name for s in scripts]
            raise typer.BadParameter(
                f"No scripts matching subset '{subset}' in {task}.\n"
                f"Available scripts: {', '.join(all_names)}"
            )
        scripts = filtered
    return scripts


@app.command()
def main(
    task: str = typer.Argument(
        None,
        help="Task+dataset directory name (e.g., Code-classification-POJ104). "
        "Use --list to see all available tasks.",
    ),
    gpus: str = typer.Option(
        "0,1,2,3,4,5,6,7",
        "--gpus",
        help="Comma-separated GPU ids to use.",
    ),
    subset: str | None = typer.Option(
        None,
        "--subset",
        help="Only run scripts whose filename contains this substring "
        "(e.g., Java250, inv-, codebert).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print commands without executing them."
    ),
    list_tasks: bool = typer.Option(
        False, "--list", help="List all available task directories and exit."
    ),
) -> None:
    """Run all model scripts for a given task+dataset, distributed across GPUs."""

    # --list mode
    if list_tasks:
        tasks = discover_tasks()
        typer.echo("Available tasks:")
        for t in tasks:
            n_scripts = len(
                [s for s in (SCRIPT_DIR / t).glob("*.sh") if not _is_skipped(s.name)]
            )
            typer.echo(f"  {t}  ({n_scripts} scripts)")
        raise typer.Exit(0)

    if task is None:
        typer.echo("Error: Missing argument 'TASK'.\n")
        typer.echo("Use --list to see available tasks, or --help for usage.")
        raise typer.Exit(1)

    scripts = discover_scripts(task, subset)
    gpu_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise typer.BadParameter("No GPU ids provided")

    total = len(scripts)
    typer.echo(f"Task: {task}")
    typer.echo(f"Scripts to run: {total}")
    typer.echo(f"GPUs: {', '.join(gpu_ids)}")
    typer.echo("")

    if dry_run:
        for i, script in enumerate(scripts):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            typer.echo(f"  [GPU {gpu_id}] bash {script.name}")
        typer.echo(f"\n[dry-run] {total} jobs across {len(gpu_ids)} GPUs")
        raise typer.Exit(0)

    # Work-stealing GPU pool
    gpu_pool: queue.Queue[str] = queue.Queue()
    for gid in gpu_ids:
        gpu_pool.put(gid)

    running = 0
    failed = 0
    lock = threading.Lock()
    failures: list[str] = []
    pbar = tqdm(total=total, desc=task, unit="script")

    def run_script(script: Path) -> None:
        nonlocal running, failed
        gpu_id = gpu_pool.get()
        name = script.name
        log_path = script.with_suffix(".log")
        try:
            with lock:
                running += 1
                pbar.set_postfix(running=running, failed=failed, refresh=True)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id

            with open(log_path, "w") as log_file:
                log_file.write(f"[launch] GPU {gpu_id}: {name}\n")
                log_file.write(f"[launch] cmd: bash {script}\n\n")
                log_file.flush()

                proc = subprocess.Popen(
                    ["bash", str(script), gpu_id],
                    cwd=str(script.parent),
                    env=env,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
                code = proc.wait()

            with lock:
                running -= 1
                if code != 0:
                    failed += 1
                    failures.append(f"{name} (exit {code})")
                    tqdm.write(f"[FAIL] GPU {gpu_id}: {name} (exit {code})")
                else:
                    tqdm.write(f"[done] GPU {gpu_id}: {name}")
                pbar.set_postfix(running=running, failed=failed, refresh=False)
                pbar.update(1)
        finally:
            gpu_pool.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_script, s) for s in scripts]
        for f in futures:
            f.result()
    pbar.close()

    if failures:
        typer.echo(f"\n[error] {len(failures)}/{total} scripts failed:")
        for failure in failures:
            typer.echo(f"  - {failure}")
        raise typer.Exit(1)

    typer.echo(f"\n[done] All {total} scripts for {task} completed successfully.")


if __name__ == "__main__":
    app()
