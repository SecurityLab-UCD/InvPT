#!/usr/bin/env python3
"""Parse downstream evaluation logs into a summary table."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScorePair:
    regular: float | None
    augmented: float | None


@dataclass(frozen=True)
class ParsedResult:
    model: str
    dataset: str
    task: str
    metric: str
    scores: ScorePair


CLONE_TASKS = {
    "Clone-detection-POJ104",
    "Clone-detection-CodeNet",
}
CLS_TASKS = {
    "Code-classification-POJ104",
    "Code-classification-CodeNet",
}

MAP_PATTERN = re.compile(r"MAP@R\W*[:=]?\W*(\d*\.?\d+)")
TEST_ACC_PATTERN = re.compile(r"\btest_acc\b\s*=\s*(\d*\.?\d+)")
FLOAT64_PATTERN = re.compile(r"np\.float64\((\d*\.?\d+)\)")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return ""


def parse_clone_score(path: Path) -> float | None:
    text = _read_text(path).strip()
    if not text:
        return None
    match = MAP_PATTERN.search(text)
    if match:
        return float(match.group(1))
    match = FLOAT64_PATTERN.search(text)
    if match:
        return float(match.group(1))
    try:
        data = json.loads(text.replace("'", '"'))
    except json.JSONDecodeError:
        return None
    value = data.get("MAP@R")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_classification_score(path: Path) -> float | None:
    text = _read_text(path)
    if not text:
        return None
    matches = TEST_ACC_PATTERN.findall(text)
    if not matches:
        return None
    return float(matches[-1])


def iter_result_dirs(results_root: Path) -> Iterable[tuple[str, Path]]:
    if not results_root.exists():
        return iter(())
    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        yield model_dir.name, model_dir


def infer_task(task_dir: Path) -> str | None:
    name = task_dir.name
    if name in CLONE_TASKS or name in CLS_TASKS:
        return name
    return None


def dataset_label(task: str, subset: str | None) -> str:
    if subset:
        return f"{task}/{subset}"
    return task


def collect_results(results_root: Path) -> list[ParsedResult]:
    results: list[ParsedResult] = []
    for model_name, model_dir in iter_result_dirs(results_root):
        for task_dir in sorted(model_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task = infer_task(task_dir)
            if task is None:
                continue
            if task in CLONE_TASKS:
                collect_clone_results(results, model_name, task_dir, task)
            elif task in CLS_TASKS:
                collect_classification_results(results, model_name, task_dir, task)
    return results


def collect_clone_results(
    results: list[ParsedResult],
    model: str,
    task_dir: Path,
    task: str,
) -> None:
    task_paths = list(task_dir.iterdir())
    if any(p.is_dir() for p in task_paths):
        for subset_dir in sorted(p for p in task_paths if p.is_dir()):
            regular = parse_clone_score(subset_dir / "test.log")
            augmented = parse_clone_score(subset_dir / "aug_test.log")
            results.append(
                ParsedResult(
                    model=model,
                    dataset=dataset_label(task, subset_dir.name),
                    task=task,
                    metric="MAP@R",
                    scores=ScorePair(regular, augmented),
                )
            )
    else:
        regular = parse_clone_score(task_dir / "test.log")
        augmented = parse_clone_score(task_dir / "aug_test.log")
        results.append(
            ParsedResult(
                model=model,
                dataset=dataset_label(task, None),
                task=task,
                metric="MAP@R",
                scores=ScorePair(regular, augmented),
            )
        )


def collect_classification_results(
    results: list[ParsedResult],
    model: str,
    task_dir: Path,
    task: str,
) -> None:
    task_paths = list(task_dir.iterdir())
    if any(p.is_dir() for p in task_paths):
        for subset_dir in sorted(p for p in task_paths if p.is_dir()):
            regular = parse_classification_score(subset_dir / "test_train.log")
            augmented = parse_classification_score(subset_dir / "aug_test.log")
            results.append(
                ParsedResult(
                    model=model,
                    dataset=dataset_label(task, subset_dir.name),
                    task=task,
                    metric="Accuracy",
                    scores=ScorePair(regular, augmented),
                )
            )
    else:
        regular = parse_classification_score(task_dir / "test_train.log")
        augmented = parse_classification_score(task_dir / "aug_test.log")
        results.append(
            ParsedResult(
                model=model,
                dataset=dataset_label(task, None),
                task=task,
                metric="Accuracy",
                scores=ScorePair(regular, augmented),
            )
        )


def format_score(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def render_table(results: list[ParsedResult], digits: int) -> str:
    headers = ["Model", "Dataset", "Metric", "Regular (%)", "Augmented (%)"]
    rows = []
    for result in results:
        rows.append(
            [
                result.model,
                result.dataset,
                result.metric,
                format_score(result.scores.regular, digits),
                format_score(result.scores.augmented, digits),
            ]
        )

    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    header_line = "  ".join(h.ljust(widths[idx]) for idx, h in enumerate(headers))
    divider = "  ".join("-" * widths[idx] for idx in range(len(headers)))
    body = [
        "  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers)))
        for row in rows
    ]
    return "\n".join([header_line, divider, *body])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse downstream results into a table."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing result folders.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for percentage scores.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    results = collect_results(results_root)
    if not results:
        print(f"No results found under {results_root}")
        return
    print(render_table(results, args.digits))


if __name__ == "__main__":
    main()
