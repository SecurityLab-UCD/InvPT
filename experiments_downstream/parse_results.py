#!/usr/bin/env python3
"""Parse downstream evaluation logs into pivot-style Markdown tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Score parsing helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

# Ordered subsets that become column pairs (<subset>, <subset> aug).
SUBSETS = ["Java250", "Python800", "C++1400", "POJ104"]

# Map from task category to (codenet_task_dir_name, poj104_task_dir_name).
CLONE_CODENET = "Clone-detection-CodeNet"
CLONE_POJ104 = "Clone-detection-POJ104"
CLS_CODENET = "Code-classification-CodeNet"
CLS_POJ104 = "Code-classification-POJ104"

OPERATOR_KEYS = [
    ("localvarrenaming", "VarRe"),
    ("for2while", "F2W"),
    ("while2for", "W2F"),
    ("pp2addassignment", "PP2AA"),
    ("addassignment2equalassignment", "AA2EA"),
    ("reverseifelse", "RevIf"),
]
PYTHON_ONLY_OPS = {
    "localvarrenaming",
    "addassignment2equalassignment",
    "reverseifelse",
}


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}"


def _collect_clone_row(model_dir: Path, digits: int) -> dict[str, str]:
    row: dict[str, str] = {}
    # CodeNet subsets
    codenet_dir = model_dir / CLONE_CODENET
    for subset in ("Java250", "Python800", "C++1400"):
        subset_dir = codenet_dir / subset
        reg = parse_clone_score(subset_dir / "test.log")
        aug = parse_clone_score(subset_dir / "aug_test.log")
        row[subset] = _fmt(reg, digits)
        row[f"{subset} aug"] = _fmt(aug, digits)
    # POJ104 (logs directly in the task dir)
    poj_dir = model_dir / CLONE_POJ104
    reg = parse_clone_score(poj_dir / "test.log")
    aug = parse_clone_score(poj_dir / "aug_test.log")
    row["POJ104"] = _fmt(reg, digits)
    row["POJ104 aug"] = _fmt(aug, digits)
    return row


def _collect_cls_row(model_dir: Path, digits: int) -> dict[str, str]:
    row: dict[str, str] = {}
    # CodeNet subsets
    codenet_dir = model_dir / CLS_CODENET
    for subset in ("Java250", "Python800", "C++1400"):
        subset_dir = codenet_dir / subset
        reg = parse_classification_score(subset_dir / "test_train.log")
        aug = parse_classification_score(subset_dir / "aug_test.log")
        row[subset] = _fmt(reg, digits)
        row[f"{subset} aug"] = _fmt(aug, digits)
    # POJ104
    poj_dir = model_dir / CLS_POJ104
    reg = parse_classification_score(poj_dir / "test_train.log")
    aug = parse_classification_score(poj_dir / "aug_test.log")
    row["POJ104"] = _fmt(reg, digits)
    row["POJ104 aug"] = _fmt(aug, digits)
    return row


def _collect_per_op_clone_row(
    model_dir: Path, subset: str, digits: int
) -> dict[str, str]:
    row: dict[str, str] = {}
    if subset == "POJ104":
        task_dir = model_dir / CLONE_POJ104
    else:
        task_dir = model_dir / CLONE_CODENET / subset

    row["Original"] = _fmt(parse_clone_score(task_dir / "test.log"), digits)
    row["All (cum.)"] = _fmt(parse_clone_score(task_dir / "aug_test.log"), digits)
    is_python = subset == "Python800"
    for op_key, short_name in OPERATOR_KEYS:
        if is_python and op_key not in PYTHON_ONLY_OPS:
            row[short_name] = "n/a"
            continue
        row[short_name] = _fmt(
            parse_clone_score(task_dir / f"aug_test_{op_key}.log"), digits
        )
    return row


def _collect_per_op_cls_row(
    model_dir: Path, subset: str, digits: int
) -> dict[str, str]:
    row: dict[str, str] = {}
    if subset == "POJ104":
        task_dir = model_dir / CLS_POJ104
    else:
        task_dir = model_dir / CLS_CODENET / subset

    row["Original"] = _fmt(
        parse_classification_score(task_dir / "test_train.log"), digits
    )
    row["All (cum.)"] = _fmt(
        parse_classification_score(task_dir / "aug_test.log"), digits
    )
    is_python = subset == "Python800"
    for op_key, short_name in OPERATOR_KEYS:
        if is_python and op_key not in PYTHON_ONLY_OPS:
            row[short_name] = "n/a"
            continue
        row[short_name] = _fmt(
            parse_classification_score(task_dir / f"aug_test_{op_key}.log"),
            digits,
        )
    return row


def build_table(
    results_root: Path,
    collector,
    digits: int,
) -> pd.DataFrame:
    columns = []
    for s in SUBSETS:
        columns.extend([s, f"{s} aug"])

    rows: list[dict[str, str]] = []
    if not results_root.exists():
        return pd.DataFrame(columns=["Model"] + columns)

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        row = collector(model_dir, digits)
        row["Model"] = model_dir.name
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Model"] + columns)

    df = pd.DataFrame(rows)
    df = df[["Model"] + columns]
    return df


def build_per_operator_table(
    results_root: Path,
    subset: str,
    collector,
    digits: int,
) -> pd.DataFrame:
    columns = ["Original", "All (cum.)"] + [short for _, short in OPERATOR_KEYS]
    rows: list[dict[str, str]] = []
    if not results_root.exists():
        return pd.DataFrame(columns=["Model"] + columns)

    for model_dir in sorted(results_root.iterdir()):
        if not model_dir.is_dir():
            continue
        row = collector(model_dir, subset, digits)
        row["Model"] = model_dir.name
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["Model"] + columns)

    df = pd.DataFrame(rows)
    df = df[["Model"] + columns]
    return df


def _has_scored_cells(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    for value in df.drop(columns=["Model"]).to_numpy().ravel():
        if value not in {"-", "n/a"}:
            return True
    return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse downstream results into pivot Markdown tables."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing model result folders.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=2,
        help="Decimal places for percentage scores.",
    )
    parser.add_argument(
        "--per-operator",
        action="store_true",
        help="Print per-operator robustness breakdown tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()

    clone_df = build_table(results_root, _collect_clone_row, args.digits)
    cls_df = build_table(results_root, _collect_cls_row, args.digits)

    found = False
    if not clone_df.empty:
        found = True
        print("# Clone Detection (MAP@R)\n")
        print(clone_df.to_markdown(index=False))
        print()

    if not cls_df.empty:
        found = True
        print("# Code Classification (Acc)\n")
        print(cls_df.to_markdown(index=False))
        print()

    if args.per_operator:
        for subset in SUBSETS:
            clone_per_op_df = build_per_operator_table(
                results_root,
                subset,
                _collect_per_op_clone_row,
                args.digits,
            )
            if _has_scored_cells(clone_per_op_df):
                found = True
                print(f"# Per-Operator Clone Detection — {subset} (MAP@R)\n")
                print(clone_per_op_df.to_markdown(index=False))
                print()

            cls_per_op_df = build_per_operator_table(
                results_root,
                subset,
                _collect_per_op_cls_row,
                args.digits,
            )
            if _has_scored_cells(cls_per_op_df):
                found = True
                print(f"# Per-Operator Code Classification — {subset} (Acc)\n")
                print(cls_per_op_df.to_markdown(index=False))
                print()

    if not found:
        print(f"No results found under {results_root}")


if __name__ == "__main__":
    main()
