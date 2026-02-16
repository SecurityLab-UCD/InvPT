from __future__ import annotations

from pathlib import Path

from experiments_downstream.parse_results import (
    _collect_clone_row,
    _collect_cls_row,
    build_table,
    parse_classification_score,
    parse_clone_score,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_parse_clone_score_float64(tmp_path: Path) -> None:
    log = tmp_path / "test.log"
    log.write_text("{'MAP@R': np.float64(0.5621)}\n")
    assert parse_clone_score(log) == 0.5621


def test_parse_clone_score_missing(tmp_path: Path) -> None:
    assert parse_clone_score(tmp_path / "nonexistent.log") is None


def test_parse_classification_score(tmp_path: Path) -> None:
    log = tmp_path / "test_train.log"
    log.write_text(
        "02/16/2026 00:00:00 - INFO - __main__ -   ***** Test results *****\n"
        "02/16/2026 00:00:00 - INFO - __main__ -   test_acc = 0.3992\n"
    )
    assert parse_classification_score(log) == 0.3992


def test_parse_classification_score_missing(tmp_path: Path) -> None:
    assert parse_classification_score(tmp_path / "nonexistent.log") is None


def test_build_table_clone_and_classification(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    # Clone detection – POJ104 (files directly in task dir)
    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "test.log",
        "{'MAP@R': np.float64(0.5621)}\n",
    )
    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "aug_test.log",
        "{'MAP@R': np.float64(0.4321)}\n",
    )

    # Classification – CodeNet / Java250
    test_train_log = (
        "02/16/2026 00:00:00 - INFO - __main__ -   ***** Test results *****\n"
        "02/16/2026 00:00:00 - INFO - __main__ -   test_acc = 0.3992\n"
    )
    _write(
        results_root
        / "inv-codebert"
        / "Code-classification-CodeNet"
        / "Java250"
        / "test_train.log",
        test_train_log,
    )
    _write(
        results_root
        / "inv-codebert"
        / "Code-classification-CodeNet"
        / "Java250"
        / "aug_test.log",
        test_train_log.replace("0.3992", "0.2777"),
    )

    clone_df = build_table(results_root, _collect_clone_row, 4)
    assert len(clone_df) == 1
    row = clone_df.iloc[0]
    assert row["Model"] == "inv-codebert"
    assert row["POJ104"] == "56.2100"
    assert row["POJ104 aug"] == "43.2100"

    cls_df = build_table(results_root, _collect_cls_row, 4)
    assert len(cls_df) == 1
    row = cls_df.iloc[0]
    assert row["Model"] == "inv-codebert"
    assert row["Java250"] == "39.9200"
    assert row["Java250 aug"] == "27.7700"


def test_build_table_placeholder_for_missing(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "test.log",
        "{'MAP@R': np.float64(0.5621)}\n",
    )

    clone_df = build_table(results_root, _collect_clone_row, 4)
    row = clone_df.iloc[0]
    # POJ104 regular is present, but aug and CodeNet subsets should be "-"
    assert row["POJ104"] == "56.2100"
    assert row["POJ104 aug"] == "-"
    assert row["Java250"] == "-"
