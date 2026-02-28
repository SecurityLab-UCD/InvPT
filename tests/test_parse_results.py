from __future__ import annotations

from pathlib import Path

from experiments_downstream.parse_results import (
    _collect_clone_row,
    _collect_cls_row,
    _collect_per_op_clone_row,
    _collect_per_op_cls_row,
    build_per_operator_table,
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


def test_build_per_operator_table_clone_python_subset(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    base = results_root / "inv-codebert" / "Clone-detection-CodeNet" / "Python800"
    _write(base / "test.log", "{'MAP@R': np.float64(0.5000)}\n")
    _write(base / "aug_test.log", "{'MAP@R': np.float64(0.4100)}\n")
    _write(base / "aug_test_localvarrenaming.log", "{'MAP@R': np.float64(0.4900)}\n")
    _write(
        base / "aug_test_addassignment2equalassignment.log",
        "{'MAP@R': np.float64(0.4700)}\n",
    )
    _write(base / "aug_test_reverseifelse.log", "{'MAP@R': np.float64(0.4500)}\n")

    df = build_per_operator_table(
        results_root, "Python800", _collect_per_op_clone_row, 2
    )
    row = df.iloc[0]
    assert row["Model"] == "inv-codebert"
    assert row["Original"] == "50.00"
    assert row["All (cum.)"] == "41.00"
    assert row["VarRe"] == "49.00"
    assert row["AA2EA"] == "47.00"
    assert row["RevIf"] == "45.00"
    assert row["F2W"] == "n/a"
    assert row["W2F"] == "n/a"
    assert row["PP2AA"] == "n/a"


def test_build_per_operator_table_cls_poj104_subset(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    base = results_root / "inv-codebert" / "Code-classification-POJ104"
    metric = (
        "02/16/2026 00:00:00 - INFO - __main__ -   ***** Test results *****\n"
        "02/16/2026 00:00:00 - INFO - __main__ -   test_acc = {score}\n"
    )
    _write(base / "test_train.log", metric.format(score="0.8000"))
    _write(base / "aug_test.log", metric.format(score="0.7000"))
    _write(base / "aug_test_localvarrenaming.log", metric.format(score="0.7800"))
    _write(base / "aug_test_for2while.log", metric.format(score="0.7600"))
    _write(base / "aug_test_while2for.log", metric.format(score="0.7700"))
    _write(base / "aug_test_pp2addassignment.log", metric.format(score="0.7400"))
    _write(
        base / "aug_test_addassignment2equalassignment.log",
        metric.format(score="0.7900"),
    )
    _write(base / "aug_test_reverseifelse.log", metric.format(score="0.7500"))

    df = build_per_operator_table(results_root, "POJ104", _collect_per_op_cls_row, 2)
    row = df.iloc[0]
    assert row["Model"] == "inv-codebert"
    assert row["Original"] == "80.00"
    assert row["All (cum.)"] == "70.00"
    assert row["VarRe"] == "78.00"
    assert row["F2W"] == "76.00"
    assert row["W2F"] == "77.00"
    assert row["PP2AA"] == "74.00"
    assert row["AA2EA"] == "79.00"
    assert row["RevIf"] == "75.00"
