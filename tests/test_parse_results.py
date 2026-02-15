from __future__ import annotations

from pathlib import Path

from experiments_downstream.parse_results import collect_results, render_table


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_collect_results_clone_and_classification(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "test.log",
        "{'MAP@R': np.float64(0.5621)}\n",
    )
    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "aug_test.log",
        "{'MAP@R': np.float64(0.4321)}\n",
    )

    test_train_log = """
    02/16/2026 00:00:00 - INFO - __main__ -   ***** Test results *****
    02/16/2026 00:00:00 - INFO - __main__ -   test_acc = 0.3992
    """
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

    results = collect_results(results_root)
    assert len(results) == 2
    clone = next(r for r in results if r.task == "Clone-detection-POJ104")
    cls = next(r for r in results if r.task == "Code-classification-CodeNet")

    assert clone.scores.regular == 0.5621
    assert clone.scores.augmented == 0.4321
    assert cls.scores.regular == 0.3992
    assert cls.scores.augmented == 0.2777


def test_render_table_placeholder_for_missing(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _write(
        results_root / "inv-codebert" / "Clone-detection-POJ104" / "test.log",
        "{'MAP@R': np.float64(0.5621)}\n",
    )

    table = render_table(collect_results(results_root), 4)
    assert "-" in table
