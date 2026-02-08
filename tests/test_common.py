import os

from modeling.common import default_num_proc


def test_default_num_proc_returns_cpu_count(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: 123)
    assert default_num_proc() == 123


def test_default_num_proc_is_at_least_one(monkeypatch) -> None:
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert default_num_proc() == 1
