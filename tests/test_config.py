from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from modeling._types import ContraMode
from modeling.config import PretrainConfig, load_config, merge_cli_overrides


def _write_yaml(data: dict | None, suffix: str = ".yaml") -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        if data is not None:
            yaml.dump(data, f)
        return f.name


class TestLoadConfig:
    def test_full_config(self) -> None:
        path = _write_yaml(
            {
                "dataset_path": "data/test.jsonl",
                "model_name": "test-model",
                "batch_size": 32,
                "num_epochs": 1,
                "gradient_accumulation_steps": 2,
                "num_proc": 4,
                "seed": 42,
                "run_name": "test-run",
                "learning_rate": 1e-5,
                "resume": True,
                "alpha": 0.5,
                "temperature": 0.1,
                "max_seq_length": 128,
                "sample_rate": 0.5,
                "contra_mode": "supcon",
            }
        )
        config = load_config(path)
        assert config.batch_size == 32
        assert config.contra_mode == ContraMode.SUPCON
        assert config.resume is True
        assert config.seed == 42

    def test_partial_config_uses_defaults(self) -> None:
        path = _write_yaml({"run_name": "minimal"})
        config = load_config(path)
        assert config.run_name == "minimal"
        assert config.batch_size == 256  # default
        assert config.contra_mode == ContraMode.INFO_NCE  # default

    def test_empty_yaml_uses_all_defaults(self) -> None:
        path = _write_yaml(None)
        config = load_config(path)
        assert config == PretrainConfig()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_contra_mode_enum_casting(self) -> None:
        for mode in ["info_nce", "supcon", "grouped"]:
            path = _write_yaml({"contra_mode": mode})
            config = load_config(path)
            assert isinstance(config.contra_mode, ContraMode)


class TestMergeCliOverrides:
    def test_overrides_applied(self) -> None:
        config = PretrainConfig(seed=0, batch_size=64)
        merged = merge_cli_overrides(config, {"seed": 42, "batch_size": None})
        assert merged.seed == 42
        assert merged.batch_size == 64  # None means not provided

    def test_config_key_ignored(self) -> None:
        config = PretrainConfig()
        merged = merge_cli_overrides(
            config, {"config": "experiments/base.yaml", "seed": 99}
        )
        assert merged.seed == 99
        assert not hasattr(merged, "config")

    def test_no_overrides(self) -> None:
        config = PretrainConfig(seed=7)
        merged = merge_cli_overrides(config, {"config": None})
        assert merged.seed == 7
