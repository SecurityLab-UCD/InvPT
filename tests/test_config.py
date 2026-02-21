import tempfile

import pytest
import yaml
from typer.testing import CliRunner

from modeling.cli import app
from modeling.config import PretrainConfig, load_config

runner = CliRunner()


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
            }
        )
        config = load_config(path)
        assert config.batch_size == 32
        assert config.resume is True
        assert config.seed == 42

    def test_partial_config_uses_defaults(self) -> None:
        path = _write_yaml({"run_name": "minimal"})
        config = load_config(path)
        assert config.run_name == "minimal"
        assert config.batch_size == 256  # default

    def test_empty_yaml_uses_all_defaults(self) -> None:
        path = _write_yaml(None)
        config = load_config(path)
        assert config == PretrainConfig()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")


class TestCli:
    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "pretrain" in result.output

    def test_run_help(self) -> None:
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "CONFIG" in result.output

    def test_run_missing_config(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_pretrain_help(self) -> None:
        result = runner.invoke(app, ["pretrain", "--help"])
        assert result.exit_code == 0
        assert "--batch-size" in result.output
