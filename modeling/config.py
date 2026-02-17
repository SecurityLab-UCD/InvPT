from dataclasses import dataclass, field
from pathlib import Path

import dacite
import yaml

from ._types import ContraMode, ModelType
from .common import default_num_proc

_DACITE_CONFIG = dacite.Config(cast=[ContraMode, ModelType])


@dataclass
class PretrainConfig:
    """Configuration for InvPT pre-training experiments."""

    dataset_path: str = "data/csn_jp.jsonl"
    model_name: str = "microsoft/codebert-base"
    batch_size: int = 256
    num_epochs: int = 10
    gradient_accumulation_steps: int = 1
    num_proc: int = field(default_factory=default_num_proc)
    seed: int = 0
    run_name: str = "InvariantBERT"
    learning_rate: float = 2e-4
    resume: bool = False
    alpha: float = 1.0
    mlm_weight: float = 1.0
    temperature: float = 0.07
    max_seq_length: int = 256
    sample_rate: float = 1.0
    checkpoint: str | None = None
    tokenizer_name: str | None = None
    contra_mode: ContraMode = ContraMode.INFO_NCE
    max_num_augs: int = 6
    self_contrast: bool = True
    model_type: ModelType = ModelType.ROBERTA
    pooling: str = "cls"
    include_nl: bool = False


def load_config(path: str | Path) -> PretrainConfig:
    """Load a PretrainConfig from a YAML file.

    Missing keys use dataclass defaults.

    Raises:
        FileNotFoundError: If the config file does not exist.
        dacite.DaciteError: If the YAML contains invalid fields or types.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}

    return dacite.from_dict(
        data_class=PretrainConfig,
        data=data,
        config=_DACITE_CONFIG,
    )
