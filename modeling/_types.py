from enum import Enum


class ContraMode(str, Enum):
    INFO_NCE = "info_nce"
    SUPCON = "supcon"
    GROUPED = "grouped"


class ModelType(str, Enum):
    ROBERTA = "roberta"
    MODERNBERT = "modernbert"
