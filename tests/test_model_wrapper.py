"""Tests for SplitHeadWrapper architecture dispatch and pooling."""

import pytest
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from modeling.model import ContrastiveTrainer, SplitHeadWrapper

# ---------------------------------------------------------------------------
# SplitHeadWrapper with RoBERTa
# ---------------------------------------------------------------------------


def _make_roberta_wrapper(
    vocab_size: int = 64,
    hidden_size: int = 32,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    intermediate_size: int = 64,
    max_position_embeddings: int = 32,
) -> SplitHeadWrapper:
    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    mlm = RobertaForMaskedLM(config)
    return SplitHeadWrapper(mlm)


class TestSplitHeadWrapperRoberta:
    def test_forward_produces_loss_and_hidden(self) -> None:
        wrapper = _make_roberta_wrapper()
        B, seq = 2, 8
        input_ids = torch.randint(0, 64, (2 * B, seq))
        attention_mask = torch.ones_like(input_ids)
        labels_a = torch.randint(0, 64, (B, seq))
        labels_b = torch.randint(0, 64, (B, seq))

        mlm_loss, last_hidden = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_a=labels_a,
            labels_b=labels_b,
            split_at=B,
        )
        assert mlm_loss.shape == ()
        assert last_hidden.shape == (2 * B, seq, 32)

    def test_encoder_dispatch(self) -> None:
        wrapper = _make_roberta_wrapper()
        encoder = wrapper._get_encoder()
        assert encoder is wrapper.mlm_model.roberta

    def test_lm_head_dispatch(self) -> None:
        wrapper = _make_roberta_wrapper()
        dummy = torch.randn(1, 4, 32)
        logits = wrapper._apply_lm_head(dummy)
        assert logits.shape == (1, 4, 64)  # vocab_size=64


# ---------------------------------------------------------------------------
# SplitHeadWrapper with ModernBERT (skipped if transformers < 4.48)
# ---------------------------------------------------------------------------

_has_modernbert = True
try:
    from transformers import ModernBertConfig, ModernBertForMaskedLM
except ImportError:
    _has_modernbert = False

requires_modernbert = pytest.mark.skipif(
    not _has_modernbert,
    reason="ModernBERT requires transformers >= 4.48",
)


def _make_modernbert_wrapper(
    vocab_size: int = 64,
    hidden_size: int = 32,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 2,
    intermediate_size: int = 64,
    max_position_embeddings: int = 32,
) -> SplitHeadWrapper:
    config = ModernBertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    mlm = ModernBertForMaskedLM(config)
    return SplitHeadWrapper(mlm)


@requires_modernbert
class TestSplitHeadWrapperModernBert:
    def test_forward_produces_loss_and_hidden(self) -> None:
        wrapper = _make_modernbert_wrapper()
        B, seq = 2, 8
        input_ids = torch.randint(0, 64, (2 * B, seq))
        attention_mask = torch.ones_like(input_ids)
        labels_a = torch.randint(0, 64, (B, seq))
        labels_b = torch.randint(0, 64, (B, seq))

        mlm_loss, last_hidden = wrapper(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels_a=labels_a,
            labels_b=labels_b,
            split_at=B,
        )
        assert mlm_loss.shape == ()
        assert last_hidden.shape == (2 * B, seq, 32)

    def test_encoder_dispatch(self) -> None:
        wrapper = _make_modernbert_wrapper()
        encoder = wrapper._get_encoder()
        assert encoder is wrapper.mlm_model.model

    def test_lm_head_dispatch(self) -> None:
        wrapper = _make_modernbert_wrapper()
        dummy = torch.randn(1, 4, 32)
        logits = wrapper._apply_lm_head(dummy)
        assert logits.shape == (1, 4, 64)  # vocab_size=64


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


class TestPooling:
    def test_cls_pooling(self) -> None:
        """CLS pooling extracts the first token."""
        trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
        trainer.pooling = "cls"
        hidden = torch.randn(3, 10, 32)
        mask = torch.ones(3, 10)
        pooled = trainer._pool(hidden, mask)
        assert pooled.shape == (3, 32)
        assert torch.equal(pooled, hidden[:, 0, :])

    def test_mean_pooling(self) -> None:
        """Mean pooling averages over non-padding tokens."""
        trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
        trainer.pooling = "mean"
        hidden = torch.ones(2, 4, 8)  # all ones
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.long)
        pooled = trainer._pool(hidden, mask)
        assert pooled.shape == (2, 8)
        # All non-padding tokens are 1, so mean should be 1
        assert torch.allclose(pooled, torch.ones(2, 8))

    def test_mean_pooling_ignores_padding(self) -> None:
        """Padding tokens (mask=0) should not affect mean pooling."""
        trainer = ContrastiveTrainer.__new__(ContrastiveTrainer)
        trainer.pooling = "mean"
        hidden = torch.zeros(1, 4, 2)
        hidden[0, 0, :] = 2.0  # only first token has value
        hidden[0, 1, :] = 4.0  # second token
        hidden[0, 2, :] = 99.0  # padding — should be ignored
        hidden[0, 3, :] = 99.0  # padding — should be ignored
        mask = torch.tensor([[1, 1, 0, 0]])
        pooled = trainer._pool(hidden, mask)
        # mean of [2, 4] = 3
        assert torch.allclose(pooled, torch.tensor([[3.0, 3.0]]))


# ---------------------------------------------------------------------------
# Config loading with new fields
# ---------------------------------------------------------------------------


class TestConfigModelType:
    def test_default_model_type(self) -> None:
        from modeling._types import ModelType
        from modeling.config import PretrainConfig

        cfg = PretrainConfig()
        assert cfg.model_type == ModelType.ROBERTA
        assert cfg.pooling == "cls"

    def test_modernbert_config_from_yaml(self) -> None:
        import tempfile

        import yaml

        from modeling._types import ModelType
        from modeling.config import load_config

        data = {"model_type": "modernbert", "pooling": "mean"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            path = f.name
        cfg = load_config(path)
        assert cfg.model_type == ModelType.MODERNBERT
        assert cfg.pooling == "mean"
