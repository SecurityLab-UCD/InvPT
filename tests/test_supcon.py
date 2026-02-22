import torch
import torch.nn.functional as F

from modeling.model import build_positive_mask, supcon_loss


class TestBuildPositiveMask:
    def test_basic_mask(self):
        """Items with same function_id are positives (excl diagonal)."""
        ids = torch.tensor([10, 20, 10, 20])
        mask = build_positive_mask(ids)
        expected = torch.tensor(
            [
                [False, False, True, False],
                [False, False, False, True],
                [True, False, False, False],
                [False, True, False, False],
            ]
        )
        assert torch.equal(mask, expected)

    def test_all_same_id(self):
        """All items same function_id => all off-diagonal are positives."""
        ids = torch.tensor([5, 5, 5])
        mask = build_positive_mask(ids)
        expected = ~torch.eye(3, dtype=torch.bool)
        assert torch.equal(mask, expected)

    def test_all_unique(self):
        """All unique function_ids => no positives."""
        ids = torch.tensor([1, 2, 3, 4])
        mask = build_positive_mask(ids)
        assert not mask.any()

    def test_diagonal_always_false(self):
        """Self-comparisons must always be False."""
        ids = torch.tensor([1, 1, 2, 2, 1])
        mask = build_positive_mask(ids)
        assert not mask.diagonal().any()


class TestSupConLoss:
    def test_identical_embeddings_positives_low_loss(self):
        """When all positives have identical embeddings, loss should be low."""
        e1 = F.normalize(torch.randn(1, 128), dim=1)
        e2 = F.normalize(torch.randn(1, 128), dim=1)
        embeddings = torch.cat([e1, e1, e2, e2], dim=0)
        ids = torch.tensor([0, 0, 1, 1])
        loss = supcon_loss(embeddings, ids, temperature=0.07)
        assert loss.item() >= 0.0
        assert loss.item() < 1.0

    def test_no_positives_returns_zero(self):
        """All unique function_ids => loss is 0."""
        embeddings = F.normalize(torch.randn(4, 64), dim=1)
        ids = torch.tensor([0, 1, 2, 3])
        loss = supcon_loss(embeddings, ids, temperature=0.1)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """Loss should produce finite gradients."""
        embeddings = torch.randn(6, 64, requires_grad=True)
        ids = torch.tensor([0, 0, 1, 1, 2, 2])
        loss = supcon_loss(embeddings, ids, temperature=0.1)
        loss.backward()
        assert embeddings.grad is not None
        assert torch.isfinite(embeddings.grad).all()

    def test_numerically_stable_large_logits(self):
        """With very large embedding values, loss should still be finite."""
        embeddings = torch.randn(4, 64) * 100
        ids = torch.tensor([0, 0, 1, 1])
        loss = supcon_loss(embeddings, ids, temperature=0.01)
        assert torch.isfinite(loss)

    def test_batch_with_mixed_positives(self):
        """Realistic: some functions have 3 augs, some have 1."""
        embeddings = F.normalize(torch.randn(8, 64), dim=1)
        ids = torch.tensor([100, 100, 100, 200, 200, 300, 300, 300])
        loss = supcon_loss(embeddings, ids, temperature=0.1)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0
