import torch
import torch.nn.functional as F
from transformers import DataCollatorForLanguageModeling, RobertaTokenizerFast

from modeling.dataloader import grouped_contra_data_collator
from modeling.model import grouped_contrastive_loss, info_nce_loss
from modeling.pretrain import compute_function_id, regroup_dataset

# ---------------------------------------------------------------------------
# grouped_contrastive_loss tests
# ---------------------------------------------------------------------------


class TestGroupedContrastiveLoss:
    def test_identical_positives_low_loss(self):
        """When all augs are copies of their anchor, loss should be low."""
        B, D, max_K = 4, 128, 2
        anchors = F.normalize(torch.randn(B, D), dim=1)
        # Each anchor's augs are copies of itself
        aug_list = []
        for i in range(B):
            for _ in range(max_K):
                aug_list.append(anchors[i])
        aug_embeddings = torch.stack(aug_list)  # [B*max_K, D]
        group_sizes = torch.tensor([max_K] * B)

        loss = grouped_contrastive_loss(anchors, aug_embeddings, group_sizes)
        assert loss.item() >= 0.0
        assert loss.item() < 1.0

    def test_no_augs_returns_zero(self):
        """All group_sizes=0 yields loss=0."""
        B, D, max_K = 3, 64, 2
        anchors = F.normalize(torch.randn(B, D), dim=1)
        aug_embeddings = torch.zeros(B * max_K, D)
        group_sizes = torch.tensor([0, 0, 0])

        loss = grouped_contrastive_loss(anchors, aug_embeddings, group_sizes)
        assert loss.item() == 0.0

    def test_gradient_flows(self):
        """Loss should produce finite gradients."""
        B, D, max_K = 4, 64, 3
        anchors = torch.randn(B, D, requires_grad=True)
        augs = torch.randn(B * max_K, D, requires_grad=True)
        group_sizes = torch.tensor([3, 2, 1, 3])

        loss = grouped_contrastive_loss(anchors, augs, group_sizes, temperature=0.1)
        loss.backward()
        assert anchors.grad is not None
        assert torch.isfinite(anchors.grad).all()
        assert augs.grad is not None
        assert torch.isfinite(augs.grad).all()

    def test_numerical_stability_large_logits(self):
        """With very large embedding values, loss should still be finite."""
        B, D, max_K = 4, 64, 2
        anchors = torch.randn(B, D) * 100
        augs = torch.randn(B * max_K, D) * 100
        group_sizes = torch.tensor([2, 2, 2, 2])

        loss = grouped_contrastive_loss(anchors, augs, group_sizes, temperature=0.01)
        assert torch.isfinite(loss)

    def test_variable_group_sizes(self):
        """Mix of different aug counts per anchor."""
        B, D, max_K = 4, 64, 4
        anchors = F.normalize(torch.randn(B, D), dim=1)
        augs = F.normalize(torch.randn(B * max_K, D), dim=1)
        group_sizes = torch.tensor([1, 4, 2, 3])

        loss = grouped_contrastive_loss(anchors, augs, group_sizes, temperature=0.1)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

    def test_equivalent_to_infonce_when_k1(self):
        """With exactly 1 aug per anchor, grouped loss ~ InfoNCE."""
        torch.manual_seed(42)
        B, D = 8, 64
        query = F.normalize(torch.randn(B, D), dim=1)
        key = F.normalize(torch.randn(B, D), dim=1)

        infonce = info_nce_loss(query, key, temperature=0.1)

        # Grouped: max_K=1, group_sizes all 1
        group_sizes = torch.ones(B, dtype=torch.long)
        grouped = grouped_contrastive_loss(query, key, group_sizes, temperature=0.1)

        # Not exactly equal (grouped includes anchor-to-anchor terms in denom),
        # but should be in the same ballpark
        assert abs(infonce.item() - grouped.item()) / max(infonce.item(), 1e-6) < 1.0

    def test_single_anchor_still_works(self):
        """Edge case: B=1 should not crash (no negatives from other groups)."""
        B, D, max_K = 1, 64, 3
        anchors = F.normalize(torch.randn(B, D), dim=1)
        augs = F.normalize(torch.randn(B * max_K, D), dim=1)
        group_sizes = torch.tensor([3])

        loss = grouped_contrastive_loss(anchors, augs, group_sizes, temperature=0.1)
        assert torch.isfinite(loss)

    def test_partial_group_padding_ignored(self):
        """Padding positions (beyond group_size) should not affect loss."""
        torch.manual_seed(123)
        B, D = 3, 64
        anchors = F.normalize(torch.randn(B, D), dim=1)

        # max_K=3: group 0 has 2 real augs, group 1 has 1, group 2 has 3
        max_K = 3
        augs = F.normalize(torch.randn(B * max_K, D), dim=1)
        group_sizes = torch.tensor([2, 1, 3])
        loss1 = grouped_contrastive_loss(anchors, augs, group_sizes, temperature=0.1)

        # Change padding positions (index 2 for group 0, indices 4-5 for group 1)
        augs_modified = augs.clone()
        augs_modified[2] = torch.randn(D)  # group 0, slot 2 (padding)
        augs_modified[4] = torch.randn(D)  # group 1, slot 1 (padding)
        augs_modified[5] = torch.randn(D)  # group 1, slot 2 (padding)
        loss2 = grouped_contrastive_loss(
            anchors, augs_modified, group_sizes, temperature=0.1
        )

        assert torch.isclose(loss1, loss2, atol=1e-6)


# ---------------------------------------------------------------------------
# grouped_contra_data_collator tests
# ---------------------------------------------------------------------------


def _make_grouped_feature(seq_len, aug_counts):
    """Create a mock grouped feature dict for testing the collator."""
    features = []
    for n_augs in aug_counts:
        f = {
            "code_input_ids": [1] + [100] * (seq_len - 2) + [2],
            "code_attention_mask": [1] * seq_len,
            "code_special_tokens_mask": [1] + [0] * (seq_len - 2) + [1],
            "aug_input_ids_list": [],
            "aug_attention_mask_list": [],
            "aug_special_tokens_mask_list": [],
            "function_id": hash(str(len(features))) & 0x7FFFFFFFFFFFFFFF,
        }
        for k in range(n_augs):
            f["aug_input_ids_list"].append([1] + [200 + k] * (seq_len - 2) + [2])
            f["aug_attention_mask_list"].append([1] * seq_len)
            f["aug_special_tokens_mask_list"].append([1] + [0] * (seq_len - 2) + [1])
        features.append(f)
    return features


class TestGroupedContraDataCollator:
    def _get_mlm_collator(self):
        tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/codebert-base")
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        )

    def test_output_shapes(self):
        """Verify correct tensor shapes for code and aug batches."""
        mlm_collator = self._get_mlm_collator()
        seq_len = 16
        features = _make_grouped_feature(seq_len, aug_counts=[2, 3])
        batch = grouped_contra_data_collator(mlm_collator, 6, features)

        B = 2
        max_K = 3  # max(2, 3)
        assert batch["code_input_ids"].shape == (B, seq_len)
        assert batch["aug_input_ids"].shape == (B * max_K, seq_len)
        assert batch["group_sizes"].shape == (B,)
        assert batch["group_sizes"].tolist() == [2, 3]

    def test_padding_has_no_mlm_labels(self):
        """Padding aug slots should have all labels=-100 (no MLM masking)."""
        mlm_collator = self._get_mlm_collator()
        seq_len = 16
        # Group 0: 1 aug, Group 1: 3 augs → max_K=3, group 0 has 2 padding slots
        features = _make_grouped_feature(seq_len, aug_counts=[1, 3])
        batch = grouped_contra_data_collator(mlm_collator, 6, features)

        # Group 0's padding slots are indices 1 and 2 in the flattened aug batch
        # (group 0 occupies slots 0..2, real=1, padding=slots 1,2)
        padding_labels_1 = batch["aug_labels"][1]
        padding_labels_2 = batch["aug_labels"][2]
        assert (padding_labels_1 == -100).all()
        assert (padding_labels_2 == -100).all()

    def test_group_sizes_correct(self):
        """group_sizes should reflect actual aug counts."""
        mlm_collator = self._get_mlm_collator()
        features = _make_grouped_feature(16, aug_counts=[1, 2, 4])
        batch = grouped_contra_data_collator(mlm_collator, 6, features)
        assert batch["group_sizes"].tolist() == [1, 2, 4]

    def test_max_num_augs_truncation(self):
        """Features with more augs than max_num_augs get truncated."""
        mlm_collator = self._get_mlm_collator()
        features = _make_grouped_feature(16, aug_counts=[5, 3])
        batch = grouped_contra_data_collator(mlm_collator, 2, features)

        B = 2
        max_K = 2
        assert batch["aug_input_ids"].shape == (B * max_K, 16)
        assert batch["group_sizes"].tolist() == [2, 2]

    def test_function_id_passed_through(self):
        """function_id should be present in the batch."""
        mlm_collator = self._get_mlm_collator()
        features = _make_grouped_feature(16, aug_counts=[2, 1])
        batch = grouped_contra_data_collator(mlm_collator, 6, features)
        assert "function_id" in batch
        assert batch["function_id"].shape == (2,)


# ---------------------------------------------------------------------------
# regroup_dataset tests
# ---------------------------------------------------------------------------


class TestRegroupDataset:
    def _make_flat_dataset(self):
        """Create a mock flat HF dataset with multiple rows per function."""
        from datasets import Dataset

        data = {
            "repo": ["r1", "r1", "r1", "r2", "r2"],
            "func_name": ["f1", "f1", "f1", "f2", "f2"],
            "language": ["python"] * 5,
            "code": [
                "def foo(): pass",
                "def foo(): pass",
                "def foo(): pass",
                "def bar(): pass",
                "def bar(): pass",
            ],
            "docstring": [""] * 5,
            "transformed": [
                "def foo_v1(): pass",
                "def foo_v2(): pass",
                "def foo_v3(): pass",
                "def bar_v1(): pass",
                "def bar_v2(): pass",
            ],
            "aug_type": [
                "LocalVarRenaming",
                "ReverseIfElse",
                "AddAssignment2EqualAssignment",
                "LocalVarRenaming",
                "ReverseIfElse",
            ],
        }
        return Dataset.from_dict(data)

    def test_groups_by_code(self):
        """Rows with the same code should be grouped together."""
        dataset = self._make_flat_dataset()
        grouped = regroup_dataset(dataset, max_num_augs=6)
        assert len(grouped) == 2  # two distinct functions

    def test_preserves_all_augmentations(self):
        """All augmentations should be preserved in the grouped output."""
        dataset = self._make_flat_dataset()
        grouped = regroup_dataset(dataset, max_num_augs=6)

        # Find the group with 3 augs (foo) and 2 augs (bar)
        total_real_augs = sum(grouped["num_augs"])
        assert total_real_augs == 5  # 3 + 2

    def test_padding_to_max_num_augs(self):
        """transformed_list should be padded to max_num_augs with empty strings."""
        dataset = self._make_flat_dataset()
        grouped = regroup_dataset(dataset, max_num_augs=4)

        for transforms in grouped["transformed_list"]:
            assert len(transforms) == 4

    def test_truncation(self):
        """Groups with more augs than max_num_augs get truncated."""
        dataset = self._make_flat_dataset()
        grouped = regroup_dataset(dataset, max_num_augs=2)

        for num in grouped["num_augs"]:
            assert num <= 2

    def test_function_id_deterministic(self):
        """function_id should match compute_function_id on the code string."""
        dataset = self._make_flat_dataset()
        grouped = regroup_dataset(dataset, max_num_augs=6)

        for i in range(len(grouped)):
            expected_fid = compute_function_id(grouped["code"][i])
            assert grouped["function_id"][i] == expected_fid
