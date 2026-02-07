import torch
import torch.nn.functional as F
from transformers import Trainer

from ._types import ContraMode


def info_nce_loss(query, key, temperature=0.07):
    device = query.device
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    logits = torch.matmul(query, key.transpose(-1, -2)) / temperature
    labels = torch.arange(query.size(0)).long().to(device)
    loss = F.cross_entropy(logits, labels)
    return loss


def barlow_twins_loss(query, key, lambda_param=0.005):
    # Normalize representations along batch dimension
    query = (query - query.mean(dim=0)) / query.std(dim=0)
    key = (key - key.mean(dim=0)) / key.std(dim=0)

    N = query.size(0)

    # Cross-correlation matrix
    c = torch.mm(query.T, key) / N

    # Loss calculation
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambda_param * off_diag
    return loss


def off_diagonal(x):
    # Returns the off-diagonal elements of a square matrix
    n, _ = x.shape
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def build_positive_mask(function_ids: torch.Tensor) -> torch.Tensor:
    """Build a boolean mask where mask[i, j] = True iff function_ids[i] == function_ids[j] and i != j.

    Args:
        function_ids: tensor of shape [N] with integer function identifiers.

    Returns:
        Boolean tensor of shape [N, N].
    """
    mask = function_ids.unsqueeze(0) == function_ids.unsqueeze(1)
    mask.fill_diagonal_(False)
    return mask


def supcon_loss(
    embeddings: torch.Tensor,
    function_ids: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    For each anchor i, all j with same function_id are positives (excl. self).
    Anchors with no positives contribute zero to the loss.

    Args:
        embeddings: [N, D] embeddings (code + aug concatenated).
        function_ids: [N] integer function IDs.
        temperature: scalar temperature.
        eps: small constant for numerical stability.

    Returns:
        Scalar loss averaged over anchors that have at least one positive.
    """
    device = embeddings.device
    N = embeddings.size(0)

    embeddings = F.normalize(embeddings, dim=1)

    # Pairwise cosine similarities scaled by temperature: [N, N]
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Log-sum-exp stability: subtract row-wise max (excluding self)
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim_for_max = sim_matrix.masked_fill(self_mask, float("-inf"))
    max_sim, _ = sim_for_max.max(dim=1, keepdim=True)
    max_sim = max_sim.clamp(min=0.0)

    exp_sim = torch.exp(sim_matrix - max_sim)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)

    # Denominator: sum over all j != i
    denom = exp_sim.sum(dim=1, keepdim=True) + eps

    # Log probabilities
    log_prob = torch.log(exp_sim / denom + eps)

    # Positive mask
    pos_mask = build_positive_mask(function_ids)

    num_positives = pos_mask.float().sum(dim=1)
    has_positive = num_positives > 0
    num_positives = num_positives.clamp(min=1.0)

    # Mean log-prob over positives per anchor
    masked_log_prob = log_prob * pos_mask.float()
    per_anchor_loss = -masked_log_prob.sum(dim=1) / num_positives

    if has_positive.any():
        loss = per_anchor_loss[has_positive].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


def grouped_contrastive_loss(
    anchor_embeddings: torch.Tensor,
    aug_embeddings: torch.Tensor,
    group_sizes: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Grouped multi-key contrastive loss.

    Each anchor has a variable number of augmentation positives (given by
    ``group_sizes``).  Negatives are all other anchors and their augmentations.

    Uses per-positive log-prob averaging (SupCon-style) with log-sum-exp
    stabilization.

    Args:
        anchor_embeddings: ``[B, D]`` CLS embeddings of anchor codes.
        aug_embeddings: ``[B * max_K, D]`` CLS embeddings of flattened
            augmentations (padded groups have zero-vectors).
        group_sizes: ``[B]`` number of real augmentations per anchor.
        temperature: contrastive temperature.
        eps: numerical stability constant.

    Returns:
        Scalar loss averaged over anchors that have at least one augmentation.
    """
    device = anchor_embeddings.device
    B = anchor_embeddings.size(0)
    total_augs = aug_embeddings.size(0)
    max_K = total_augs // B

    # Normalize
    anchor_embeddings = F.normalize(anchor_embeddings, dim=1)  # [B, D]
    aug_embeddings = F.normalize(aug_embeddings, dim=1)  # [B*max_K, D]

    # Reshape aug embeddings to [B, max_K, D]
    aug_reshaped = aug_embeddings.view(B, max_K, -1)

    # Validity mask: [B, max_K] — True for real augmentations
    arange_K = torch.arange(max_K, device=device).unsqueeze(0)  # [1, max_K]
    valid_mask = arange_K < group_sizes.unsqueeze(1)  # [B, max_K]

    # --- Similarity matrices ---
    # anchor-to-anchor: [B, B]
    sim_a2a = torch.matmul(anchor_embeddings, anchor_embeddings.T) / temperature
    # anchor-to-all-augs: [B, B*max_K]
    sim_a2aug = torch.matmul(anchor_embeddings, aug_embeddings.T) / temperature

    # --- Build denominator exclusion mask [B, B + B*max_K] ---
    # Exclude: self-anchor (diagonal of a2a) + padding aug positions
    all_logits = torch.cat([sim_a2a, sim_a2aug], dim=1)  # [B, B + B*max_K]

    # Self-anchor exclusion
    self_anchor_mask = torch.eye(B, dtype=torch.bool, device=device)  # [B, B]

    # Padding aug exclusion: [B, B*max_K]
    aug_valid_global = valid_mask.reshape(-1)  # [B*max_K]
    aug_invalid_mask = ~aug_valid_global.unsqueeze(0).expand(B, -1)  # [B, B*max_K]

    denom_exclude = torch.cat([self_anchor_mask, aug_invalid_mask], dim=1)

    # Log-sum-exp stability
    max_logit = (
        all_logits.masked_fill(denom_exclude, float("-inf"))
        .max(dim=1, keepdim=True)
        .values
    )
    max_logit = max_logit.clamp(min=0.0)

    exp_logits = torch.exp(all_logits - max_logit)
    exp_logits = exp_logits.masked_fill(denom_exclude, 0.0)
    denom = exp_logits.sum(dim=1, keepdim=True) + eps  # [B, 1]

    # --- Positive logits: anchor i vs its own augmentations ---
    # sim_a2aug reshaped to [B, B, max_K] — index [i, i, :] = anchor i's augs
    sim_a2aug_grouped = sim_a2aug.view(B, B, max_K)
    pos_logits = sim_a2aug_grouped[
        torch.arange(B, device=device), torch.arange(B, device=device), :
    ]  # [B, max_K]

    # Per-positive log-prob
    exp_pos = torch.exp(pos_logits - max_logit)  # [B, max_K]
    log_prob_pos = torch.log(exp_pos / denom + eps)  # [B, max_K]
    log_prob_pos = log_prob_pos.masked_fill(~valid_mask, 0.0)

    # Average over valid positives per anchor, then over anchors
    per_anchor_loss = -log_prob_pos.sum(dim=1) / group_sizes.float().clamp(min=1.0)

    has_augs = group_sizes > 0
    if has_augs.any():
        loss = per_anchor_loss[has_augs].mean()
    else:
        loss = torch.tensor(0.0, device=device, requires_grad=True)

    return loss


class ContrastiveTrainer(Trainer):
    def __init__(
        self, alpha=1.0, temperature=0.07, contra_mode="info_nce", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.contra_mode = ContraMode(contra_mode)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.contra_mode == ContraMode.GROUPED:
            return self._compute_grouped_loss(model, inputs, return_outputs)

        # Move inputs to the device the model lives on (supports DDP).
        # Clone input_ids tensors to avoid in-place modification conflicts
        # during backward when the MLM head or embeddings share storage.
        device = model.device
        code_input_ids = inputs["code_input_ids"].to(device).clone()
        code_attention_mask = inputs["code_attention_mask"].to(device)
        code_labels = inputs["code_labels"].to(device)
        aug_input_ids = inputs["aug_input_ids"].to(device).clone()
        aug_attention_mask = inputs["aug_attention_mask"].to(device)
        aug_labels = inputs["aug_labels"].to(device)

        # Forward pass for MLM
        # use bi-encoder training, encode code and augmentation separately using self.model
        code_outputs = model(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
            labels=code_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        code_hidden_states = code_outputs.hidden_states[-1]
        code_embeddings = code_hidden_states[:, 0, :]

        aug_outputs = model(
            input_ids=aug_input_ids,
            attention_mask=aug_attention_mask,
            labels=aug_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        aug_hidden_states = aug_outputs.hidden_states[-1]
        aug_embeddings = aug_hidden_states[:, 0, :]

        # Average MLM losses so the combined MLM term is on the same scale
        # as the single contrastive term (~3-8 each), letting alpha express
        # a genuine preference rather than compensating for a 2x scale artifact.
        code_mlm_loss = code_outputs.loss
        aug_mlm_loss = aug_outputs.loss
        mlm_loss = (code_mlm_loss + aug_mlm_loss) / 2

        # Compute contrastive loss between code and its augmentation
        if self.contra_mode == ContraMode.SUPCON:
            all_embeddings = torch.cat([code_embeddings, aug_embeddings], dim=0)
            function_ids = inputs["function_id"].to(device)
            all_function_ids = torch.cat([function_ids, function_ids], dim=0)
            contrastive_loss = supcon_loss(
                all_embeddings, all_function_ids, self.temperature
            )
        else:
            contrastive_loss = info_nce_loss(
                code_embeddings,
                aug_embeddings,
                self.temperature,
            )

        # Total loss with weighting (adjust alpha as needed)
        total_loss = mlm_loss + self.alpha * contrastive_loss

        return (total_loss, code_outputs) if return_outputs else total_loss

    def _compute_grouped_loss(self, model, inputs, return_outputs=False):
        """Compute loss for grouped multi-key contrast mode.

        Inputs contain:
          - code_input_ids: [B, seq_len]
          - code_attention_mask, code_labels: same shape
          - aug_input_ids: [B * max_K, seq_len]
          - aug_attention_mask, aug_labels: same shape
          - group_sizes: [B]
        """
        device = model.device
        code_input_ids = inputs["code_input_ids"].to(device).clone()
        code_attention_mask = inputs["code_attention_mask"].to(device)
        code_labels = inputs["code_labels"].to(device)
        aug_input_ids = inputs["aug_input_ids"].to(device).clone()
        aug_attention_mask = inputs["aug_attention_mask"].to(device)
        aug_labels = inputs["aug_labels"].to(device)
        group_sizes = inputs["group_sizes"].to(device)

        # Forward anchor
        code_outputs = model(
            input_ids=code_input_ids,
            attention_mask=code_attention_mask,
            labels=code_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        code_embeddings = code_outputs.hidden_states[-1][:, 0, :]  # [B, D]

        # Forward all augmentations (flattened [B*max_K, seq_len])
        aug_outputs = model(
            input_ids=aug_input_ids,
            attention_mask=aug_attention_mask,
            labels=aug_labels,
            output_hidden_states=True,
            return_dict=True,
        )
        aug_embeddings = aug_outputs.hidden_states[-1][:, 0, :]  # [B*max_K, D]

        # MLM loss (padding augs have labels=-100, contribute 0)
        mlm_loss = (code_outputs.loss + aug_outputs.loss) / 2

        # Contrastive loss
        contrastive_loss = grouped_contrastive_loss(
            code_embeddings, aug_embeddings, group_sizes, self.temperature
        )

        total_loss = mlm_loss + self.alpha * contrastive_loss

        return (total_loss, code_outputs) if return_outputs else total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override the default prediction_step to handle custom inputs during evaluation.
        """
        # Move inputs to device
        device = self.args.device
        code_input_ids = inputs["code_input_ids"].to(device)
        code_attention_mask = inputs["code_attention_mask"].to(device)
        code_labels = inputs["code_labels"].to(device)

        # Prepare inputs for the model
        # Since evaluation usually focuses on the MLM task, we can use code inputs
        inputs_for_model = {
            "input_ids": code_input_ids,
            "attention_mask": code_attention_mask,
            "labels": code_labels,
        }

        with torch.no_grad():
            outputs = model(**inputs_for_model)

            if prediction_loss_only:
                loss = outputs.loss
                return (loss, None, None)
            else:
                loss = outputs.loss
                logits = outputs.logits
                return (loss, logits, code_labels)
