from enum import Enum

import torch
import torch.nn.functional as F
from .common import DEVICE
from transformers import Trainer


class ContraType(str, Enum):
    INFO_NCE = "info_nce"
    BARLOW_TWINS = "barlow_twins"


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


class ContrastiveTrainer(Trainer):
    def __init__(self, alpha=1.0, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # Move inputs to device
        code_input_ids = inputs["code_input_ids"].to(DEVICE)
        code_attention_mask = inputs["code_attention_mask"].to(DEVICE)
        code_labels = inputs["code_labels"].to(DEVICE)
        aug_input_ids = inputs["aug_input_ids"].to(DEVICE)
        aug_attention_mask = inputs["aug_attention_mask"].to(DEVICE)
        aug_labels = inputs["aug_labels"].to(DEVICE)

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

        # compute MLM loss for code and augmentation separately
        code_mlm_loss = code_outputs.loss
        aug_mlm_loss = aug_outputs.loss
        mlm_loss = code_mlm_loss + aug_mlm_loss

        # Compute contrastive loss between code and its augmentation
        contrastive_loss = info_nce_loss(
            code_embeddings,
            aug_embeddings,
            self.temperature,
        )

        # Total loss with weighting (adjust alpha as needed)
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
