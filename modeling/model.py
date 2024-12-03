import torch
import torch.nn.functional as F
from transformers import Trainer
from common import DEVICE


def info_nce_loss(query, key, temperature=0.07):
    device = query.device
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    logits = torch.matmul(query, key.transpose(-1, -2)) / temperature
    labels = torch.arange(query.size(0)).long().to(device)
    loss = F.cross_entropy(logits, labels)
    return loss


class ContraBERTTrainer(Trainer):
    def __init__(self, alpha=1.0, device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

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

        # Concatenate inputs for MLM
        input_ids = torch.cat([code_input_ids, aug_input_ids], dim=0)
        attention_mask = torch.cat([code_attention_mask, aug_attention_mask], dim=0)
        labels = torch.cat([code_labels, aug_labels], dim=0)

        # Forward pass for MLM
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        # Compute MLM loss
        mlm_loss = outputs.loss

        # Get embeddings (CLS token)
        hidden_states = outputs.hidden_states[
            -1
        ]  # [2*batch_size, seq_len, hidden_size]
        cls_embeddings = hidden_states[:, 0, :]  # [2*batch_size, hidden_size]

        # Split embeddings
        batch_size = code_input_ids.size(0)
        code_embeddings = cls_embeddings[:batch_size]
        aug_embeddings = cls_embeddings[batch_size:]

        # Compute contrastive loss between code and its augmentation
        contrastive_loss = info_nce_loss(code_embeddings, aug_embeddings)

        # Total loss with weighting (adjust alpha as needed)
        total_loss = mlm_loss + self.alpha * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss
