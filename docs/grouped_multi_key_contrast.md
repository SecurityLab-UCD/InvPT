# Grouped Multi-Key Contrastive Loss

## Motivation

In the standard InfoNCE setup (`--contra_mode info_nce`), each training example
is a pair (anchor code, one augmentation). The contrastive loss treats the
diagonal entries of the $B \times B$ similarity matrix as positives and everything
else as negatives. When the dataset contains multiple augmentations of the same
function (e.g. VarRe, RevIf, AA2EA applied to the same source), those
augmentations land in different batch rows and are **pushed apart as negatives**
even though they are semantically equivalent.

SupCon (`--contra_mode supcon`) mitigates this by masking: it checks
`function_id` at loss time and treats all same-ID embeddings as positives.
However, it relies on same-function augmentations happening to co-occur in the
same mini-batch, which becomes unlikely at small batch sizes.

Grouped Multi-Key Contrast (`--contra_mode grouped`) solves this structurally:
it **regroups the dataset** so that every batch item already bundles an anchor
with _all_ of its augmentations. Every augmentation is guaranteed to be present
as a positive, regardless of batch size.

## Algorithm

### 1. Dataset Regrouping

Before training, the flat JSONL dataset (one row per `(code, transformed)` pair)
is regrouped by `function_id = SHA-256(code)[:8]`:

```
Input (flat):
  row 0: code="def foo()...", transformed="def renamed_foo()...", aug_type=VarRe
  row 1: code="def foo()...", transformed="def foo_rev()...",     aug_type=RevIf
  row 2: code="def bar()...", transformed="def bar_v1()...",      aug_type=VarRe

Output (grouped):
  group 0: code="def foo()...", transformed_list=["def renamed_foo()...", "def foo_rev()..."]
  group 1: code="def bar()...", transformed_list=["def bar_v1()..."]
```

Each group's augmentation list is truncated to `max_num_augs` (default 6) and
padded with empty strings to a fixed length for uniform Arrow storage.

### 2. Collation

The grouped collator produces:

| Tensor           | Shape                   | Description                                           |
| ---------------- | ----------------------- | ----------------------------------------------------- |
| `code_input_ids` | $[B, L]$                | Tokenized anchors (MLM-masked)                        |
| `aug_input_ids`  | $[B \cdot K_{\max}, L]$ | Flattened tokenized augmentations (MLM-masked)        |
| `group_sizes`    | $[B]$                   | Number of real (non-padding) augmentations per anchor |

where $B$ is the batch size, $L$ is `max_seq_length`, and
$K_{\max} = \min(\max_i K_i,\; \texttt{max\_num\_augs})$.

Padding augmentation slots use `attention_mask = 0` and
`special_tokens_mask = 1` so the MLM collator assigns `labels = -100` to every
token (i.e. padding augmentations contribute zero to the MLM loss).

### 3. Forward Pass

A single shared encoder $f_\theta$ processes both anchors and augmentations:

$$\mathbf{h}_i = f_\theta(\text{code}_i) \quad \text{for } i = 1, \dots, B \qquad \to [B, D]$$

$$\mathbf{h}_{i,k} = f_\theta(\text{aug}_{i,k}) \quad \text{for } i = 1, \dots, B,\; k = 1, \dots, K_{\max} \qquad \to [B \cdot K_{\max}, D]$$

The CLS token embedding (position 0 of the last hidden layer) is used as the
sequence representation. Both forward passes also produce MLM logits for the
masked language modeling objective.

### 4. Contrastive Loss

**Definitions.** Given:

- Anchor embeddings: $\mathbf{a}_i = \text{normalize}(\mathbf{h}_i)$ for $i = 1, \dots, B$
- Augmentation embeddings: $\mathbf{v}_{i,k} = \text{normalize}(\mathbf{h}_{i,k})$ for $i = 1, \dots, B$, $k = 1, \dots, K_{\max}$
- Group sizes: $K_i$ (number of real augmentations for anchor $i$)
- Temperature: $\tau$
- Similarity function: $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v}$ (dot product of $\ell_2$-normalized vectors)

**Candidate pool.** For each anchor $i$, the candidate pool consists of:

- All other anchors $\mathbf{a}_j$ where $j \neq i$
- All valid (non-padding) augmentation embeddings $\mathbf{v}_{j,k}$ for all $j$ and $k < K_j$

**Positive set.** For anchor $i$: $\mathcal{P}(i) = \{\mathbf{v}_{i,k} : k < K_i\}$.

**Per-positive loss.** For each anchor $i$ and each of its valid positives $\mathbf{v}_{i,k}$:

$$\ell_{i,k} = -\log \frac{\exp\!\bigl(\text{sim}(\mathbf{a}_i, \mathbf{v}_{i,k}) / \tau\bigr)}{\displaystyle\sum_{\substack{j=1 \\ j \neq i}}^{B} \exp\!\bigl(\text{sim}(\mathbf{a}_i, \mathbf{a}_j) / \tau\bigr) + \sum_{j=1}^{B} \sum_{\substack{k'=0 \\ k' < K_j}}^{K_{\max}-1} \exp\!\bigl(\text{sim}(\mathbf{a}_i, \mathbf{v}_{j,k'}) / \tau\bigr)}$$

**Per-anchor loss** (SupCon-style averaging over positives):

$$\mathcal{L}_i = \frac{1}{K_i} \sum_{k=0}^{K_i - 1} \ell_{i,k}$$

**Batch loss** (averaged over anchors with at least one augmentation):

$$\mathcal{L}_{\text{grouped}} = \frac{1}{|\{i : K_i > 0\}|} \sum_{\substack{i=1 \\ K_i > 0}}^{B} \mathcal{L}_i$$

**Note:** The denominator includes the anchor's own augmentations as well. This
differs from some formulations that exclude positives from the denominator; our
version follows the SupCon convention (Khosla et al., 2020) where the
denominator sums over _all_ non-self entries.

### 5. Total Training Loss

The total loss combines MLM and contrastive objectives:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \alpha \cdot \mathcal{L}_{\text{grouped}}$$

where the MLM loss averages over anchor and augmentation views:

$$\mathcal{L}_{\text{MLM}} = \frac{\mathcal{L}_{\text{MLM}}(\text{code}) + \mathcal{L}_{\text{MLM}}(\text{aug})}{2}$$

The MLM loss for augmentations is automatically averaged only over non-padding
tokens (padding augmentations have all labels set to $-100$ and contribute zero).

### 6. Numerical Stability

The implementation uses the log-sum-exp trick for numerical stability.
Given the raw logits $z_{i,n}$ for anchor $i$ against candidate $n$:

1. Compute $m_i = \max_{n \notin \text{excluded}} z_{i,n}$, clamped to $\geq 0$
2. Subtract before exponentiation: $\exp(z_{i,n} - m_i)$
3. This prevents overflow in $\exp(\cdot)$ when similarity values are large

## Parameters

- `--contra_mode`
    - default: `info_nce`
    - Selects the contrastive loss mode. Set to `grouped` to enable grouped multi-key contrast.
- `--max_num_augs`
    - default: `6`
    - Maximum augmentations per anchor group ($K_{\max}$). Higher values use more GPU memory per batch item. Set to the number of transformation operators applied to each language (e.g. 3 for Python-only, 6 for Java-only, 6 for mixed). Only used when `--contra_mode grouped`.
- `--alpha`
    - default: `1.0`
    - Weight $\alpha$ of the contrastive loss relative to MLM.
- `--temperature`
    - default: `0.07`
    - Contrastive temperature $\tau$. Lower values sharpen the distribution and increase the penalty for hard negatives.
- `--batch_size`
    - default: `256`
    - Total batch size $B$. Each item now requires $K_{\max}$ augmentation forward passes, so reduce batch size and increase `--gradient_accumulation_steps` compared to `info_nce`/`supcon` modes.

## Comparison with Other Modes

| Property                                      | `info_nce`             | `supcon`                               | `grouped`                                             |
| --------------------------------------------- | ---------------------- | -------------------------------------- | ----------------------------------------------------- |
| Positives per anchor                          | 1 (diagonal)           | Variable (depends on batch collisions) | All $K_i$ augmentations (guaranteed)                  |
| Requires same-function co-occurrence in batch | N/A                    | Yes                                    | No (grouped at dataset level)                         |
| Dataset format                                | Flat (code, aug) pairs | Flat (code, aug) pairs                 | Grouped (code, $[\text{aug}_1, \dots, \text{aug}_K]$) |
| Memory per batch item                         | 2 forward passes       | 2 forward passes                       | $1 + K_{\max}$ forward passes                         |
| Handles variable aug counts                   | N/A                    | Naturally (mask-based)                 | Yes (padding + `group_sizes` mask)                    |
