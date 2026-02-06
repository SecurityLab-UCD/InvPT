# Supervised Contrastive Loss with Multi-Positive Masking

## Motivation

In the standard InfoNCE setup (`--contra_mode info_nce`), each training example
is a pair (anchor code, one augmentation). The contrastive loss treats the
diagonal entries of the $B \times B$ similarity matrix as positives and everything
else as negatives. When the dataset contains multiple augmentations of the same
function (e.g. VarRe, RevIf, AA2EA applied to the same source), those
augmentations land in different batch rows and are **pushed apart as negatives**
even though they are semantically equivalent.

SupCon (`--contra_mode supcon`) mitigates this by building a **positive mask**
at loss time: it assigns a deterministic `function_id` to every code snippet and
treats all embeddings that share the same `function_id` within a mini-batch as
positives. This allows the model to learn invariance across all co-occurring
augmentations of a function without any changes to the dataset format or the
data loading pipeline.

## Algorithm

### 1. Function ID Assignment

Each code snippet receives a deterministic identifier derived from the
**original (non-augmented) source code**:

$$\texttt{function\_id}(c) = \text{int64}\!\bigl(\text{SHA-256}(c)[0{:}8]\bigr) \;\mathbin{\&}\; \texttt{0x7FFFFFFFFFFFFFFF}$$

Concretely, `compute_function_id` in `modeling/pretrain.py` hashes the code
string with SHA-256, takes the first 8 bytes, converts to a 64-bit integer, and
masks the sign bit to guarantee a positive value that fits in a signed `int64`.

Because the hash is computed from the canonical code, all augmentations of the
same function map to the same `function_id`.

### 2. Dataset & Collation

SupCon uses the **same flat dataset format** as InfoNCE: one JSONL row per
`(code, transformed)` pair. No regrouping is required.

The collator (`contra_data_collator` in `modeling/dataloader.py`) produces:

| Tensor                | Shape    | Description                               |
| --------------------- | -------- | ----------------------------------------- |
| `code_input_ids`      | $[B, L]$ | Tokenized anchors (MLM-masked)            |
| `code_attention_mask` | $[B, L]$ | Attention mask for anchors                |
| `code_labels`         | $[B, L]$ | MLM labels for anchors ($-100$ = no loss) |
| `aug_input_ids`       | $[B, L]$ | Tokenized augmentations (MLM-masked)      |
| `aug_attention_mask`  | $[B, L]$ | Attention mask for augmentations          |
| `aug_labels`          | $[B, L]$ | MLM labels for augmentations              |
| `function_id`         | $[B]$    | Integer function identifier per sample    |

where $B$ is the batch size and $L$ is `max_seq_length`.

### 3. Forward Pass

A single shared encoder $f_\theta$ processes both anchors and augmentations in
two separate forward passes:

$$\mathbf{h}^{\text{code}}_i = f_\theta(\text{code}_i) \quad \text{for } i = 1, \dots, B \qquad \to [B, D]$$

$$\mathbf{h}^{\text{aug}}_i = f_\theta(\text{aug}_i) \quad \text{for } i = 1, \dots, B \qquad \to [B, D]$$

The CLS token embedding (position 0 of the last hidden layer) is used as the
sequence representation. Both forward passes also produce MLM logits for the
masked language modeling objective.

The two sets of embeddings are then concatenated into a single pool:

$$\mathbf{z} = [\mathbf{h}^{\text{code}}_1, \dots, \mathbf{h}^{\text{code}}_B, \mathbf{h}^{\text{aug}}_1, \dots, \mathbf{h}^{\text{aug}}_B] \qquad \to [2B, D]$$

The function IDs are likewise duplicated:

$$\text{ids} = [\text{id}_1, \dots, \text{id}_B, \text{id}_1, \dots, \text{id}_B] \qquad \to [2B]$$

This duplication is correct because `aug_i` is a transformation of `code_i` and
therefore shares the same `function_id`.

### 4. Positive Mask Construction

Given the duplicated function ID vector of length $N = 2B$, the positive mask is
an $N \times N$ boolean matrix:

$$M_{ij} = \begin{cases} \text{True} & \text{if } \text{ids}_i = \text{ids}_j \text{ and } i \neq j \\ \text{False} & \text{otherwise} \end{cases}$$

Implemented in `build_positive_mask` (`modeling/model.py`):

```python
mask = function_ids.unsqueeze(0) == function_ids.unsqueeze(1)  # [N, N]
mask.fill_diagonal_(False)
```

**Example.** Consider a batch of size $B = 4$ drawn from three distinct
functions $A$, $B$, $C$ where function $A$ appears twice (two different
augmentations sampled independently):

```
Batch:  [A,  B,  C,  A]
IDs:    [hA, hB, hC, hA, hA, hB, hC, hA]   (code ids ++ aug ids)
         0   1   2   3   4   5   6   7
```

The positive mask has `True` at positions where IDs match and $i \neq j$:

|       | 0   | 1   | 2   | 3   | 4   | 5   | 6   | 7   |
| ----- | --- | --- | --- | --- | --- | --- | --- | --- |
| **0** | -   |     |     | +   | +   |     |     | +   |
| **1** |     | -   |     |     |     | +   |     |     |
| **2** |     |     | -   |     |     |     | +   |     |
| **3** | +   |     |     | -   | +   |     |     | +   |
| **4** | +   |     |     | +   | -   |     |     | +   |
| **5** |     | +   |     |     |     | -   |     |     |
| **6** |     |     | +   |     |     |     | -   |     |
| **7** | +   |     |     | +   | +   |     |     | -   |

Entries marked `+` are positives; `-` is the diagonal (self, excluded). All
blank entries are negatives.

Anchors for function $A$ (indices 0, 3, 4, 7) each have 3 positives. Anchors
for functions $B$ and $C$ each have 1 positive (their own augmentation).

### 5. Contrastive Loss

**Definitions.** Let:

- $N = 2B$ (total embeddings)
- $\mathbf{z}_i = \text{normalize}(\mathbf{h}_i)$ ($\ell_2$-normalized embedding for index $i$)
- $\tau$ = temperature parameter
- $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v}$ (cosine similarity of normalized vectors)
- $\mathcal{P}(i) = \{j : M_{ij} = \text{True}\}$ (positive set for anchor $i$)

**Similarity matrix.** Compute pairwise scaled cosine similarities:

$$s_{ij} = \frac{\text{sim}(\mathbf{z}_i, \mathbf{z}_j)}{\tau} \qquad \text{for all } i, j \in \{1, \dots, N\}$$

**Per-positive loss.** For each anchor $i$ that has at least one positive
($|\mathcal{P}(i)| > 0$) and each $p \in \mathcal{P}(i)$:

$$\ell_{i,p} = -\log \frac{\exp(s_{ip})}{\displaystyle\sum_{\substack{j=1 \\ j \neq i}}^{N} \exp(s_{ij})}$$

**Per-anchor loss** (averaged over all positives):

$$\mathcal{L}_i = \frac{1}{|\mathcal{P}(i)|} \sum_{p \in \mathcal{P}(i)} \ell_{i,p}$$

**Batch loss** (averaged over anchors with at least one positive):

$$\mathcal{L}_{\text{supcon}} = \frac{1}{|\{i : |\mathcal{P}(i)| > 0\}|} \sum_{\substack{i=1 \\ |\mathcal{P}(i)| > 0}}^{N} \mathcal{L}_i$$

**Note:** The denominator of $\ell_{i,p}$ sums over **all** $j \neq i$,
including other positives. This follows the SupCon convention (Khosla et al.,
NeurIPS 2020) where positives are not excluded from the partition function.

### 6. Total Training Loss

The total loss combines MLM and contrastive objectives:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \alpha \cdot \mathcal{L}_{\text{supcon}}$$

where the MLM loss averages over anchor and augmentation views:

$$\mathcal{L}_{\text{MLM}} = \frac{\mathcal{L}_{\text{MLM}}(\text{code}) + \mathcal{L}_{\text{MLM}}(\text{aug})}{2}$$

### 7. Numerical Stability

The implementation uses the log-sum-exp trick to prevent overflow in the
exponential computation. Given the raw logits $s_{ij}$:

1. Compute $m_i = \max_{j \neq i} s_{ij}$, clamped to $\geq 0$
2. Subtract before exponentiation: $\exp(s_{ij} - m_i)$
3. Zero out self-similarities after exponentiation
4. Denominator: $\sum_{j \neq i} \exp(s_{ij} - m_i) + \epsilon$
5. Log probabilities: $\log\!\bigl(\exp(s_{ij} - m_i) / \text{denom} + \epsilon\bigr)$

where $\epsilon = 10^{-8}$ is a small constant to avoid $\log(0)$.

### 8. Handling Edge Cases

- **No positives in batch.** If no anchor in the batch has any matching
  `function_id` (i.e. every function appears exactly once), the SupCon loss
  degenerates gracefully: each anchor has exactly one positive (its paired
  augmentation at offset $B$), recovering behavior similar to InfoNCE.

- **All-unique batch.** In the extreme case where every row has a unique
  `function_id` _and_ the augmentation for each anchor happens to be assigned a
  different ID (which cannot happen by construction since IDs are duplicated),
  the loss returns `0.0` with `requires_grad=True` to avoid breaking the
  computation graph.

- **Variable positive counts.** Different anchors may have different numbers of
  positives depending on how many same-function rows co-occur in the batch. The
  per-anchor averaging ($1 / |\mathcal{P}(i)|$) ensures that anchors with many
  positives are not over-weighted relative to anchors with few.

## Parameters

- `--contra_mode`
    - default: `info_nce`
    - Set to `supcon` to enable supervised contrastive loss with multi-positive masking.
- `--alpha`
    - default: `1.0`
    - Weight $\alpha$ of the contrastive loss relative to MLM.
- `--temperature`
    - default: `0.07`
    - Contrastive temperature $\tau$. Lower values sharpen the distribution and increase the penalty for hard negatives.
- `--batch_size`
    - default: `256`
    - Total batch size $B$. Larger batches increase the probability of same-function co-occurrences, which strengthens the multi-positive signal. SupCon has the same memory footprint as InfoNCE (2 forward passes per batch item).

## Comparison with Other Modes

| Property                                      | `info_nce`             | `supcon`                               | `grouped`                                             |
| --------------------------------------------- | ---------------------- | -------------------------------------- | ----------------------------------------------------- |
| Positives per anchor                          | 1 (diagonal)           | Variable (depends on batch collisions) | All $K_i$ augmentations (guaranteed)                  |
| Requires same-function co-occurrence in batch | N/A                    | Yes                                    | No (grouped at dataset level)                         |
| Dataset format                                | Flat (code, aug) pairs | Flat (code, aug) pairs                 | Grouped (code, $[\text{aug}_1, \dots, \text{aug}_K]$) |
| Memory per batch item                         | 2 forward passes       | 2 forward passes                       | $1 + K_{\max}$ forward passes                         |
| Data pipeline changes                         | None                   | None (adds `function_id` field only)   | Requires dataset regrouping step                      |
| Handles variable aug counts                   | N/A                    | Naturally (mask-based)                 | Yes (padding + `group_sizes` mask)                    |
| Positive guarantee                            | Always 1               | Minimum 1 (paired aug)                 | All $K_i$ (by construction)                           |

## When to Prefer SupCon over InfoNCE

SupCon is beneficial when:

1. **Multiple augmentations per function exist** in the dataset (e.g. both VarRe
   and RevIf applied to the same source). Without SupCon, co-occurring
   augmentations of the same function are treated as negatives.

2. **Batch size is large enough** for same-function collisions to occur
   frequently. With a dataset of $F$ unique functions and batch size $B$, the
   expected number of same-function pairs grows quadratically with the number of
   augmentations per function. At $B = 256$ with 3 augmentations per function,
   collisions are common.

3. **Simplicity is preferred** over the grouped mode. SupCon requires no changes
   to the dataset format or collation pipeline --- only the loss function changes.

## When to Prefer Grouped over SupCon

The grouped mode (`--contra_mode grouped`) is preferable when:

1. **Small batch sizes** make same-function co-occurrences rare, degrading the
   multi-positive signal of SupCon.

2. **Guaranteed positive coverage** is desired: grouped mode ensures _every_
   augmentation of a function is present as a positive, whereas SupCon relies on
   stochastic co-occurrence.

## References

- Khosla, P., Teterwak, P., Wang, C., Sarkis, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., & Krishnan, D. (2020). Supervised Contrastive Learning. _NeurIPS 2020_.
