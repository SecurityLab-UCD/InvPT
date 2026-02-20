# Self-Contrast: Analysis

## What Is Self-Contrast?

Self-contrast is a data-level strategy for handling rows in the pre-training dataset
that have **no code transformation** (i.e., `transformed` is `None`).
Instead of discarding these rows, self-contrast copies the original code into the
`transformed` field, creating a pair where both the anchor and the augmentation are
the **same source code**.

Because MLM masking is applied **independently** to each copy
(`DataCollatorForLanguageModeling` with `mlm_probability=0.15`), the model sees
two views of the same code with **different random masks**. This produces an "easy"
contrastive pair — same semantics, same syntax, different masked tokens.

### Implementation

The entire mechanism is a single `dataset.map` call in `pretrain.py:268-276`:

```python
if self_contrast:
    dataset = dataset.map(
        lambda transformed, code: {
            "transformed": transformed if transformed is not None else code
        },
        input_columns=["transformed", "code"],
        num_proc=num_proc,
    )
```

No changes are needed in the loss functions, model architecture, or training loop.
The contrastive loss (InfoNCE, SupCon, or Grouped) treats self-contrast pairs
identically to invariant-contrast pairs.

## Why It Is Feasible

### 1. MLM Masking Creates Distinct Views

Even though both copies share the same token sequence, the `DataCollatorForLanguageModeling`
samples a fresh 15% mask for each copy independently. This means:

- The `code_input_ids` has some tokens replaced with `[MASK]`.
- The `aug_input_ids` has a **different** set of tokens replaced with `[MASK]`.

The CLS embeddings produced by the encoder therefore differ, because the model
conditions on different visible tokens. This difference is what makes the contrastive
signal non-trivial — the model must learn that both masked views correspond to the
same underlying program.

### 2. Stochastic Masking as a Natural Data Augmentation

MLM masking is itself a form of stochastic data augmentation.
Given a sequence of length $L$, the number of possible 15%-masks is
$\binom{L}{\lfloor 0.15L \rfloor}$, which is astronomically large for typical code
sequences ($L \approx 100$–$256$). Each training step therefore produces a genuinely
novel pair, even if the underlying code is the same. The model never sees the exact
same (mask_a, mask_b) combination twice in practice.

### 3. Zero Implementation Overhead

Self-contrast requires no additional model components, no separate training stage,
and no architectural changes. It is purely a data-filling operation that replaces
`None` with the original code string. The downstream data collator, tokenizer,
and loss functions are completely agnostic to whether a pair is self-contrast or
invariant-contrast.

### 4. Consistency with the Shared-Encoder Design

InvPT uses a **single shared encoder** for both the original and transformed code
(unlike ContraBERT's momentum contrast with a separate momentum encoder).
In this architecture, the same parameters process both copies.
Self-contrast is a natural fit: since the encoder is shared, feeding the same code
twice with different masks is equivalent to asking "do two partial observations of
the same program yield similar representations?" — a well-defined learning objective.

## Why It Is Beneficial

### 1. Implicit Curriculum Learning

Self-contrast creates an implicit curriculum with two difficulty levels:

| Pair Type          | Difficulty | Source of Difference                           |
| ------------------ | ---------- | ---------------------------------------------- |
| Self-contrast      | Easy       | Different MLM masks only                       |
| Invariant-contrast | Hard       | Syntactic transformation + different MLM masks |

**Easy pairs** (self-contrast) produce CLS embeddings that are already relatively
close, generating small but consistent gradients that stabilize training.
**Hard pairs** (invariant-contrast) produce more distant embeddings, generating
larger gradients that push the model to learn semantic invariance.

Training on both simultaneously provides the benefits of curriculum learning
without explicit scheduling: the easy pairs act as a regularizer, preventing the
model from being destabilized by the harder invariant pairs early in training.
As the model improves, self-contrast pairs become trivially easy and the learning
signal shifts toward the harder invariant pairs, naturally concentrating the
gradient on the most informative examples.

### 2. Data Efficiency — No Wasted Rows

Not every code sample in the dataset has a successful transformation.
Transformation operators may fail (e.g., no for-loops to convert, no local
variables to rename). Without self-contrast, these rows are **discarded**:

```python
# Without self-contrast: rows without augmentation are dropped
dataset = dataset.filter(
    lambda transformed: transformed is not None and transformed != "",
    input_columns=["transformed"],
)
```

This wastes a potentially large fraction of the dataset. With self-contrast,
**every row contributes** to both the MLM and contrastive objectives. The model
sees more code, learns better MLM representations, and the contrastive batch
is denser (more positives in SupCon mode, since rows sharing the same
`function_id` all participate).

### 3. MLM Regularization on the Full Corpus

Even if the contrastive gradient from a self-contrast pair is small, the **MLM
loss** on that pair is still fully informative — the model must predict different
masked tokens in each copy. Retaining these rows ensures the MLM objective
covers the entire pre-training corpus, not just the transformable subset.
This is important because the transformation operators have language-specific
coverage (e.g., Python lacks `F2W` and `W2F`), so dropping untransformable rows
would bias the MLM objective toward code containing certain syntactic patterns.

### 4. Representation Stability via Alignment

Self-contrast directly optimizes what Chen & He (2021) call the **alignment**
property of contrastive representations: embeddings of the same instance
(under different views) should be close. By ensuring the encoder maps different
masked views of the same code to similar CLS vectors, self-contrast encourages
a smooth, stable representation manifold where small input perturbations
(missing tokens) do not cause large embedding shifts. This stability transfers
to downstream tasks where inputs may be noisy or partial.

### 5. Compatibility with SupCon Multi-Positive Masking

In SupCon mode, all embeddings sharing the same `function_id` are treated as
positives. Self-contrast rows share `function_id` with their corresponding
invariant-contrast rows (since `function_id` is a hash of the original code).
This means self-contrast rows add **additional positive pairs** to the SupCon
loss, increasing the effective number of positives per anchor. Khosla et al.
(2020) showed that SupCon benefits from more positives per class, making
self-contrast especially synergistic with the SupCon contrastive mode.

## Connection to Prior Work

Self-contrast is conceptually related to several established techniques:

- **SimCLR** (Chen et al., 2020): Uses two random augmentations of the same
  image as a positive pair. Self-contrast uses two random MLM masks of the
  same code as a positive pair — the MLM mask plays the role of the
  stochastic augmentation.
- **Dropout-as-augmentation** (Gao et al., 2021, SimCSE): Passes the same
  sentence through the encoder twice with different dropout masks to create
  positive pairs. Self-contrast is analogous but uses MLM masking instead
  of dropout as the source of stochasticity.
- **BERT's own MLM objective**: Standard MLM already trains the model to be
  robust to random token masking. Self-contrast extends this to the
  contrastive objective, requiring not just correct token prediction but
  also consistent sequence-level (CLS) representations across masks.

## Summary

Self-contrast is a simple, zero-cost data strategy that:

1. Turns untransformable rows into useful training signal (data efficiency).
2. Creates an implicit easy/hard curriculum (training stability).
3. Ensures MLM coverage of the full corpus (unbiased language modeling).
4. Encourages alignment and smoothness in the representation space (downstream robustness).
5. Requires no changes to the model, loss, or training loop (implementation simplicity).
