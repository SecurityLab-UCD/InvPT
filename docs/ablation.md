# Tier 1 Ablation Studies

## Goal

Justify the core design decisions of InvPT for the COLM submission. Each
ablation removes or substitutes exactly one component of the full method to
isolate its contribution. All ablations use **CodeBERT**
(`microsoft/codebert-base`) as the representative backbone to keep compute
manageable.

## Full Method (Control)

The full training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}}(X) + \mathcal{L}_{\text{MLM}}(X_{\text{inv}}) + \alpha \cdot \mathcal{L}_{\text{SupCon}}(X, X_{\text{inv}})$$

with `alpha=1.0`, `temperature=0.1`, `contra_mode=supcon`, `self_contrast=true`.

Config: `experiments/supcon/codebert.yaml` (run name `InvCodeBERT-supcon`).

## Ablation Matrix

| #   | Ablation         | Question answered                                         | Config                           |
| --- | ---------------- | --------------------------------------------------------- | -------------------------------- |
| 1a  | MLM-only         | Does the contrastive loss help at all?                    | `ablation/mlm_only.yaml`         |
| 1b  | Contrastive-only | Does MLM help, or is contrastive alone sufficient?        | `ablation/contra_only.yaml`      |
| 1c  | Full method      | Control — both losses together.                           | `supcon/codebert.yaml`           |
| 2   | No self-contrast | Is self-contrast (same code, different masks) beneficial? | `ablation/no_self_contrast.yaml` |
| 3   | InfoNCE          | Does SupCon's multi-positive masking outperform InfoNCE?  | `ablation/infonce.yaml`          |

### What changes per ablation

| #   | `mlm_weight` | `alpha` | `contra_mode` | `self_contrast` | run_name                           |
| --- | :----------: | :-----: | ------------- | :-------------: | ---------------------------------- |
| 1a  |     1.0      |  **0**  | supcon        |      true       | `InvCodeBERT-ablation-mlm-only`    |
| 1b  |    **0**     |   1.0   | supcon        |      true       | `InvCodeBERT-ablation-contra-only` |
| 1c  |     1.0      |   1.0   | supcon        |      true       | `InvCodeBERT-supcon`               |
| 2   |     1.0      |   1.0   | supcon        |    **false**    | `InvCodeBERT-ablation-no-selfcon`  |
| 3   |     1.0      |   1.0   | **info_nce**  |      true       | `InvCodeBERT-ablation-infonce`     |

## Ablation Details

### 1a — MLM-Only (`alpha=0`)

$$\mathcal{L} = \mathcal{L}_{\text{MLM}}(X) + \mathcal{L}_{\text{MLM}}(X_{\text{inv}})$$

Removes the contrastive objective entirely. The model still trains on both the
original code $X$ and its invariant augmentation $X_{\text{inv}}$ via MLM, so it
sees the same data as the full method — only the explicit alignment signal is
missing. This isolates the contribution of contrastive learning.

**Expected outcome.** If the full method outperforms this baseline, the
contrastive loss provides value beyond what MLM on augmented data alone achieves.

### 1b — Contrastive-Only (`mlm_weight=0`)

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{SupCon}}(X, X_{\text{inv}})$$

Removes the MLM objective. The encoder is trained purely to align
representations of semantically equivalent code. This tests whether MLM serves
as a useful regularizer or whether contrastive learning alone is sufficient.

**Expected outcome.** Contrastive-only may underperform on token-level
downstream tasks (e.g., clone detection with fine-grained attention) where MLM's
token prediction signal is important.

**Code change required.** The loss was previously hardcoded as
`mlm_loss + alpha * contrastive_loss`. A new `mlm_weight` parameter (default
`1.0`) was added to make the MLM coefficient configurable:

$$\mathcal{L} = \texttt{mlm\_weight} \cdot \mathcal{L}_{\text{MLM}} + \alpha \cdot \mathcal{L}_{\text{contrastive}}$$

See [Code Changes](#code-changes) for details.

### 2 — No Self-Contrast (`self_contrast=false`)

Same loss as the full method, but rows without a real code transformation are
**dropped** instead of using the original code as its own augmentation. This
reduces the training set size (rows that failed all transformation operators are
excluded) and removes the "easy" contrastive pairs where both views are the same
code with different MLM masks.

**Expected outcome.** If the full method outperforms this ablation, self-contrast
is beneficial — it acts as an implicit curriculum (easy pairs stabilize training)
and improves data efficiency. See `doc/self-contrast.md` for a detailed analysis.

### 3 — InfoNCE vs SupCon (`contra_mode="info_nce"`)

Replaces SupCon with standard InfoNCE. InfoNCE treats only the diagonal of the
similarity matrix as positives — each anchor has exactly one positive (its paired
augmentation). SupCon builds a positive mask from `function_id`, so all
augmentations of the same function within a batch are treated as positives.

**Expected outcome.** If SupCon outperforms InfoNCE, the multi-positive masking
improves learning by avoiding false negatives (same-function augmentations pushed
apart). See `doc/supervised_contrastive_mask.md` for a detailed comparison.

## Code Changes

Only ablation 1b required modifying source code. The other ablations were
already expressible through existing config parameters (`alpha`,
`self_contrast`, `contra_mode`).

| File                   | Change                                                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `modeling/config.py`   | Added `mlm_weight: float = 1.0` to `PretrainConfig`                                                                             |
| `modeling/model.py`    | `ContrastiveTrainer.__init__` accepts `mlm_weight`; `compute_loss` and `_compute_grouped_loss` use `self.mlm_weight * mlm_loss` |
| `modeling/pretrain.py` | `main()` accepts `mlm_weight` and forwards to `ContrastiveTrainer`                                                              |
| `modeling/cli.py`      | Added `--mlm-weight` option to the `pretrain` command                                                                           |

The default `mlm_weight=1.0` preserves existing behavior for all current
experiments.

## Running

Smoke-test each config with a 1% sample:

```bash
# Control
python -m modeling run experiments/supcon/codebert.yaml --sample-rate 0.01

# Ablations
python -m modeling run experiments/ablation/mlm_only.yaml --sample-rate 0.01
python -m modeling run experiments/ablation/contra_only.yaml --sample-rate 0.01
python -m modeling run experiments/ablation/no_self_contrast.yaml --sample-rate 0.01
python -m modeling run experiments/ablation/infonce.yaml --sample-rate 0.01
```

### What to verify

- **mlm_only:** Contrastive loss contributes 0 to total loss (alpha=0).
- **contra_only:** MLM loss contributes 0 to total loss (mlm_weight=0).
- **no_self_contrast:** Dataset is smaller (filtered rows without augmentation).
- **infonce:** Loss function dispatches to `info_nce_loss` instead of `supcon_loss`.
