# Ablation Studies

## Goal

Justify the core design decisions of InvPT for the COLM submission. Each
ablation removes or substitutes exactly one component of the full method to
isolate its contribution. All ablations use **CodeBERT**
(`microsoft/codebert-base`) as the representative backbone to keep compute
manageable.

## Full Method (Control)

The full training loss is:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}}(X) + \mathcal{L}_{\text{MLM}}(X_{\text{inv}}) + \alpha \cdot \mathcal{L}_{\text{SupCon}}(X, X_{\text{inv}})$$

with `alpha=1.0`, `temperature=0.1`, `self_contrast=true`.

Config: `experiments/supcon/codebert.yaml` (run name `InvCodeBERT-supcon`).

## Ablation Matrix

| #   | Ablation         | Question answered                                         | Config                           |
| --- | ---------------- | --------------------------------------------------------- | -------------------------------- |
| 1   | MLM-only         | Does the contrastive loss help at all?                    | `ablation/mlm_only.yaml`         |
| 2   | No self-contrast | Is self-contrast (same code, different masks) beneficial? | `ablation/no_self_contrast.yaml` |
| 3   | + NL (bimodal)   | Is PL-only training better than bimodal NL+PL training?   | `ablation/include_nl.yaml`       |

### What changes per ablation

| #   | `alpha` | `self_contrast` | `include_nl` | run_name                           |
| --- | :-----: | :-------------: | :----------: | ---------------------------------- |
| 1   |  **0**  |      true       |    false     | `InvCodeBERT-ablation-mlm-only`    |
| 2   |   1.0   |    **false**    |    false     | `InvCodeBERT-ablation-no-selfcon`  |
| 3   |   1.0   |      true       |   **true**   | `InvCodeBERT-ablation-include-nl`  |

## Ablation Details

### 1 — MLM-Only (`alpha=0`)

$$\mathcal{L} = \mathcal{L}_{\text{MLM}}(X) + \mathcal{L}_{\text{MLM}}(X_{\text{inv}})$$

Removes the contrastive objective entirely. The model still trains on both the
original code $X$ and its invariant augmentation $X_{\text{inv}}$ via MLM, so it
sees the same data as the full method — only the explicit alignment signal is
missing. This isolates the contribution of contrastive learning.

**Expected outcome.** If the full method outperforms this baseline, the
contrastive loss provides value beyond what MLM on augmented data alone achieves.

### 2 — No Self-Contrast (`self_contrast=false`)

Same loss as the full method, but rows without a real code transformation are
**dropped** instead of using the original code as its own augmentation. This
reduces the training set size (rows that failed all transformation operators are
excluded) and removes the "easy" contrastive pairs where both views are the same
code with different MLM masks.

**Expected outcome.** If the full method outperforms this ablation, self-contrast
is beneficial — it acts as an implicit curriculum (easy pairs stabilize training)
and improves data efficiency. See `doc/self-contrast.md` for a detailed analysis.

### 3 — Include NL (`include_nl=true`)

Re-introduces natural language docstrings into the pre-training input, restoring
the bimodal NL+PL setup used by CodeBERT, GraphCodeBERT, and ContraBERT. When
`include_nl=true`, each code input is formatted as `[CLS] <docstring> [SEP]
<code> [EOS]` instead of the default `[CLS] <code> [EOS]`. The same
prepending applies to the invariant-transformed code.

This ablation directly tests InvPT's core hypothesis: that PL-only pre-training
is sufficient (and preferable) for learning robust code representations. Prior
work universally relies on NL-PL paired training, and the ICSE'26 submission
showed that adding NL back actually *degraded* performance while consuming more
memory — likely due to overfitting on NL descriptions rather than learning
program semantics.

**Expected outcome.** The PL-only control should outperform this NL-inclusive
variant, especially on robustness, confirming that docstrings are not needed and
can even be harmful for invariant pre-training.

**Code change required.** Add an `include_nl: bool = False` parameter to
`PretrainConfig`. In `tokenize_grouped`, when `include_nl=true`, prepend the
docstring to the code before tokenization (for both anchor and augmented
inputs). The docstring field is already carried through the data pipeline but
currently unused during tokenization.

| File                   | Change                                                              |
| ---------------------- | ------------------------------------------------------------------- |
| `modeling/config.py`   | Add `include_nl: bool = False` to `PretrainConfig`                  |
| `modeling/pretrain.py` | `tokenize_grouped` prepends docstring to code when `include_nl` set |

## Running

Smoke-test each config with a 1% sample:

```bash
# Control
python -m modeling run experiments/supcon/codebert.yaml --sample-rate 0.01

# Ablations
python -m modeling run experiments/ablation/mlm_only.yaml --sample-rate 0.01
python -m modeling run experiments/ablation/no_self_contrast.yaml --sample-rate 0.01
python -m modeling run experiments/ablation/include_nl.yaml --sample-rate 0.01
```

### What to verify

- **mlm_only:** Contrastive loss contributes 0 to total loss (alpha=0).
- **no_self_contrast:** Dataset is smaller (filtered rows without augmentation).
- **include_nl:** Tokenized inputs start with docstring before code (`[CLS] docstring [SEP] code [EOS]`).
