# ModernBERT Support

## Overview

InvPT supports [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) as an alternative
backbone to RoBERTa/CodeBERT/ContraBERT. ModernBERT is a modernized BERT architecture
from AnswerDotAI that incorporates recent advances in transformer design while retaining
the encoder-only, MLM-based pre-training paradigm that InvPT builds on.

Two variants are available on HuggingFace:

| Variant | Layers | Hidden | Params | HF Identifier                  |
| ------- | ------ | ------ | ------ | ------------------------------ |
| Base    | 22     | 768    | 149M   | `answerdotai/ModernBERT-base`  |
| Large   | 28     | 1024   | 395M   | `answerdotai/ModernBERT-large` |

## Why ModernBERT for InvPT

### Trained on Code

ModernBERT was pre-trained on 2 trillion tokens of English text **and code**.
RoBERTa (and by extension CodeBERT/ContraBERT) was trained on ~160GB of English text only.
This means ModernBERT already has a code-aware representation before InvPT's continued
pre-training, giving it a stronger starting point for learning invariant code representations.

### Longer Context (8,192 tokens)

RoBERTa's absolute position embeddings cap out at 512 tokens. ModernBERT uses Rotary
Position Embeddings (RoPE) with a maximum of 8,192 tokens. This allows processing longer
functions without truncation — particularly useful for languages like Java and C++ where
functions routinely exceed 512 tokens.

### Architectural Improvements

| Feature             | RoBERTa                     | ModernBERT                                                          |
| ------------------- | --------------------------- | ------------------------------------------------------------------- |
| Position embeddings | Absolute (learned, 512 max) | RoPE (8,192 max)                                                    |
| Attention pattern   | Full attention all layers   | Local-global alternating (sliding window 128 + full every 3 layers) |
| Attention backend   | Standard SDPA               | Flash Attention 2 + unpadding                                       |
| FFN activation      | GELU                        | GeGLU (gated)                                                       |
| Normalization       | Post-LayerNorm              | Pre-Norm (no bias)                                                  |
| Weight tying        | No                          | Yes (embeddings tied to decoder)                                    |

The local-global alternating attention is especially relevant: it reduces the quadratic
cost of full attention on long sequences while preserving global information flow through
periodic full-attention layers (every 3rd layer).

Flash Attention and unpadding (skipping compute on padding tokens) provide significant
speedups, particularly for variable-length code batches where padding waste is high.

## Configuration

### Experiment Config

Create a YAML config in `experiments/` (or use the provided `modernbert_base.yaml`):

```yaml
model_name: "answerdotai/ModernBERT-base"
model_type: "modernbert"
pooling: "mean"
max_seq_length: 512
run_name: "InvPT-ModernBERT-base"
```

Two new fields control the behavior:

- **`model_type`**: Either `"roberta"` (default) or `"modernbert"`. Determines how
  the wrapper accesses the encoder and LM head internals.
- **`pooling`**: Either `"cls"` (default, CLS token) or `"mean"` (mean over non-padding
  tokens). Mean pooling is recommended for ModernBERT (see below).

All other config fields (`alpha`, `temperature`, etc.) work identically.

### CLI Usage

```bash
# From YAML config (recommended)
python modeling/cli.py run experiments/modernbert_base.yaml

# Direct CLI options
python modeling/cli.py pretrain \
    --model-name answerdotai/ModernBERT-base \
    --model-type modernbert \
    --pooling mean \
    --max-seq-length 512
```

## Training Pipeline

The training pipeline is **architecture-agnostic** — ModernBERT uses the exact same
loss computation, contrastive learning, and curriculum as RoBERTa/CodeBERT/ContraBERT.
The only ModernBERT-specific logic lives in `SplitHeadWrapper` (encoder/LM-head dispatch)
and the `pooling` config field.

### Loss Function

The total loss is identical for both model types:

```
L = L_MLM(code) + L_MLM(aug) + alpha * L_contrastive(code, aug)
```

- **MLM loss**: 15% random token masking applied independently to both the original code
  and its augmentation. Computed per-chunk via `SplitHeadWrapper` to avoid materializing
  the full `[2B, seq_len, vocab_size]` logit tensor.
- **Contrastive loss**: Computed on pooled embeddings (CLS or mean) from the shared
  encoder's last hidden states. Controlled by `alpha` (default 1.0) and `temperature`
  (default 0.07).

### Self-Contrast

Self-contrast (`self_contrast: true`, the default) provides the "easy" curriculum signal.
When a dataset row has no successful transformation (e.g., the code had no variables to
rename), the original code is reused as its own augmentation. The contrastive signal then
comes from **different MLM masks** applied to identical code — the encoder must learn that
the same code under different masks maps to the same representation.

This is independent of model type and works identically for ModernBERT. When
`self_contrast` is disabled, rows without transformations are dropped from the dataset.

### What Differs

| Aspect                  | RoBERTa                                   | ModernBERT                    |
| ----------------------- | ----------------------------------------- | ----------------------------- |
| Loss function           | `L_MLM + alpha * L_contrastive`           | Same                          |
| Contrastive loss        | SupCon                                    | Same                          |
| Self-contrast           | Supported                                 | Same                          |
| MLM masking             | 15% via `DataCollatorForLanguageModeling` | Same                          |
| Pooling for contrastive | CLS (default)                             | Mean (recommended)            |
| Encoder dispatch        | `model.roberta`                           | `model.model`                 |
| LM head dispatch        | `model.lm_head()`                         | `model.decoder(model.head())` |

## Pooling: Mean vs CLS

ModernBERT uses **mean pooling** by default (`config.classifier_pooling = "mean"`),
not CLS-token pooling. This is a deliberate design choice driven by the architecture:

- In local attention layers (sliding window of 128 tokens), the CLS token at position 0
  can only attend to tokens within its window. It does **not** see the full sequence.
- Full attention layers (every 3rd) allow global information flow, but the CLS token
  still receives a biased view compared to mean pooling over all positions.
- Mean pooling aggregates information from all positions equally, weighted by the
  attention mask (padding tokens are excluded).

For InvPT's contrastive learning, this means:

```
CLS pooling:  embedding = last_hidden[:, 0, :]
Mean pooling: embedding = (last_hidden * mask).sum(1) / mask.sum(1)
```

The `pooling` config field controls this in both `compute_loss` and
`_compute_grouped_loss` of `ContrastiveTrainer`. It is architecture-independent —
you can use mean pooling with RoBERTa too, though CLS is the conventional choice there.

## Architecture Dispatch

The `SplitHeadWrapper` auto-detects the model architecture via attribute inspection:

| Component        | RoBERTa                 | ModernBERT                          | Detection                       |
| ---------------- | ----------------------- | ----------------------------------- | ------------------------------- |
| Encoder backbone | `model.roberta`         | `model.model`                       | `hasattr(mlm_model, "roberta")` |
| LM head          | `model.lm_head(hidden)` | `model.decoder(model.head(hidden))` | `hasattr(mlm_model, "lm_head")` |

ModernBERT splits the LM head into two stages:

1. **`head`**: `ModernBertPredictionHead` — dense layer + GELU activation + layer norm
2. **`decoder`**: `nn.Linear(hidden_size, vocab_size)` — projection to vocabulary

The wrapper applies these sequentially per-chunk to maintain the memory optimization
(never materializing the full `[N, seq_len, vocab_size]` logit tensor).

## Tokenizer

ModernBERT uses a BPE tokenizer (from the OLMo lineage) with:

- Vocabulary size: 50,368
- Special tokens: `[CLS]` (50281), `[SEP]` (50282), `[PAD]` (50283), `[MASK]` (50284)
- No `token_type_ids` (unlike BERT/RoBERTa)

The tokenizer is loaded via `AutoTokenizer.from_pretrained()` and is fully compatible
with InvPT's data pipeline — `DataCollatorForLanguageModeling` works without changes
since it only requires `pad_token_id` and `special_tokens_mask`.

## Downstream Evaluation

All 7 downstream tasks support ModernBERT via the `--model_type modernbert` flag.
The `MODEL_CLASSES` dict in each `run.py` maps `"modernbert"` to Auto classes:

```python
MODEL_CLASSES = {
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizer),
    "modernbert": (AutoConfig, AutoModel, AutoTokenizer),
}
```

For classification tasks (Defect-detection, Code-classification), `AutoModelForSequenceClassification`
is used instead of `AutoModel`.

Example downstream usage:

```bash
cd downstream/Clone-detection-POJ-104
python ./code/run.py \
    --model_type=modernbert \
    --model_name_or_path=saved_models/InvPT-ModernBERT-base/final \
    --tokenizer_name=saved_models/InvPT-ModernBERT-base/final \
    --do_train --do_test \
    --train_data_file=./dataset/train.jsonl \
    --eval_data_file=./dataset/valid.jsonl \
    --test_data_file=./dataset/test.jsonl \
    --block_size 400 \
    --train_batch_size 8 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --epoch 2
```

Note: the downstream `model.py` wrappers extract embeddings via `outputs[0][:, 0, :]`
(CLS token) as a fallback when no pooled output is available. `ModernBertModel` returns
`BaseModelOutput` with only `last_hidden_state`, so the CLS fallback is used. This is
acceptable for fine-tuning where the model learns task-specific representations, but
mean pooling may be worth implementing in downstream tasks for consistency.

## Requirements

- `transformers >= 4.48.0` (ModernBERT was added in this release)
- `flash-attn` (optional, recommended for GPU efficiency)

Install:

```bash
uv add "transformers>=4.48"
pip install flash-attn  # optional
```

## Saved Model Format

The pre-trained model is saved as a standard HuggingFace checkpoint:

```python
unwrapped.mlm_model.save_pretrained(save_path)  # saves ModernBertForMaskedLM
tokenizer.save_pretrained(save_path)
```

The saved checkpoint is loadable with:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained("saved_models/InvPT-ModernBERT-base/final")
tokenizer = AutoTokenizer.from_pretrained("saved_models/InvPT-ModernBERT-base/final")
```

The `config.json` in the saved directory contains `"model_type": "modernbert"`, so
Auto classes automatically resolve to the correct ModernBERT implementation.
