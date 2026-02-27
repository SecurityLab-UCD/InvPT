import json

import matplotlib.lines
import matplotlib.pyplot as plt
import torch
import typer
from sklearn.manifold import TSNE
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Row 1: CodeBERT-based models; Row 2: GraphCodeBERT-based models
ROW1_MODELS = {
    "CodeBERT": "microsoft/codebert-base",
    "ContraBERT_C": "../saved_models/ContraBERT_C",
    "InvCodeBERT": "../saved_models/InvCodeBERT-supcon/final",
}
ROW2_MODELS = {
    "GraphCodeBERT": "microsoft/graphcodebert-base",
    "ContraBERT_G": "../saved_models/ContraBERT_G",
    "InvGraphCodeBERT": "../saved_models/InvGraphCodeBERT-supcon/final",
}


def to_2d(vectors, **tsne_kw):
    tsne = TSNE(
        n_components=2,
        perplexity=30,  # try 5-50; smaller = tighter clusters
        init="pca",
        random_state=0,
        **tsne_kw,
    )
    return tsne.fit_transform(vectors)  # [N, 2]


def embed_batch(model, tok, texts, max_len=512, batch_size=32):
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        toks = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**toks).last_hidden_state  # [bs, L, H]
            cls = out[:, 0]  # use [CLS]; or mean-pool
            vecs.append(cls.cpu())
    return torch.cat(vecs).numpy()  # [N, H]


def main(
    input_test_file: str = "dataset/aug_test.jsonl",
    output_file: str = "clusters.png",
    pids: str | None = "82,85,91,95,100",
    legend: bool = True,
):
    rows = [ROW1_MODELS, ROW2_MODELS]
    ncols = max(len(r) for r in rows)

    with open(input_test_file) as f:
        data = [json.loads(line) for line in f]

    target_labels = sorted(map(int, pids.split(",")) if pids else range(81, 105))

    programs, labels = [], []
    for item in data:
        label = int(item["label"])
        if label not in target_labels:
            continue
        programs.append(item["code"])
        labels.append(label)

    # Check which labels actually exist in the data
    actual_labels = sorted(set(labels))
    print(f"Target labels: {target_labels}")
    print(f"Labels found in data: {actual_labels}")
    print(f"Missing labels: {set(target_labels) - set(actual_labels)}")

    # Separate tokenizers for each model family
    codebert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    graphcodebert_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/graphcodebert-base"
    )
    tokenizers = {0: codebert_tokenizer, 1: graphcodebert_tokenizer}

    # Load all encoders
    all_models = {}
    for row_models in rows:
        all_models.update(row_models)
    encoders = {
        n: AutoModel.from_pretrained(p).to(DEVICE).eval() for n, p in all_models.items()
    }

    # Compute embeddings per row (using corresponding tokenizer)
    coords = {}
    for row_idx, row_models in enumerate(rows):
        tok = tokenizers[row_idx]
        for name in row_models:
            emb = embed_batch(encoders[name], tok, programs)
            coords[name] = to_2d(emb)

    # Use the number of target labels for consistent coloring
    NUM_CLASSES = len(target_labels)
    cmap = plt.get_cmap("tab10", NUM_CLASSES)

    # Create a mapping from label to color index
    label_to_color_idx = {label: i for i, label in enumerate(target_labels)}

    fig, axes = plt.subplots(
        2, ncols, figsize=(4.2 * ncols, 4.2 * 2), sharex=False, sharey=False
    )

    for row_idx, row_models in enumerate(rows):
        for col_idx, (name, _) in enumerate(row_models.items()):
            ax = axes[row_idx, col_idx]
            xy = coords[name]
            colors = [label_to_color_idx[label] for label in labels]
            ax.scatter(xy[:, 0], xy[:, 1], c=colors, cmap=cmap, s=8, alpha=0.9)
            ax.set_xlabel(name, fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(False)
        # Hide unused axes in this row
        for col_idx in range(len(row_models), ncols):
            axes[row_idx, col_idx].set_visible(False)

    # Create legend for all target labels (including missing ones)
    if legend:
        handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=cmap(i),
                label=str(label),
                markersize=7,
            )
            for i, label in enumerate(target_labels)
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=NUM_CLASSES, title="Problem ID"
        )

    fig.tight_layout()
    plt.savefig(output_file, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(main)
