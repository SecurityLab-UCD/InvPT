from transformers import AutoTokenizer, AutoModel
import torch
import pathlib
import json
import fire
from tqdm import tqdm
import matplotlib.lines
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# put the models you want to compare into a dict
MODELS = {
    "GraphCodeBERT": "microsoft/graphcodebert-base",
    "ContraBERT_G": "../saved_models/ContraBERT_G",
    "InvBERT": "../saved_models/InvBERT",
    "InvContraBERT": "../saved_models/InvContraBERT",
}

from sklearn.manifold import TSNE


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
):
    with open(input_test_file) as f:
        data = [json.loads(line) for line in f]

    numbers = [1, 4, 10, 14, 17, 22]
    target_labels = set(map(lambda x: 81 + x, numbers))
    programs, labels = [], []
    for item in data:
        label = int(item["label"])
        if label not in target_labels:
            continue
        programs.append(item["code"])
        labels.append(label)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    encoders = {
        n: AutoModel.from_pretrained(p).to(DEVICE).eval() for n, p in MODELS.items()
    }

    embeddings = {}  # model name → [N, H]

    for name in MODELS:
        embeddings[name] = embed_batch(encoders[name], tokenizer, programs)

    coords = {n: to_2d(v) for n, v in embeddings.items()}

    NUM_CLASSES = len(set(labels))
    cmap = plt.get_cmap("tab10", NUM_CLASSES)
    fig, axes = plt.subplots(
        1, len(MODELS), figsize=(4.2 * len(MODELS), 4.2), sharex=False, sharey=False
    )

    for ax, (name, xy) in zip(axes, coords.items()):
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, cmap=cmap, s=8, alpha=0.9)
        ax.set_title(name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    # optional legend – one entry per class
    handles = [
        matplotlib.lines.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=cmap(i),
            label=l,
            markersize=7,
        )
        for i, l in enumerate(target_labels)
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=NUM_CLASSES, title="Problem ID"
    )

    fig.tight_layout()
    plt.savefig(output_file, dpi=500, bbox_inches="tight")


if __name__ == "__main__":
    fire.Fire(main)
