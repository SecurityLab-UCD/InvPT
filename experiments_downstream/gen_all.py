#!/usr/bin/env python3
"""Generate downstream evaluation scripts for all model+task combinations."""

import stat
from pathlib import Path

BASE = Path(__file__).parent

models = [
    # (short_name, model_path, tokenizer, model_type)
    # Baselines
    ("codebert", "microsoft/codebert-base", "microsoft/codebert-base", "roberta"),
    (
        "graphcodebert",
        "microsoft/graphcodebert-base",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    (
        "contrabert_c",
        "./saved_models/ContraBERT_C",
        "microsoft/codebert-base",
        "roberta",
    ),
    (
        "contrabert_g",
        "./saved_models/ContraBERT_G",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    # Our trained models
    (
        "inv-codebert",
        "./saved_models/InvCodeBERT-supcon",
        "microsoft/codebert-base",
        "roberta",
    ),
    (
        "inv-graphcodebert",
        "./saved_models/InvGraphCodeBERT-supcon",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    (
        "inv-contrabert_c",
        "./saved_models/InvContraBERT_C-supcon/final",
        "microsoft/codebert-base",
        "roberta",
    ),
    (
        "inv-contrabert_g",
        "./saved_models/InvContraBERT_G-supcon/final",
        "microsoft/graphcodebert-base",
        "roberta",
    ),
    (
        "inv-modernbert",
        "./saved_models/aug-only/InvModernBERT-supcon/final",
        "answerdotai/ModernBERT-base",
        "modernbert",
    ),
]

# These are the literal shell lines we want in the output.
# Using chr(36) to produce '$' so no shell can possibly interpret them.
D = chr(36)  # dollar sign
SD_LINE = f'SCRIPT_DIR="{D}(cd "{D}(dirname "{D}{{BASH_SOURCE[0]}}")" && pwd)"'
RD_LINE = f'ROOT_DIR="{D}(cd "{D}SCRIPT_DIR/../.." && pwd)"'


def model_ref(model_path: str) -> str:
    """Return the shell expression for the model path."""
    if model_path.startswith("./"):
        return f'"{D}ROOT_DIR/{model_path[2:]}"'
    return model_path


def tokenizer_ref(tokenizer_path: str) -> str:
    """Return the shell expression for the tokenizer path."""
    if tokenizer_path.startswith("./"):
        return f'"{D}ROOT_DIR/{tokenizer_path[2:]}"'
    return tokenizer_path


def make_script(
    task_dir: str,
    short: str,
    mpath: str,
    subset: str | None = None,
    args_extra: str = "",
) -> str:
    mr = model_ref(mpath)
    op = f'"{D}ROOT_DIR/results/{short}/{task_dir}"'
    sl = f" ({subset})" if subset else ""
    return "\n".join(
        [
            "#!/bin/bash",
            f"# Downstream evaluation: {task_dir}{sl} with {short}",
            "set -euo pipefail",
            "",
            "# Parse CUDA device argument (default: 0)",
            'CUDA_DEVICE="${1:-0}"',
            "",
            SD_LINE,
            RD_LINE,
            "",
            'export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"',
            "",
            f'cd "{D}ROOT_DIR/downstream/{task_dir}"',
            f"./run.sh {mr} {op}{args_extra}",
            "",
        ]
    )


def write_script(task_dir: str, fname: str, content: str) -> None:
    dp = BASE / task_dir
    dp.mkdir(parents=True, exist_ok=True)
    fp = dp / fname
    fp.write_text(content)
    fp.chmod(fp.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


count = 0
for short, mpath, tok, mtype in models:
    tok_ref = tokenizer_ref(tok)
    # Clone-detection-POJ104: run.sh <model> <output> <model_type> <tokenizer>
    write_script(
        "Clone-detection-POJ104",
        f"clone-poj104_{short}.sh",
        make_script(
            "Clone-detection-POJ104", short, mpath, args_extra=f" {mtype} {tok_ref}"
        ),
    )
    count += 1

    # Clone-detection-CodeNet: run.sh <model> <save_path> <subset> <model_type> <tokenizer>
    for sub in ["Java250", "Python800", "C++1400"]:
        write_script(
            "Clone-detection-CodeNet",
            f"clone-codenet_{sub}_{short}.sh",
            make_script(
                "Clone-detection-CodeNet",
                short,
                mpath,
                subset=sub,
                args_extra=f" {sub} {mtype} {tok_ref}",
            ),
        )
        count += 1

    # Clone-detection-BigCloneBench: run.sh <model> <output>
    write_script(
        "Clone-detection-BigCloneBench",
        f"clone-bcb_{short}.sh",
        make_script("Clone-detection-BigCloneBench", short, mpath),
    )
    count += 1

    # Code-classification-POJ104: run.sh <model> <save_path> <subset> <model_type> <tokenizer>
    write_script(
        "Code-classification-POJ104",
        f"cls-poj104_Cpp_{short}.sh",
        make_script(
            "Code-classification-POJ104",
            short,
            mpath,
            subset="Cpp",
            args_extra=f" Cpp {mtype} {tok_ref}",
        ),
    )
    count += 1

    # Code-classification-CodeNet: run.sh <model> <save_path> <subset> <model_type> <tokenizer>
    for sub in ["Java250", "Python800", "C++1400"]:
        write_script(
            "Code-classification-CodeNet",
            f"cls-codenet_{sub}_{short}.sh",
            make_script(
                "Code-classification-CodeNet",
                short,
                mpath,
                subset=sub,
                args_extra=f" {sub} {mtype} {tok_ref}",
            ),
        )
        count += 1

    # Defect-detection: run.sh <model> <output>
    write_script(
        "Defect-detection",
        f"defect_{short}.sh",
        make_script("Defect-detection", short, mpath),
    )
    count += 1

    # Code-translation: run.sh <model> <output>
    write_script(
        "Code-translation",
        f"translation_{short}.sh",
        make_script("Code-translation", short, mpath),
    )
    count += 1

print(f"Generated {count} scripts in {BASE}")
