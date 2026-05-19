#!/usr/bin/env python
"""Download ELF datasets and checkpoints to local directories.

By default downloads everything needed for the ELF-B OpenWebText smoke test:
- The JAX T5-small encoder pickle (required for both training and eval)
- The OpenWebText T5-tokenized dataset (required for training; not needed for unconditional eval)
- The ELF-B-owt pretrained checkpoint (required for the eval sanity check)

Layout written under the repo root:
    ./assets_download/
        encoder/t5_small_encoder_jax.pkl
        datasets/<dataset-name>/...
        checkpoints/<model-name>/...

Usage:
    # Minimal: encoder + ELF-B-owt checkpoint (enough for unconditional eval)
    python download_assets.py --eval-only

    # Everything for ELF-B OpenWebText training + eval
    python download_assets.py --task owt --size B

    # Conditional tasks
    python download_assets.py --task de-en
    python download_assets.py --task xsum

    # Pick specific items
    python download_assets.py --encoder --dataset owt --checkpoint ELF-B-owt
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets_download"
ENCODER_DIR = ASSETS_DIR / "encoder"
DATASETS_DIR = ASSETS_DIR / "datasets"
CHECKPOINTS_DIR = ASSETS_DIR / "checkpoints"

HF_ORG = "embedded-language-flows"

ENCODER_REPO = f"{HF_ORG}/t5_small_encoder_jax"
ENCODER_FILE = "t5_small_encoder_jax.pkl"

DATASETS = {
    "owt": [f"{HF_ORG}/openwebtext-t5"],
    "de-en": [
        f"{HF_ORG}/wmt14_de-en_train_t5",
        f"{HF_ORG}/wmt14_de-en_validation_t5",
    ],
    "xsum": [
        f"{HF_ORG}/xsum_train_t5",
        f"{HF_ORG}/xsum_validation_t5",
    ],
}

CHECKPOINTS = {
    "ELF-B-owt": f"{HF_ORG}/ELF-B-owt",
    "ELF-M-owt": f"{HF_ORG}/ELF-M-owt",
    "ELF-L-owt": f"{HF_ORG}/ELF-L-owt",
    "ELF-B-de-en": f"{HF_ORG}/ELF-B-de-en",
    "ELF-B-xsum": f"{HF_ORG}/ELF-B-xsum",
}

TASK_DEFAULT_CHECKPOINT = {
    ("owt", "B"): "ELF-B-owt",
    ("owt", "M"): "ELF-M-owt",
    ("owt", "L"): "ELF-L-owt",
    ("de-en", "B"): "ELF-B-de-en",
    ("xsum", "B"): "ELF-B-xsum",
}


def download_encoder() -> Path:
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[encoder] {ENCODER_REPO}/{ENCODER_FILE} -> {ENCODER_DIR}")
    path = hf_hub_download(
        repo_id=ENCODER_REPO,
        filename=ENCODER_FILE,
        repo_type="model",
        local_dir=str(ENCODER_DIR),
    )
    return Path(path)


def download_dataset(repo_id: str) -> Path:
    name = repo_id.split("/")[-1]
    target = DATASETS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
    )
    return target


def download_checkpoint(name: str) -> Path:
    repo_id = CHECKPOINTS[name]
    target = CHECKPOINTS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    print(f"[checkpoint] {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target),
    )
    return target


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--task",
        choices=list(DATASETS.keys()),
        help="High-level task. Selects matching dataset + default checkpoint.",
    )
    p.add_argument("--size", choices=["B", "M", "L"], default="B", help="Model size (only used with --task)")
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Download only the encoder + checkpoint (skip the dataset). Eval is fine without the dataset for unconditional generation.",
    )
    p.add_argument("--encoder", action="store_true", help="Download the T5 encoder pickle")
    p.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        action="append",
        default=[],
        help="Download a dataset (repeatable)",
    )
    p.add_argument(
        "--checkpoint",
        choices=list(CHECKPOINTS.keys()),
        action="append",
        default=[],
        help="Download a pretrained checkpoint (repeatable)",
    )
    p.add_argument("--all", action="store_true", help="Download everything")
    return p.parse_args()


def main():
    args = parse_args()

    do_encoder = args.encoder
    datasets = list(args.dataset)
    checkpoints = list(args.checkpoint)

    if args.all:
        do_encoder = True
        datasets = list(DATASETS.keys())
        checkpoints = list(CHECKPOINTS.keys())

    if args.task:
        do_encoder = True
        if not args.eval_only:
            datasets.append(args.task)
        ckpt = TASK_DEFAULT_CHECKPOINT.get((args.task, args.size))
        if ckpt:
            checkpoints.append(ckpt)
        else:
            print(f"warning: no default checkpoint for task={args.task} size={args.size}", file=sys.stderr)

    # Default action: minimal eval setup
    if not (do_encoder or datasets or checkpoints):
        print("No options given; defaulting to --eval-only --task owt --size B")
        do_encoder = True
        checkpoints.append("ELF-B-owt")

    datasets = sorted(set(datasets))
    checkpoints = sorted(set(checkpoints))

    print(f"Plan: encoder={do_encoder}, datasets={datasets}, checkpoints={checkpoints}")
    print(f"Output root: {ASSETS_DIR}")

    if do_encoder:
        download_encoder()

    for d in datasets:
        for repo in DATASETS[d]:
            download_dataset(repo)

    for c in checkpoints:
        download_checkpoint(c)

    print("\nDone. Suggested config overrides for local assets:")
    print(f"  encoder_checkpoint={ENCODER_DIR / ENCODER_FILE}")
    if datasets:
        for d in datasets:
            for repo in DATASETS[d]:
                name = repo.split("/")[-1]
                print(f"  data_path={DATASETS_DIR / name}  (or eval_data_path)")
    if checkpoints:
        for c in checkpoints:
            print(f"  --checkpoint_path {CHECKPOINTS_DIR / c}")


if __name__ == "__main__":
    main()
