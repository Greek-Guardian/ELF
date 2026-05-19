"""Dataset loading and DataLoader construction (PyTorch).

Translated from src/utils/data_utils.py (JAX).
JAX/jax.process_count → torch.distributed equivalents.
"""

import json
from typing import Dict, Optional

import numpy as np
from datasets import DatasetDict
from datasets import load_dataset as hf_load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0


# ---------------------------------------------------------------------------
# DDP rank helpers (safe even when distributed is not initialised)
# ---------------------------------------------------------------------------

def _world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# Padding token helpers
# ---------------------------------------------------------------------------

def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = (
        tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    )
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def prepare_batch(batch: Dict, config, device=None) -> Dict:
    """Move numpy batch to torch tensors on `device` and sample label-drop mask."""
    import torch

    result = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.tensor(v, device=device)
        else:
            result[k] = v

    input_ids = result["input_ids"]
    batch_size = input_ids.shape[0]

    label_drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    if config.label_drop_prob > 0:
        label_drop_mask = torch.rand(batch_size, device=device) < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


# ---------------------------------------------------------------------------
# Sequence padding / truncation
# ---------------------------------------------------------------------------

def pad_and_truncate(ids_list, target_len: int, pad_token_id: int):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate(
                [ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)]
            )
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
) -> DataLoader:
    """Create a PyTorch DataLoader with an optional DistributedSampler."""

    def collate_fn(batch_list):
        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])
                if max_input_seq_length is not None:
                    cond = cond[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
    )
    if distributed and _world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=_world_size(),
            rank=_rank(),
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


# ---------------------------------------------------------------------------
# JSONL eval dataset
# ---------------------------------------------------------------------------

def load_jsonl_dataset(path: str, tokenizer, input_key="input", output_key="output"):
    """Load a JSONL eval set (one {input, output} per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(
                    data[input_key], add_special_tokens=False
                )["input_ids"],
                "input_ids": tokenizer(
                    data[output_key], add_special_tokens=False
                )["input_ids"],
            })
    return examples


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _looks_like_save_to_disk_arrow(ds) -> bool:
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir=None):
    """Load a dataset from HF Hub or local Arrow format."""
    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(
                f"Expected dataset at {path!r} to have a single split, got {splits}."
            )
        ds = ds[splits[0]]

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            "re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(
            repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir,
        )
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(
                    f"Expected dataset at {path!r} to have a single split, got {splits}."
                )
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")

    return train_dataset, eval_dataset
