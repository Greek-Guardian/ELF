# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ELF (Embedded Language Flows) is the official JAX/Flax implementation of a continuous diffusion language model trained with Flow Matching. Targeted at TPU (v5p-class). All entry points run from `src/`.

## Commands

All commands assume `cd src/` first. `train.py` and `eval.py` insert the repo root onto `sys.path` automatically and call `jax.distributed.initialize()` before any JAX import — keep that ordering when adding new entry points.

```bash
# Install (TPU JAX wheels)
pip install -r requirements.txt

# Train
python train.py --config configs/training_configs/train_owt_ELF-B.yml

# Evaluate (HF repo id or local checkpoint dir/file)
python eval.py \
    --config configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt

# Override any Config field from the CLI (repeatable)
python eval.py --config <yml> --checkpoint_path <path> \
    --config_override global_batch_size=64 \
    --config_override num_samples=200
```

There is no test suite, linter, or build step. Use the unconditional eval as a smoke test (expects Gen. PPL ≈ 24, entropy ≈ 5.15 for ELF-B at 32 SDE steps).

## Architecture

### Two-objective training

A single transformer (`modules/model.py:ELF`) is trained on a mixture of:

1. **Denoiser (L2)** — flow-matching loss in continuous T5 embedding space. `train_step.py` samples `t ~ logit_normal`, builds `z = t*x0 + (1-t)*noise*denoiser_noise_scale`, and regresses the velocity.
2. **Decoder (CE)** — cross-entropy on token logits produced by a *factored unembedding* (`hidden -> text_encoder_dim -> vocab`), gated by `decoder_step_active`.

`config.decoder_prob` chooses the branch per step. The decoder branch is what makes ELF "continuous until the last step" — discretization only happens via this shared-weight decoder head at `t=1`.

### Frozen T5 encoder

`modules/t5_encoder.py` provides a JAX port of T5-small. Weights are loaded once from a `.pkl` (`encoder_checkpoint`, default pulled from HF) and held *outside* the `TrainState` — they're passed as a separate `encoder_params` arg into `train_step` and never updated. `latent_mean`/`latent_std` rescale encoder outputs into the diffusion space; changing the encoder requires re-fitting these.

### Conditioning model

For conditional tasks (translation, summarization), the data collator concatenates `condition_input_ids + input_ids` into a single sequence and produces a `cond_seq_mask`. The denoiser preserves cond positions verbatim (`add_noise` in `utils/sampling_utils.py` overrides noised values where `cond_seq_mask=1`). `label_drop_prob` masks attention from x-tokens to cond-tokens for CFG training — done *before* encoding so the encoder also sees the unconditional view.

### Prefix-token conditioning

`ELF.build_context` prepends learnable prefix tokens (time, self-cond CFG scale, and optional model-mode tokens) instead of using AdaLN. `TextRotaryEmbeddingFast` is told the prefix length via `num_empty_token` so RoPE skips them. When editing the model, the prefix length math (`prefix_len + model_mode_offset`) drives both rope offsetting and the final `x[:, prefix_len + model_mode_offset:]` slice — keep them in sync.

### Sampling

Sampling configs are decoupled from training configs. Each training YAML points at `sampling_configs_path`, and the evaluator iterates *every* entry in that file (each entry = one `SamplingConfig`: sampler `ode`/`sde`, steps, CFG list, SC-CFG list, time schedule). To add a sampling sweep, append entries to the sampling YAML — no code change needed.

## Config system

- `configs/config.py:Config` is a plain class (not a dataclass) holding all defaults. Annotations on `None`-valued fields are used by `apply_config_overrides` to coerce CLI strings.
- `load_config_from_yaml` only sets fields that already exist on `Config` — typos in YAML are silently ignored.
- `--config_override field=value` is the standard escape hatch; prefer it over editing YAMLs for one-off runs. Use `field=none` to set back to `None`.
- `batch_size` is per-host, `global_batch_size` is across all hosts. Eval derives one from the other; if both are unset it errors.

## Checkpointing

- Saved to `output_dir/checkpoint_<step>` at end of each epoch (or fractional via `save_freq < 1`); only process 0 writes.
- `--resume` is not required — training auto-detects the latest checkpoint in `output_dir` via `find_latest_checkpoint`.
- `load_checkpoint` accepts a local path, a local directory (uses latest inside), or an HF repo id (e.g. `embedded-language-flows/ELF-B-owt`). The encoder `.pkl` is loaded separately by `load_encoder_checkpoint`.
- If `hf_repo_id` is set, the entire `output_dir` is mirrored to HF after each save.

## Data format

Datasets are HuggingFace `Dataset` objects pre-tokenized with the T5 tokenizer.

- **Unconditional**: each example has `input_ids`.
- **Conditional**: each example has `input_ids` (target) and `condition_input_ids` (source). The collator handles concatenation and mask construction.
- For evaluation only, JSONL files (`{"input": ..., "output": ...}` per line) are supported via `load_jsonl_dataset` in `utils/data_utils.py` — the loader switches on `.jsonl` extension.
- `data_path` accepts an HF repo id, a local `save_to_disk` directory, or a JSONL file (eval only).

## Conventions to preserve

- `jax.distributed.initialize()` runs before any other JAX import in entry points. Wrap in try/except for single-host runs.
- The `jax.random.split(rng, 11)` in `train_step.py` is intentionally 11-way (not 10) to preserve the RNG stream of released checkpoints. Don't "clean up" the unused splits.
- The Muon optimizer (`config.optimizer = "muon"`) is the default for ELF-B training; `blr` is *base* learning rate and the effective LR is `blr * batch_size / 256`.

## Running on a single 8x H800 node (Hopper, sm_90)

The codebase is TPU-first but works on a CUDA 12 single-host setup with no source changes — the `try/except` around `jax.distributed.initialize()` already handles single-host, and `pmap` distributes across all visible GPUs.

- **Install**: `requirements.txt` already pulls `jax[cuda12_pip]==0.4.38`; the host only needs a recent NVIDIA driver.
- **Launch**: use [run_h800.sh](run_h800.sh) — it sets `XLA_PYTHON_CLIENT_MEM_FRACTION`, latency-hiding scheduler, async collectives, Triton GEMM, and `cd`s into `src/` so config paths line up.
- **Config**: [src/configs/training_configs/train_owt_ELF-B_h800.yml](src/configs/training_configs/train_owt_ELF-B_h800.yml) drops `global_batch_size` from 512 → 128 and sets `grad_accum_steps=4` so the effective batch and the resulting `blr * batch / 256` LR match the v5p-64 paper run. If you OOM, halve the batch and double grad-accum; if you have headroom, raise the batch first.
- **bf16**: not wired through the model. If GPU throughput is bandwidth-bound, add `jax.config.update("jax_default_matmul_precision", "bfloat16")` in the entry points — leave the parameter dtype alone. Validate that L2/CE losses still converge before committing.
