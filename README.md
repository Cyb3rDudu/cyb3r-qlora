# cyb3r-qlora

Reasoning-focused QLoRA fine-tuning project for the `Qwen/Qwen3.6-27B` security model named **cyb3r**.

## Project scope

This repository documents a compact fine-tuning run focused on improving:

- structured security reasoning
- global reassessment during multi-step tasks
- branch ranking across attack paths
- reduced rabbit-hole fixation
- preserved tool-call competence

The target is not a generic assistant. The target is a security model that reasons more like a disciplined operator.

## Model target

- Base family: `Qwen/Qwen3.6-27B`
- Runtime inference variant: Q8 weights
- Training method: 4-bit QLoRA on the original model checkpoint
- Hardware: 2× RTX 3090 with NVLink

## Short instruction summary

1. Build a 25k reasoning-first subset.
2. Add 5k agent-policy rows.
3. Train QLoRA 4-bit on the original 27B checkpoint and keep the Q8 export for inference.
4. Use 4096 context.
5. Run 500 steps.
6. Evaluate for:
   - better global reassessment
   - better branch ranking
   - less brute-force fixation
   - preserved tool-call competence

## Repository layout

```text
cyb3r-qlora/
├── README.md
├── PLAN.md
├── DATASET_AUDIT.md
├── HOW_TO_RUN.md
├── scripts/
│   ├── setup_train_env.sh        # build .venv (uv) + patch triton for NixOS
│   ├── patch_triton_nixos.py     # idempotent triton wheel patches
│   ├── run_train.sh              # launch single-process model-parallel run
│   ├── train_unsloth.py          # unsloth QLoRA + SFTTrainer entrypoint
│   ├── build_reasoning_dataset.py
│   ├── select_subset.py
│   ├── eval_prompts.md
│   └── unsloth_config_example.md
└── samples/
```

`outputs/`, `.venv/`, and the local dataset are gitignored (see `.gitignore`).

## How to run

Full instructions live in [HOW_TO_RUN.md](HOW_TO_RUN.md). The short version
on `carrier`:

```bash
bash scripts/setup_train_env.sh   # one-time: build venv + patch triton
bash scripts/run_train.sh          # train
RESUME=1 bash scripts/run_train.sh # resume from latest checkpoint
```

The run shards the 27B 4-bit model across both 3090s (32+32 layers by default)
and trains single-process; there is no `accelerate launch` / DDP step.

## Current status

The training pipeline runs end-to-end on `carrier`. The committed scripts
build the NixOS-compatible venv, patch Triton, and launch a single-process
model-parallel QLoRA run on the two 3090s.

This repository documents:

- the target fine-tuning shape
- the small test-run plan
- helper scripts for subset prep, inspection, and NixOS-compatible training
- representative dataset samples only
- a generated local subset under `/home/dudu/datasets/cyb3r-dataset`, which
  stays outside git

No full dataset is stored here.

## Current local subset shape

When built locally on `carrier`, the current subset produces:

- `28,522` train rows
- `1,478` eval rows
- `25,000` reasoning-first rows
- `5,000` agent-policy rows
