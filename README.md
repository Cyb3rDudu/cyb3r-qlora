# cyb3r-qlora

Reasoning-focused QLoRA fine-tuning project for a Qwen-class 27B security model named **cyb3r**.

## Project scope

This repository documents a compact fine-tuning run focused on improving:

- structured security reasoning
- global reassessment during multi-step tasks
- branch ranking across attack paths
- reduced rabbit-hole fixation
- preserved tool-call competence

The target is not a generic assistant. The target is a security model that reasons more like a disciplined operator.

## Model target

- Base family: Qwen-class 27B
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
└── samples/
```

## Current status

This repository is focused on documenting:

- the target fine-tuning shape
- the small test-run plan
- helper scripts for subset prep and inspection
- representative dataset samples only
- a generated local subset under `/home/dudu/Documents/cyb3r-dataset`, which stays outside git

No full dataset is stored here.

## Current local subset shape

When built locally on `carrier`, the current subset produces:

- `28,567` train rows
- `1,433` eval rows
- `25,000` reasoning-first rows
- `5,000` agent-policy rows
