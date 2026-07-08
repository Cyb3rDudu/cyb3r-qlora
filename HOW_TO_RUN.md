# How To Run On carrier

These instructions are for the box named `carrier` (NixOS, 2× RTX 3090 with
NVLink, NVIDIA driver exposing CUDA 13.x). The repo is expected at:

- `/home/dudu/Code/cyb3r-qlora`

and the local training dataset workspace at:

- `/home/dudu/datasets/cyb3r-dataset`

## 0. Prerequisites (already satisfied on carrier)

The NixOS configuration (`carrier-nixos/packages.nix`) already provides:

- `uv`, `python3` (3.13 interpreter, used to build the 3.12 venv below)
- `cudaPackages.cudatoolkit` (provides `ptxas` / `nvcc`, currently CUDA 12.9)
- `gcc13`, `git`, `binutils`, etc.

The NVIDIA driver ships `libcuda.so.1` under `/run/opengl-driver/lib`, and
`libz.so` / `ldconfig` live under `/run/current-system/sw/lib`. `run_train.sh`
adds all of these to `LD_LIBRARY_PATH` at launch — you do not need to do it by
hand.

## 1. Prepare the repo

```bash
mkdir -p /home/dudu/Code
cd /home/dudu/Code
git clone git@github.com:Cyb3rDudu/cyb3r-qlora.git
cd cyb3r-qlora
```

## 2. Prepare the local dataset workspace

Create a local dataset directory outside git:

```bash
mkdir -p /home/dudu/datasets/cyb3r-dataset
```

Place a local selection manifest at:

```bash
/home/dudu/datasets/cyb3r-dataset/source_selection.json
```

The manifest defines abstract trace buckets and local file paths. Example:

```json
{
  "name": "cyb3r_reasoning_test",
  "reasoning_sets": [
    {
      "trace_type": "distilled_exploitability_reasoning",
      "quota": 6000,
      "files": ["/absolute/path/to/local/source.jsonl"]
    }
  ],
  "policy_sets": [
    {
      "trace_type": "branch_ranking",
      "quota": 1400,
      "files": ["/absolute/path/to/local/policy.jsonl"],
      "categories": ["branch_ranking"]
    }
  ]
}
```

## 3. Build the local training subset

This writes the generated subset into `/home/dudu/datasets/cyb3r-dataset`.

```bash
python3 scripts/build_reasoning_dataset.py \
  --selection-manifest /home/dudu/datasets/cyb3r-dataset/source_selection.json \
  --output-dir /home/dudu/datasets/cyb3r-dataset
```

Expected outputs:

- `/home/dudu/datasets/cyb3r-dataset/train.jsonl`
- `/home/dudu/datasets/cyb3r-dataset/eval.jsonl`
- `/home/dudu/datasets/cyb3r-dataset/manifest.json`

Current expected counts from the present local build:

- `28,522` train rows
- `1,478` eval rows

## 4. Build the training environment

Build the venv with the provided script. It installs the pinned
`unsloth[cu121-torch251]` stack with `uv`, then runs
`scripts/patch_triton_nixos.py` to make the Triton wheel work on NixOS (see
the **Why the Triton patch is needed** note below — it is not optional).

```bash
bash scripts/setup_train_env.sh
```

This creates `.venv/` (Python 3.12) and prints a confirmation. You only need
to re-run it if `.venv/` is deleted or if `unsloth` / `torch` are upgraded —
re-running always re-applies the Triton patch (it is idempotent).

If `uv` is missing, enter a Nix shell first:

```bash
nix shell nixpkgs#uv nixpkgs#python3
bash scripts/setup_train_env.sh
```

## 5. Start training

```bash
bash scripts/run_train.sh
```

`run_train.sh` does everything else: it sets the NixOS library paths and
Triton env vars, disables the fragile `torch.compile`/inductor path, and
launches single-process model-parallel training.

There is **no `accelerate config` step and no `accelerate launch`**. Training
runs as one process that shards the model across both GPUs. Multi-process DDP
(one full 18 GB 4-bit model per card) OOMs on this dataset's long rows.

### What it trains

- Model: `Qwen/Qwen3.6-27B` (override with `MODEL_NAME=...`)
- Dataset: the local subset from step 3
- Output: `outputs/cyb3r-reasoning-test/`
- 500 steps, seq length 4096, QLoRA r=64 / α=128, bf16, `adamw_8bit`
- Effective batch size 8 (per-device 1 × gradient-accumulation 8)

### GPU placement (balanced split)

By default the script builds an explicit per-layer `device_map` that splits
the 64 language layers **32 + 32** across the two 3090s, with the embedding
on GPU 0 and the LM head on GPU 1. This balances compute and thermals evenly
(~9 GB / ~16 GB per card, both fans sharing the load).

This is on by default. To use HuggingFace's `device_map="auto"` instead
(packs the first GPU, overflows to the second — leaves one card hot/idle):

```bash
accelerate launch --num_processes 2 scripts/train_unsloth.py \
  --model-name Qwen/Qwen3.6-27B \
  --train-file /home/dudu/datasets/cyb3r-dataset/train.jsonl \
  --eval-file /home/dudu/datasets/cyb3r-dataset/eval.jsonl \
  --output-dir outputs/cyb3r-reasoning-test \
  --max-seq-length 4096 \
  --max-steps 500 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
```

To train on a single GPU only:

```bash
CUDA_VISIBLE_DEVICES=0 BALANCED_SPLIT=0 DEVICE_MAP=cuda:0 bash scripts/run_train.sh
```

### Resume from the latest checkpoint

```bash
RESUME=1 bash scripts/run_train.sh
```

Checkpoints are saved every `100` steps under `outputs/cyb3r-reasoning-test/checkpoint-*`.

## 6. What to evaluate

Check whether the tuned model is better at:

- global reassessment
- branch ranking
- stopping low-yield rabbit holes
- preserving tool-call competence

Use `scripts/eval_prompts.md` for manual before/after comparisons.

## Notes

- Use the original model checkpoint for QLoRA training. The local Q8 weights are kept for inference because that is the practical fit for this hardware at runtime, but they are not the right artifact for adapter training.
- This repo is pinned to `Qwen/Qwen3.6-27B`, the official public post-trained 27B dense checkpoint in the Qwen3.6 family.
- Keep the adapter separate first; merge later only if needed.
- The local dataset under `/home/dudu/datasets/cyb3r-dataset` is outside git
  and stays untracked.
- If a run is interrupted, `RESUME=1` restarts from the latest saved
  checkpoint, not the exact last in-memory step.

## Why the Triton patch is needed

Triton's pip wheel makes three assumptions that are false on NixOS. The
env vars set by `run_train.sh` plus `scripts/patch_triton_nixos.py` cover all
three:

1. Triton ships a bundled `ptxas` binary under
   `triton/backends/nvidia/bin/`. NixOS refuses to run generic-Linux
   dynamically-linked ELF binaries (no `/lib64/ld-linux-x86-64.so.2`). We
   point Triton at the system `ptxas` via `TRITON_PTXAS_PATH`.
2. Triton's `ptx_get_version()` maps CUDA `12.9` → PTX `8.9`, but NVIDIA's
   CUDA-12.x `ptxas` only accepts up to PTX `8.8` (PTX 8.9 needs CUDA
   12.10+). The patch caps the returned version at `8.8`.
3. Triton's `libcuda_dirs()` hardcodes `/sbin/ldconfig`, which does not exist
   on NixOS. `TRITON_LIBCUDA_PATH` short-circuits the lookup, and the patch
   makes the function fall back to `shutil.which("ldconfig")` and tolerate a
   failing `ldconfig -p`.

If you ever upgrade `torch`/`triton`/`unsloth`, re-run
`bash scripts/setup_train_env.sh` (or just `.venv/bin/python
scripts/patch_triton_nixos.py`) — the patcher detects already-applied edits
and is safe to re-run.
