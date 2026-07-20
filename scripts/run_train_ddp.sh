#!/usr/bin/env bash
# Launch Unsloth training with Distributed Data Parallel (DDP).
#
# WHY DDP (not model-parallel):
#   device_map='auto' / balanced-split shards one model copy across both GPUs.
#   Activations must flow GPU0 -> GPU1 sequentially, so at any instant only ONE
#   card does useful work (throughput = 1 card). The cards alternate and one is
#   always idle -- exactly the behavior we observed.
#
#   DDP loads a FULL model copy on each GPU and each processes different data.
#   Both cards work simultaneously -> throughput ~= 2x single-card (minus a tiny
#   all-reduce over NVLink/PCIe; hidden=5120 bf16 -> ~1.3MB/step, negligible).
#
# This requires the per-card memory budget to hold: full 4-bit model (~14GB for
# 27B) + LoRA params + optimizer state + activations. The new torch 2.7 + fla
# stack uses less activation memory than the old stack, which is why we retest.
#
# Usage:
#   bash scripts/run_train_ddp.sh              # full epoch, eval/save every 1000
#   MAX_STEPS=20 bash scripts/run_train_ddp.sh # quick test
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# --- venv ------------------------------------------------------------------
export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
PY_BIN="$PROJECT_ROOT/.venv/bin/python"

# --- NixOS runtime library paths (same as run_train.sh) --------------------
LIBSTDCXX_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
SW_LIB="/run/current-system/sw/lib"
OGL_LIB="/run/opengl-driver/lib"
export LD_LIBRARY_PATH="${LIBSTDCXX_DIR}:${SW_LIB}:${OGL_LIB}:${LD_LIBRARY_PATH:-}"

# --- Triton on NixOS -------------------------------------------------------
export TRITON_PTXAS_PATH="${SW_LIB}/../bin/ptxas"
export TRITON_LIBCUDA_PATH="${OGL_LIB}"

# --- CUDA 12.8 toolkit (matches torch 2.7.1+cu128) -------------------------
CUDA_12_8_HOME="$(ls -d /nix/store/*-cuda-merged-12.8 2>/dev/null | head -1 || true)"
if [[ -n "$CUDA_12_8_HOME" ]] && [[ -x "$CUDA_12_8_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$CUDA_12_8_HOME"
fi

# --- torch.compile / inductor off (fragile on NixOS) -----------------------
export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

# --- DDP config ------------------------------------------------------------
# nproc_per_node = number of model copies = number of GPUs. Each gets one full
# 4-bit model copy. CUDA_VISIBLE_DEVICES restricts which physical cards are used.
NPROC="${NPROC:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"

# --- training params (defaults = 1 full epoch) -----------------------------
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-/home/dudu/datasets/cyb3r-dataset}"
OUT_DIR="${OUT_DIR:-outputs/cyb3r-reasoning}"
MAX_STEPS="${MAX_STEPS:-3565}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
BS="${BS:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"

echo ">> launching DDP training (nproc_per_node=$NPROC)"
echo ">> model:      $MODEL_NAME"
echo ">> devices:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ">> data:       $DATA_DIR"
echo ">> output:     $OUT_DIR"
echo ">> max_steps:  $MAX_STEPS  eval_steps: $EVAL_STEPS  save_steps: $SAVE_STEPS"
echo ">> effective batch: $BS x $GRAD_ACCUM x $NPROC = $((BS * GRAD_ACCUM * NPROC)) rows/step"

exec torchrun --nproc_per_node="$NPROC" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
  scripts/train_unsloth_ddp.py \
  --model-name "$MODEL_NAME" \
  --train-file "$DATA_DIR/train.jsonl" \
  --eval-file "$DATA_DIR/eval.jsonl" \
  --output-dir "$OUT_DIR" \
  --max-seq-length 4096 \
  --max-steps "$MAX_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --learning-rate "$LR" \
  --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" \
  --per-device-train-batch-size "$BS" \
  --gradient-accumulation-steps "$GRAD_ACCUM"
