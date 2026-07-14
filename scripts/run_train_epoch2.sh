#!/usr/bin/env bash
# Epoch 2+ launcher: continues training from a previous epoch's LoRA adapter.
#
# Difference from run_train.sh:
#   - Passes --load-adapter so train_unsloth.py loads the base model + the
#     saved adapter (continues training the existing LoRA weights instead of
#     creating fresh ones).
#   - Fresh cosine LR schedule (restarts from 1e-4) — standard for multi-epoch.
#
# Usage (called by auto_epoch_transition.sh, or manually):
#   LOAD_ADAPTER=outputs/cyb3r-reasoning-epoch1-final \
#   OUT_DIR=outputs/cyb3r-reasoning-epoch2 \
#   bash scripts/run_train_epoch2.sh
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
export TRITON_PTXAS_PATH="${SW_LIB}/../bin/ptxas"
export TRITON_LIBCUDA_PATH="${OGL_LIB}"

CUDA_12_8_HOME="$(ls -d /nix/store/*-cuda-merged-12.8 2>/dev/null | head -1 || true)"
if [[ -n "$CUDA_12_8_HOME" ]] && [[ -x "$CUDA_12_8_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$CUDA_12_8_HOME"
fi

export TORCHDYNAMO_DISABLE=1
export TORCH_COMPILE_DISABLE=1
export UNSLOTH_COMPILE_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-/home/dudu/datasets/cyb3r-dataset}"
OUT_DIR="${OUT_DIR:-outputs/cyb3r-reasoning-epoch2}"
LOAD_ADAPTER="${LOAD_ADAPTER:?LOAD_ADAPTER must point at the previous epoch adapter dir}"
MAX_STEPS="${MAX_STEPS:-3565}"
EVAL_STEPS="${EVAL_STEPS:-800}"
SAVE_STEPS="${SAVE_STEPS:-100}"

echo ">> launching EPOCH 2 (continuing from $LOAD_ADAPTER)"
echo ">> model:      $MODEL_NAME"
echo ">> adapter:    $LOAD_ADAPTER"
echo ">> output:     $OUT_DIR"
echo ">> max_steps:  $MAX_STEPS  eval_steps: $EVAL_STEPS  save_steps: $SAVE_STEPS"

PACKING_ARGS=()
DEVICE_MAP_ARGS=(--balanced-split)

exec "$PY_BIN" scripts/train_unsloth.py \
  --model-name "$MODEL_NAME" \
  --load-adapter "$LOAD_ADAPTER" \
  --train-file "$DATA_DIR/train.jsonl" \
  --eval-file "$DATA_DIR/eval.jsonl" \
  --output-dir "$OUT_DIR" \
  "${DEVICE_MAP_ARGS[@]}" \
  "${PACKING_ARGS[@]}" \
  --max-seq-length 4096 \
  --max-steps "$MAX_STEPS" \
  --eval-steps "$EVAL_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8
