#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# --- pick the venv python/accelerate --------------------------------------
if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PY_BIN="$PROJECT_ROOT/.venv/bin/python"
  export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
else
  PY_BIN="$(command -v python3)"
fi

# --- NixOS runtime library paths ------------------------------------------
# pip-installed wheels (torch, triton, bitsandbytes) are generic-Linux ELF
# binaries. NixOS does not put libz/libstdc++/libcuda on the dynamic linker
# search path by default, so they fail to import unless we add the nix store
# dirs and the NVIDIA driver libs explicitly.
if command -v gcc >/dev/null 2>&1; then
  LIBSTDCXX="$(gcc -print-file-name=libstdc++.so.6)"
  if [[ -f "$LIBSTDCXX" ]]; then
    LIBSTDCXX_DIR="$(dirname "$LIBSTDCXX")"
  fi
fi
LIBSTDCXX_DIR="${LIBSTDCXX_DIR:-/run/current-system/sw/lib}"
SW_LIB="/run/current-system/sw/lib"          # libz.so, ldconfig
OGL_LIB="/run/opengl-driver/lib"             # NVIDIA driver libcuda.so.1
export LD_LIBRARY_PATH="${LIBSTDCXX_DIR}:${SW_LIB}:${OGL_LIB}:${LD_LIBRARY_PATH:-}"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B}"
DATA_DIR="${DATA_DIR:-/home/dudu/datasets/cyb3r-dataset}"
OUT_DIR="${OUT_DIR:-outputs/cyb3r-reasoning-test}"
# Balanced per-layer split across both GPUs: equal compute + thermals on each
# card, instead of device_map='auto' which pins one card at 100% and leaves
# the other idle. Override with DEVICE_MAP=auto / cuda:0 / etc. if needed.
BALANCED_SPLIT="${BALANCED_SPLIT:-1}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
RESUME="${RESUME:-0}"

if [[ "$MODEL_NAME" != "Qwen/Qwen3.6-27B" ]]; then
  echo "this training repo is pinned to Qwen/Qwen3.6-27B, got: $MODEL_NAME" >&2
  exit 1
fi

RESUME_ARGS=()
if [[ "$RESUME" == "1" ]]; then
  if [[ ! -d "$OUT_DIR" ]]; then
    echo "resume requested but output directory does not exist: $OUT_DIR" >&2
    exit 1
  fi

  LATEST_CHECKPOINT="$(find "$OUT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
  if [[ -z "${LATEST_CHECKPOINT:-}" ]]; then
    echo "resume requested but no checkpoint found under: $OUT_DIR" >&2
    exit 1
  fi

  echo "resuming from checkpoint: $LATEST_CHECKPOINT"
  RESUME_ARGS=(--resume-from-checkpoint "$LATEST_CHECKPOINT")
fi

echo ">> launching single-process model-parallel training"
echo ">> model:      $MODEL_NAME"
echo ">> devices:    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  device_map=$DEVICE_MAP  balanced_split=$BALANCED_SPLIT"
echo ">> data:       $DATA_DIR"
echo ">> output:     $OUT_DIR"

DEVICE_MAP_ARGS=()
if [[ "$BALANCED_SPLIT" == "1" ]]; then
  DEVICE_MAP_ARGS=(--balanced-split)
else
  DEVICE_MAP_ARGS=(--device-map "$DEVICE_MAP")
fi

exec "$PY_BIN" scripts/train_unsloth.py \
  --model-name "$MODEL_NAME" \
  --train-file "$DATA_DIR/train.jsonl" \
  --eval-file "$DATA_DIR/eval.jsonl" \
  --output-dir "$OUT_DIR" \
  "${DEVICE_MAP_ARGS[@]}" \
  --max-seq-length 4096 \
  --max-steps 500 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  "${RESUME_ARGS[@]}"
