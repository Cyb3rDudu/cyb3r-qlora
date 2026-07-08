#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${ACCELERATE_BIN:-}" && -x "$PROJECT_ROOT/.venv/bin/accelerate" ]]; then
  export PATH="$PROJECT_ROOT/.venv/bin:$PATH"
  ACCELERATE_BIN="$PROJECT_ROOT/.venv/bin/accelerate"
fi
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"

if command -v gcc >/dev/null 2>&1; then
  LIBSTDCXX="$(gcc -print-file-name=libstdc++.so.6)"
  if [[ -f "$LIBSTDCXX" ]]; then
    export LD_LIBRARY_PATH="$(dirname "$LIBSTDCXX"):/run/current-system/sw/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-27B-Instruct}"
DATA_DIR="${DATA_DIR:-/home/dudu/datasets/cyb3r-dataset}"
OUT_DIR="${OUT_DIR:-outputs/cyb3r-reasoning-test}"
RESUME="${RESUME:-0}"

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

"$ACCELERATE_BIN" launch --num_processes 2 scripts/train_unsloth.py \
  --model-name "$MODEL_NAME" \
  --train-file "$DATA_DIR/train.jsonl" \
  --eval-file "$DATA_DIR/eval.jsonl" \
  --output-dir "$OUT_DIR" \
  --max-seq-length 4096 \
  --max-steps 500 \
  --learning-rate 1e-4 \
  --lora-r 64 \
  --lora-alpha 128 \
  --per-device-train-batch-size 1 \
  --gradient-accumulation-steps 8 \
  "${RESUME_ARGS[@]}"
