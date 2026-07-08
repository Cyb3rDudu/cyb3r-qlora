#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-27B-Instruct}"
DATA_DIR="${DATA_DIR:-/home/dudu/Documents/cyb3r-dataset}"
OUT_DIR="${OUT_DIR:-outputs/cyb3r-reasoning-test}"

accelerate launch --num_processes 2 scripts/train_unsloth.py \
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
  --gradient-accumulation-steps 8
