#!/usr/bin/env bash
# Build the training venv on NixOS (carrier).
#
# Stack: torch 2.7.1 + cu128, triton 3.3.1, unsloth, fla (flash-linear-attention),
#        causal-conv1d built against the system CUDA 12.8 toolkit.
#
# Requires (provided by the carrier-nixos flake):
#   - python3.13 on PATH
#   - uv on PATH
#   - cudaPackages_12_8.cudatoolkit installed (provides the CUDA_HOME that
#     torch's cpp_extension needs to compile causal-conv1d / fla kernels without
#     a "detected CUDA version (12.9) mismatches torch (12.8)" error)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. On carrier, rebuild the NixOS config or run: nix shell nixpkgs#uv" >&2
  exit 1
fi

# --- locate the CUDA 12.8 toolkit installed by carrier-nixos ----------------
# cudaPackages_12_8.cudatoolkit produces a "cuda-merged-12.8" store path with a
# proper bin/nvcc + include/ + lib/ layout. Find it; fall back to scanning the
# store if the well-known name isn't resolved yet.
CUDA_12_8_HOME="${CUDA_12_8_HOME:-}"
if [[ -z "$CUDA_12_8_HOME" ]]; then
  CUDA_12_8_HOME="$(ls -d /nix/store/*-cuda-merged-12.8 2>/dev/null | head -1 || true)"
fi
if [[ -z "$CUDA_12_8_HOME" ]] || [[ ! -x "$CUDA_12_8_HOME/bin/nvcc" ]]; then
  echo "ERROR: could not find CUDA 12.8 toolkit (cuda-merged-12.8) in /nix/store." >&2
  echo "       Run: sudo nixos-rebuild switch --flake ~/Code/carrier-nixos#carrier" >&2
  echo "       (it provides cudaPackages_12_8.cudatoolkit via packages.nix)" >&2
  exit 1
fi
echo "Using CUDA 12.8 toolkit at: $CUDA_12_8_HOME"
export CUDA_HOME="$CUDA_12_8_HOME"

# libstdc++ for the build tools / runtime linker
LIBSTDCXX_DIR="$(dirname "$(gcc -print-file-name=libstdc++.so.6)")"
export LIBRARY_PATH="${CUDA_HOME}/lib:${CUDA_HOME}/lib/stubs:${LIBSTDCXX_DIR}:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${LIBSTDCXX_DIR}:/run/current-system/sw/lib:/run/opengl-driver/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="8.6"

# --- build the venv ---------------------------------------------------------
uv venv --python 3.13 .venv

# torch 2.7.1 cu128 wheels (pinned; the +cu128 build matches CUDA 12.8 and
# bundles triton 3.3.1 which has the fla-compatible PTX/version handling).
uv pip install --python .venv/bin/python \
  torch==2.7.1+cu128 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128

# unsloth + the rest (keeps torch 2.7.1; --no-build-isolation not needed here
# because these are all wheels, no CUDA compile).
uv pip install --python .venv/bin/python \
  unsloth \
  transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf

# fla fast path: flash-linear-attention (pure-python + triton kernels, no CUDA
# compile) and causal-conv1d (CUDA ext, compiled against CUDA 12.8).
uv pip install --python .venv/bin/python flash-linear-attention
uv pip install --python .venv/bin/python causal-conv1d --no-build-isolation

# --- NixOS patches ----------------------------------------------------------
# Triton's bundled ptxas cannot run on NixOS (generic-Linux ELF), and on NixOS
# the python3 profile path has no Python.h so triton's build.py cannot compile
# its driver helper, and libcuda discovery hardcodes /sbin/ldconfig. Patch all
# three so the installed wheel survives a venv rebuild. Re-runnable / idempotent.
.venv/bin/python scripts/patch_triton_nixos.py

cat <<MSG

Training venv is ready (torch 2.7.1 + cu128, triton 3.3.1, fla).
Run:
  bash scripts/run_train.sh
MSG
