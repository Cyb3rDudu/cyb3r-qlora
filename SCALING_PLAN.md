# Scaling Plan: 1.3M-Row Production Run

## Context

The current run is a **500-step test** that proves the pipeline:

- Model: `Qwen/Qwen3.6-27B` (hybrid gated-deltanet + full attention, 64 layers, 27.7B params)
- Method: 4-bit QLoRA, r=64/α=128, all 7 linear target modules (~318M trainable, 1.15%)
- Hardware: 2× RTX 3090 (24 GB each, Ampere sm_86), NVLink bridge present
- Mode: **single-process model-parallel**, balanced 32+32 layer split
- Dataset: 28,522 train / 1,478 eval rows, 4096 ctx
- Measured: ~44–47 s/step, peak GPU0 9 GB / GPU1 16 GB, temps 66–80 °C

The **production target is 1.3M rows**. At the current ~46 s/step that is not
a "make it slightly faster" problem — it is a "this will not finish in a
reasonable time" problem. This document lists the realistic levers, ranked by
expected impact and effort, with the research that backs each.

---

## Why the current run is slow (the real bottleneck)

Two compounding causes, in order of impact:

1. **Linear-attention fast path is disabled.** Qwen3.6 uses gated-deltanet
   layers (48 of the 64 layers are linear attention). Unsloth ships the
   flash-linear-attention (fla) kernels for them, but they require
   `torch >= 2.7` and `triton >= 3.3`. We are on torch 2.5.1 + triton 3.1.0
   (the `unsloth[cu121-torch251]` stack), so Unsloth prints:

   > "they could not be enabled on this setup ... transformers will use a
   > slower pure PyTorch path."

   That pure-PyTorch fallback is the dominant cost per step.

2. **Pipeline-parallel is sequential.** Model-parallel (the 32+32 split) makes
   only one GPU compute at a time within a forward pass. DDP would use both
   GPUs in parallel (~2× throughput) but OOMs on this model at 4096 ctx
   (~22.6 GB peak per card vs 24 GB available).

NVLink is **not** a meaningful lever here. Reddit u/DistanceSolar1449's
analysis (r/LocalLLaMA, JohnTheNerd3's 2×3090 thread) is correct: for a hidden
size of 5120 the DDP all-reduce is ~1.3 MB per step, which "doesn't come close
to saturating PCIe." NVLink helps giant all-reduces (70B+ dense, MoE), not us.

---

## Levers, ranked

### Lever 1 — Enable the fla fast path (torch 2.7 + triton 3.3)  ★ highest impact

**Expected:** ~1.5–2× faster steps (the linear-attention layers become fast).
This is the single highest-leverage change.

**Cost:** This is the "environment surgery" codex flinched at. Concretely:

- Rebuild the venv against `torch 2.7.x + cu128` (or cu129) instead of
  cu121/torch251.
- `triton >= 3.3` (ships with torch 2.7 wheels).
- Install `flash-linear-attention` + `causal-conv1d` (the exact packages the
  Unsloth startup message points at:
  `https://github.com/fla-org/flash-linear-attention` and
  `https://github.com/Dao-AILab/causal-conv1d`).
- A `ptxas` that accepts PTX 8.9. The system CUDA is 12.9 (caps at 8.8). Two
  options:
  - Use a newer CUDA toolkit via nixpkgs if 12.10+ is available, or
  - Confirm torch 2.7's bundled CUDA runtime handles it (the torch wheel
    bundles its own ptxas under `torch/bin/` — point `TRITON_PTXAS_PATH` at it).
- Re-validate all four NixOS fixes still hold on the new stack
  (`patch_triton_nixos.py` must be re-run; the ldconfig/libz/libcuda paths
  are unchanged).

**Risk:** torch 2.7 on Ampere (sm_86) is supported, but Unsloth's exact
compatibility window for the fla path on 3090 needs a smoke test before
committing. Reddit (u/danielhanchen, Unsloth author) confirms the fla kernels
were tested on T4/A100 and "hopefully everything works smoothly" on 3090 but
it is not officially validated.

**Verdict:** Do this. It is the only lever that attacks cause #1 directly.

### Lever 2 — Sequence packing  ★ high impact, low risk

**Expected:** 2–3× more effective throughput **per row processed**, because
packing concatenates many short rows into one 4096-token sequence so no GPU
time is wasted on padding. Unsloth's own release (r/LocalLLaMA, "train LLMs
3x faster") attributes a large part of its 3× speedup to "smart auto packing."

**Cost:** Flip `packing=True` in `SFTTrainer` (currently `packing=False` in
`train_unsloth.py`). Unsloth's packing handles the attention-mask correctly
so rows do not attend across boundaries.

**Risk:** Your rows are long (p50 ≈ 3k tokens, p90 ≈ 10.6k, max 188k). Packing
helps the short rows most; the long rows already fill a sequence. Net win is
real but smaller than for a chat dataset. Needs a throughput measurement.

**Verdict:** Do this regardless. Free upside, reversible.

### Lever 3 — Sequence length: 4096 → 2048 (or dynamic)  ★ medium impact

**Expected:** ~2× step speed and ~2× memory headroom (attention is
quadratic-ish; the linear-attention layers are linear, so the win is
concentrated in the 16 full-attention layers).

**Trade-off:** This is the **reasoning-quality** question you asked about.
4096 exists to capture the long reasoning traces whole. At 2048 you truncate:
  - p50 (3k tok) → truncated ~33%
  - p90 (10.6k tok) → truncated ~80%
  - This destroys the long-tail exploitability/CVE-triage traces, which are
    the highest-value part of the dataset.

**Better alternative — dynamic length / length-grouped sampling:** keep
`max_seq_length=4096` as the cap but enable `group_by_length=True` so the
batcher groups similar-length rows, minimizing padding without truncating.
This recovers most of the packing win without losing long traces.

**Verdict:** Prefer `group_by_length=True` over cutting the cap. Only cut to
2048 if a quality audit shows the long traces are not contributing.

### Lever 4 — Revisit DDP once L1+L2 land  ★ conditional

**Expected:** ~2× throughput (both GPUs compute in parallel) — but **only
after** L1 (fla) and L2 (packing) reduce per-card memory.

Today DDP OOMs because each card holds a full 18 GB model + activations at
4096 ctx. Unsloth's newer kernels ("30–90% less VRAM") plus packing plus
`group_by_length` may bring peak per-card under ~22 GB, making DDP viable.
Unsloth now officially supports DDP via `torchrun --nproc_per_node=2` (see
`docs.unsloth.ai/basics/multi-gpu-training-with-unsloth/ddp`), auto-enabled
at >1 GPU.

**Verdict:** Re-measure after L1+L2. If a DDP smoke test fits, switch from
model-parallel to DDP for ~2× on top of the L1/L2 gains. Compounding: L1
(~1.7×) × L2 (~1.5×) × L4 (~2×) ≈ **~5× total** — turning a multi-week run
into a few days.

### Levers considered and rejected

- **More VRAM / more GPUs:** explicitly out of scope per constraint.
- **NVLink tuning:** irrelevant at hidden size 5120 (see analysis above).
- **Smaller model (14B dense):** would unlock easy DDP, but changes the
  project's whole premise (a 27B-class security model). Not a speedup, a
  scope change.
- **8-bit vs 4-bit base:** already on 4-bit QLoRA + `adamw_8bit`; nothing
  left to compress on the weights side without quality loss.

---

## Recommended execution order (next iteration)

1. **Branch the env.** Keep the working torch-2.5.1 venv intact (rename
   `.venv` → `.venv-torch251`). Build a second venv `.venv-torch27` so the
   production test run can always fall back.

2. **L1: build the torch 2.7 + fla stack** in the new venv.
   - `uv pip install torch --index-url .../cu128` (2.7.x)
   - install unsloth from source or the latest wheel that supports torch 2.7
   - `pip install flash-linear-attention causal-conv1d`
   - re-run `patch_triton_nixos.py`; re-point `TRITON_PTXAS_PATH` at the
     torch-bundled ptxas if the system one still caps at 8.8.
   - **Smoke test:** load Qwen3.6-27B, confirm the startup message no longer
     says "slower pure PyTorch path" — i.e. fla is enabled.

3. **L2: enable packing.** Set `packing=True` in the trainer. Measure s/step
   and tokens/step; confirm loss curve still looks healthy on a 50-step run.

4. **L3: enable `group_by_length=True`.** Cheaper than cutting context,
   recovers padding waste.

5. **L4: attempt DDP.** `torchrun --nproc_per_node=2 scripts/train_unsloth.py`
   with the same args. Watch peak VRAM per card. If it fits under ~22 GB,
   keep it; else stay on model-parallel.

6. **Only then** kick off the 1.3M-row run, with:
   - a fixed logging bug (see below),
   - checkpointing every 500 steps (not 100 — 1.3M rows is a long run),
   - a held-out eval slice from the 1.3M, not the old 1,478-row eval.

---

## Known issues to fix before the big run

- **Loss not logging to the nohup log file.** At step 50 with
  `logging_steps=10`, no `{'loss': ...}` line appeared in the captured log
  (tqdm under `nohup` interleaves carriage returns). Training is working
  (smoke test showed 4.48 → 1.41), but for a multi-day run you want
  machine-parseable metrics. Fix: set `report_to="tensorboard"` (or wandb)
  and `disable_tqdm=True` in `TrainingArguments`, so loss lands in an
  event file instead of a mangled progress bar.

- **Eval cadence on a huge dataset.** `eval_steps=50` on 1.3M rows will fire
  constantly and each eval pass is expensive. Raise to `eval_steps=500` and
  cap eval rows to a fixed sample (e.g. 2,000).

- **Checkpoint disk.** Each checkpoint for a 27B + 318M-LoRA is non-trivial.
  At 500-step intervals over a 1.3M-row run that is many checkpoints — set
  `save_total_limit=3` to keep only the latest three.

---

## Sources

- r/LocalLLaMA u/danielhanchen (Unsloth author), "Train MoE models 12x
  faster" — confirms fla/Triton-kernel speedups and "freeze the router" MoE
  tip. https://www.reddit.com/r/LocalLLaMA/comments/1r14h9u/
- r/LocalLLaMA u/danielhanchen, "Train LLMs 3x faster with 30% less memory"
  — smart auto-packing is a major part of the 3×. DDP guide link in
  comments. https://www.reddit.com/r/LocalLLaMA/comments/1pj51tu/
- r/LocalLLaMA u/JohnTheNerd3, "Qwen3.5 27b dense on 2x3090" — real 2×3090
  numbers; u/DistanceSolar1449 comment on why NVLink does not matter at this
  hidden size. https://www.reddit.com/r/LocalLLaMA/comments/1rianwb/
- Unsloth DDP docs (torchrun, auto-enabled >1 GPU, QLoRA-supported):
  https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp
- Unsloth startup message in our own log pointing at fla + causal-conv1d
  install URLs (the authoritative source for which packages L1 needs).
