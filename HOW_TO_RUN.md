# How To Run On carrier

These instructions assume the repo lives at:

- `/home/dudu/Code/cyb3r-qlora`

and the source corpus lives at:

- `/home/dudu/datasets/cyb3r-dataset`

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

The manifest should define abstract trace buckets and local file paths. Example shape:

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

Current expected counts from the present source corpus:

- `28,567` train rows
- `1,433` eval rows

## 4. Enter a Python 3.11 shell

Use Nix to get a compatible Python:

```bash
nix-shell -p python311 python311Packages.pip python311Packages.virtualenv git
```

Inside that shell:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth transformers datasets trl peft accelerate bitsandbytes sentencepiece protobuf
```

## 5. Configure Accelerate

Run once:

```bash
accelerate config
```

Recommended answers:

- compute environment: local machine
- distributed type: multi-GPU
- number of processes: `2`
- mixed precision: `bf16` if stable, otherwise `fp16`

## 6. Start training

Example:

```bash
accelerate launch --num_processes 2 scripts/train_unsloth.py \
  --model-name Qwen/Qwen3-27B-Instruct \
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

## 7. What to evaluate

Check whether the tuned model is better at:

- global reassessment
- branch ranking
- stopping low-yield rabbit holes
- preserving tool-call competence

Use:

- `scripts/eval_prompts.md`

for manual before/after comparisons.

## Notes

- Use the original model checkpoint for QLoRA training. The local Q8 weights are kept for inference because that is the practical fit for this hardware at runtime, but they are not the right artifact for adapter training.
- Keep the adapter separate first; merge later only if needed.
- The local dataset under `/home/dudu/datasets/cyb3r-dataset` is outside git and stays untracked.
