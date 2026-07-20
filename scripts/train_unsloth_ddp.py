#!/usr/bin/env python3
"""DDP (Distributed Data Parallel) training script for Unsloth QLoRA.

Each torchrun process loads a FULL 4-bit model copy on its assigned GPU
(LOCAL_RANK). Both GPUs process different data simultaneously, so throughput
~= 2x single-card (vs model-parallel which serializes activations and gets
1-card throughput).

Differences from train_unsloth.py:
  - NO device_map / balanced-split: each process pins the whole model to its
    own GPU via CUDA_VISIBLE_DEVICES (set by torchrun per-rank).
  - SFTTrainer + accelerate handle the DDP wrapper automatically when
    torch.distributed is initialized (torchrun does this).
"""
from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--eval-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=3565)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--packing", action="store_true",
                        help="Pack short sequences into 4096-token blocks. ~1.5x faster.")
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import unsloth  # noqa: F401
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_main = local_rank == 0
    if is_main:
        print(f">> DDP: world_size={world_size}, local_rank={local_rank}", flush=True)

    # No device_map: each process gets one GPU (its LOCAL_RANK), pinned by
    # torchrun via CUDA_VISIBLE_DEVICES. Unsloth loads the whole model there.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    def render_messages(batch):
        texts = []
        for messages in batch["messages"]:
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            texts.append("".join(parts))
        return {"text": texts}

    train_ds = load_dataset("json", data_files=args.train_file, split="train")
    eval_ds = load_dataset("json", data_files=args.eval_file, split="train")
    train_ds = train_ds.map(render_messages, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(render_messages, batched=True, remove_columns=eval_ds.column_names)

    # SFTConfig (not TrainingArguments) -- avoids the checkpoint pickling crash
    # (see commit 3b53eed). save_only_model skips the buggy trainer-state pickle.
    training_args = SFTConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        save_only_model=True,
        logging_steps=10,
        max_steps=args.max_steps,
        bf16=True,
        fp16=False,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        optim="adamw_8bit",
        report_to="none",
        dataset_text_field="text",
        max_length=args.max_seq_length,
        packing=args.packing,
        logging_nan_inf_filter=True,
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if is_main:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(">> training complete, model saved", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
