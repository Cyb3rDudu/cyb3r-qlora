#!/usr/bin/env python3
from __future__ import annotations

import argparse


def _build_balanced_device_map(model_name: str = "Qwen/Qwen3.6-27B"):
    """Split the language layers evenly across all visible CUDA GPUs.

    device_map='auto' packs the first GPU to its memory cap and overflows the
    rest, which leaves one card idle and the other pinned at 100%/hot. An
    explicit per-layer split balances compute (and therefore thermals) evenly.
    The big non-layer modules (embed_tokens, lm_head) go on opposite cards.
    """
    import torch
    from transformers import AutoConfig

    n_gpus = torch.cuda.device_count()
    if n_gpus < 2:
        return "auto"

    # Read layer count from the model config without loading weights.
    # Qwen3.5-family wraps the LLM under text_config.
    cfg = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = (
        getattr(text_cfg, "num_hidden_layers", None)
        or getattr(text_cfg, "num_layers", None)
        or 64
    )

    # layer i -> gpu (i * n_gpus) // n_layers, i.e. contiguous blocks per GPU
    blocks = {g: [] for g in range(n_gpus)}
    for i in range(n_layers):
        g = (i * n_gpus) // n_layers
        blocks[g].append(i)

    # language model prefix differs across architectures; cover the common ones
    layer_prefixes = (
        "language_model.layers",
        "model.language_model.layers",
        "model.layers",
        "layers",
    )
    device_map = {}
    for g, idxs in blocks.items():
        for i in idxs:
            for prefix in layer_prefixes:
                device_map[f"{prefix}.{i}"] = g

    # Non-layer heavy modules: spread across GPUs. embed on GPU0, lm_head on
    # the last GPU; norm/rotary live with the last layer block.
    last = n_gpus - 1
    for embed in ("model.language_model.embed_tokens", "language_model.embed_tokens", "model.embed_tokens", "embed_tokens"):
        device_map[embed] = 0
    for lm in ("lm_head", "model.lm_head"):
        device_map[lm] = last
    for norm in ("model.language_model.norm", "language_model.norm", "model.norm", "norm"):
        device_map[norm] = last
    for rotary in ("model.language_model.rotary_emb", "language_model.rotary_emb", "model.rotary_emb", "rotary_emb"):
        device_map[rotary] = last
    for visual in ("model.visual", "visual"):
        device_map[visual] = 0

    print(f"[balanced-split] {n_layers} layers across {n_gpus} GPUs: "
          f"{', '.join(f'GPU{g}={len(idxs)}' for g, idxs in blocks.items())}", flush=True)
    return device_map


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
    parser.add_argument("--packing", action="store_true",
                        help="Pack short sequences into 4096-token blocks via "
                             "best-fit-decreasing. Eliminates padding waste "
                             "(~1.5x faster for variable-length datasets). "
                             "No accuracy degradation per Unsloth benchmarks.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--device-map",
        default="auto",
        help="transformers device_map. Use 'auto' to shard the model across all "
        "visible GPUs (model-parallel), which is what 2x24GB cards need for a 27B "
        "4-bit model. Pass 'cuda:0' for a single GPU.",
    )
    parser.add_argument(
        "--balanced-split",
        action="store_true",
        help="Build an explicit device_map that splits the language layers evenly "
        "across all visible GPUs (32+32 for a 64-layer model). This balances "
        "both memory AND compute/thermals, unlike device_map='auto' which packs "
        "the first GPU and overflows to the rest (leaving one card idle/hot). "
        "Overrides --device-map when set.",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--load-adapter",
        default=None,
        help="Path to a saved LoRA adapter directory (e.g. from a previous epoch). "
        "If set, the base model is loaded and this adapter is applied on top, "
        "so training continues from the adapter's weights instead of fresh LoRA. "
        "Used for multi-epoch training where each epoch is a fresh run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import unsloth  # noqa: F401
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from trl import SFTConfig, SFTTrainer

    device_map = _build_balanced_device_map(args.model_name) if args.balanced_split else args.device_map

    # For multi-epoch continuation, Unsloth supports pointing model_name directly
    # at a directory containing adapter_config.json — it loads the base model and
    # applies the saved adapter automatically. We use a separate --load-adapter
    # flag to make this explicit and keep the base model_name stable.
    load_name = args.load_adapter if args.load_adapter else args.model_name
    if args.load_adapter:
        print(f"[epoch-continue] loading base + adapter from {args.load_adapter}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=load_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        device_map=device_map,
    )

    # When loading an existing adapter, get_peft_model would create a NEW fresh
    # adapter and discard the loaded weights. Skip it — the model is already a
    # trainable PEFT model from from_pretrained.
    if not args.load_adapter:
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        print("[epoch-continue] adapter loaded, skipping fresh get_peft_model", flush=True)

    # Load JSONL directly instead of load_dataset("json"). The datasets library
    # uses PyArrow, which infers a single schema across all rows — and breaks
    # when tool_calls.arguments dicts have heterogeneous keys across rows
    # (e.g. {command: ...} vs {url: ...} vs {query: ...}).
    #
    # We render the chat template to text HERE (before building the Dataset),
    # then store only the rendered text. This sidesteps PyArrow schema
    # inference on the nested messages/tool_calls structures entirely, since
    # the Dataset only ever sees a flat {"text": str} schema.
    import json as _json
    from datasets import Dataset

    def _load_and_render(path: str) -> "Dataset":
        texts = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = _json.loads(line)
                messages = row.get("messages", [])
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                texts.append(text)
        return Dataset.from_dict({"text": texts})

    train_ds = _load_and_render(args.train_file)
    eval_ds = _load_and_render(args.eval_file)

    # train_ds/eval_ds already have the rendered text column from _load_and_render;
    # the SFTTrainer picks it up via dataset_text_field="text" in SFTConfig.

    # Use SFTConfig (not transformers.TrainingArguments). SFTTrainer expects an
    # SFTConfig; passing TrainingArguments mostly works but crashes at
    # checkpoint time with "Can't pickle SFTConfig: it's not the same object"
    # because the Unsloth-compiled trainer's SFTConfig class identity differs
    # from the one torch.save re-imports.
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
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
