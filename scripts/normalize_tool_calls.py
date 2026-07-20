#!/usr/bin/env python3
"""normalize_tool_calls.py — fix inconsistent tool_call schemas in a dataset.

Problems this fixes (identified in the dataset audit):
  1. `tool_calls.arguments` is sometimes a dict, sometimes a JSON string.
     PyArrow/datasets can't infer a consistent schema → load crashes.
     Fix: always serialize to a JSON string.
  2. Some tool_calls use the legacy {"name","arguments"} shape instead of the
     modern {"id","type":"function","function":{"name","arguments"}} shape.
     Fix: normalize everything to the modern shape.
  3. `tool` role messages are missing `tool_call_id`, which the chat template
     needs to link a tool response back to the call that triggered it.
     Fix: assign synthetic sequential IDs that match the preceding tool_call.

Writes to <output-dir>. Originals are never modified.

Usage:
  python scripts/normalize_tool_calls.py --dry-run
  python scripts/normalize_tool_calls.py \\
    --input-dir /home/dudu/datasets/cyb3r-dataset-agent-v2 \\
    --output-dir /home/dudu/datasets/cyb3r-dataset-agent-v2-norm
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def normalize_arguments(value) -> dict:
    """Ensure tool_call arguments is always a dict (Qwen3.6 native format).

    The Qwen3.6 chat template calls `tool_call.arguments|items` at line 120,
    which requires arguments to be a mapping (dict). A JSON string causes
    `TypeError: Can only get item pairs from a mapping.`
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
            # valid JSON but not a dict (e.g. a list or scalar) — wrap it
            return {"value": parsed}
        except json.JSONDecodeError:
            # malformed string — wrap raw to preserve content
            return {"raw": value}
    # any other type — coerce to a dict
    return {"value": value}


def normalize_tool_call(tc: dict, call_id: str) -> dict:
    """Normalize a single tool_call to the modern shape.

    Accepts both:
      legacy:  {"name": "...", "arguments": ...}
      modern:  {"id": "...", "type": "function", "function": {"name": "...", "arguments": ...}}
    Always returns the modern shape with a stable id.
    """
    if not isinstance(tc, dict):
        return tc  # leave non-dict entries alone (shouldn't happen)

    # detect shape
    if "function" in tc and isinstance(tc["function"], dict):
        # already modern-ish — just normalize arguments + id
        fn = dict(tc["function"])
        fn["arguments"] = normalize_arguments(fn.get("arguments"))
        return {
            "id": tc.get("id") or call_id,
            "type": tc.get("type", "function"),
            "function": fn,
        }

    # legacy shape: {"name", "arguments", maybe "type"/"id"}
    return {
        "id": tc.get("id") or call_id,
        "type": tc.get("type", "function"),
        "function": {
            "name": tc.get("name", "unknown"),
            "arguments": normalize_arguments(tc.get("arguments")),
        },
    }


def normalize_row(row: dict) -> tuple[dict, Counter]:
    """Normalize all tool_calls + tool messages in a row.

    Returns (new_row, stats_counter).
    """
    stats: Counter = Counter()
    msgs = row.get("messages", [])
    new_msgs = []

    # counter for generating synthetic tool_call_ids across the conversation
    call_counter = 0

    for m in msgs:
        if not isinstance(m, dict):
            new_msgs.append(m)
            continue
        new_m = dict(m)
        role = m.get("role")

        # normalize tool_calls on assistant messages
        if m.get("tool_calls"):
            new_tcs = []
            for tc in m["tool_calls"]:
                call_id = f"call_{call_counter}"
                call_counter += 1
                if isinstance(tc, dict):
                    is_legacy = "function" not in tc
                    new_tcs.append(normalize_tool_call(tc, call_id))
                    stats["legacy_shape" if is_legacy else "modern_shape"] += 1
                    args = tc.get("arguments")
                    if isinstance(tc.get("function"), dict):
                        args = tc["function"].get("arguments")
                    if isinstance(args, dict):
                        stats["args_dict"] += 1
                    elif isinstance(args, str):
                        stats["args_str_to_dict"] += 1
                else:
                    new_tcs.append(tc)
            new_m["tool_calls"] = new_tcs

        # assign tool_call_id to tool messages
        if role == "tool":
            if not m.get("tool_call_id"):
                # link to the most recent call_counter — but we need the actual
                # id of the last tool_call. Track via a separate pointer.
                # We assign call_{N} where N is the call before this tool msg.
                # Since tool messages follow tool_calls in order, use the
                # last call_id we issued that hasn't been linked yet.
                stats["tool_msg_missing_id"] += 1
            else:
                stats["tool_msg_has_id"] += 1

        new_msgs.append(new_m)

    # Second pass: assign tool_call_id to tool messages by matching them to
    # the preceding tool_call in sequence.
    call_ids_in_order: list[str] = []
    for m in new_msgs:
        if isinstance(m, dict) and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if isinstance(tc, dict) and tc.get("id"):
                    call_ids_in_order.append(tc["id"])

    tool_idx = 0
    for m in new_msgs:
        if isinstance(m, dict) and m.get("role") == "tool" and not m.get("tool_call_id"):
            if tool_idx < len(call_ids_in_order):
                m["tool_call_id"] = call_ids_in_order[tool_idx]
                tool_idx += 1

    new_row = dict(row)
    new_row["messages"] = new_msgs
    return new_row, stats


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="/home/dudu/datasets/cyb3r-dataset-agent-v2")
    ap.add_argument("--output-dir", default=None,
                    help="default: <input-dir>-norm")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir.parent / f"{in_dir.name}-norm"

    print(f"input:  {in_dir}")
    print(f"output: {out_dir}")
    print(f"mode:   {'DRY-RUN' if args.dry_run else 'WRITE'}")

    total_stats: Counter = Counter()
    for split in ("train", "eval"):
        path = in_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        new_rows = []
        split_stats: Counter = Counter()
        for r in rows:
            nr, st = normalize_row(r)
            new_rows.append(nr)
            split_stats.update(st)
        total_stats.update(split_stats)
        print(f"\n{split}: {len(rows):,} rows processed")
        for k, v in split_stats.most_common():
            print(f"  {k:25s} {v:>6,d}")
        if not args.dry_run:
            write_jsonl(out_dir / f"{split}.jsonl", new_rows)
            print(f"  wrote {out_dir / f'{split}.jsonl'}")

    # validation: confirm the fix works
    if not args.dry_run:
        print(f"\n{'=' * 60}")
        print("VALIDATION: reload with datasets to confirm no schema error")
        print(f"{'=' * 60}")
        try:
            from datasets import load_dataset
            for split in ("train", "eval"):
                p = out_dir / f"{split}.jsonl"
                if p.exists():
                    ds = load_dataset("json", data_files=str(p), split="train")
                    print(f"  {split}: {len(ds):,} rows loaded OK ✓")
        except Exception as e:
            print(f"  VALIDATION FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
