#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


SAFE_METADATA_KEYS = {
    "pillar",
    "category",
    "difficulty",
    "split",
    "task_type",
    "topic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--agent-name-from", default="old_agent_name")
    parser.add_argument("--agent-name-to", default="cyb3r")
    return parser.parse_args()


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def rewrite_agent_name(text: str, old: str, new: str) -> str:
    return text.replace(old, new).replace(old.capitalize(), new.capitalize())


def sanitize_row(row: dict, trace_type: str, old_name: str, new_name: str) -> dict:
    cloned = copy.deepcopy(row)

    for message in cloned.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = rewrite_agent_name(content, old_name, new_name)

    metadata = cloned.get("metadata") or {}
    safe_metadata = {key: value for key, value in metadata.items() if key in SAFE_METADATA_KEYS}
    safe_metadata["trace_type"] = trace_type
    cloned["metadata"] = safe_metadata
    return cloned


def split_key(row: dict) -> str:
    msgs = row.get("messages", [])
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    assistant = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    trace_type = (row.get("metadata", {}) or {}).get("trace_type", "unknown")
    basis = f"{trace_type}\n{user[:400]}\n{assistant[:400]}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def collect_rows(entries: list[dict], old_name: str, new_name: str) -> tuple[list[dict], Counter]:
    selected: list[dict] = []
    counts: Counter = Counter()

    for entry in entries:
        trace_type = entry["trace_type"]
        quota = int(entry["quota"])
        paths = [Path(path) for path in entry["files"]]
        categories = set(entry.get("categories", []))

        for path in paths:
            if not path.exists():
                continue
            for row in load_jsonl(path):
                if counts[trace_type] >= quota:
                    break
                metadata = row.get("metadata") or {}
                # Match against either the legacy "category" field (agent_policy)
                # or the "ctf_class" tag added by filter_ctf_usable.py.
                category = metadata.get("category") or metadata.get("ctf_class")
                pillar = metadata.get("pillar")
                if categories and category not in categories and pillar != "agent_policy":
                    continue
                selected.append(sanitize_row(row, trace_type, old_name, new_name))
                counts[trace_type] += 1
            if counts[trace_type] >= quota:
                break

    return selected, counts


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    manifest_path = Path(args.selection_manifest)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    selection = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Support both the legacy manifest keys (reasoning_sets, policy_sets) and
    # the new agent/CTF manifest keys (agent_policy_sets, ctf_sets,
    # pentest_trajectory_sets). All are treated uniformly by collect_rows.
    all_entries = (
        selection.get("reasoning_sets", [])
        + selection.get("policy_sets", [])
        + selection.get("agent_policy_sets", [])
        + selection.get("ctf_sets", [])
        + selection.get("pentest_trajectory_sets", [])
    )

    selected, counts = collect_rows(
        all_entries,
        args.agent_name_from,
        args.agent_name_to,
    )

    random.shuffle(selected)

    train_rows = []
    eval_rows = []
    threshold = int(args.eval_ratio * 10000)
    for row in selected:
        digest = int(split_key(row), 16) % 10000
        if digest < threshold:
            eval_rows.append(row)
        else:
            train_rows.append(row)

    def write_jsonl(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "eval.jsonl", eval_rows)

    output_manifest = {
        "name": selection.get("name", "cyb3r_reasoning_test"),
        "description": selection.get("description", ""),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "trace_mix": {entry["trace_type"]: int(entry["quota"]) for entry in all_entries},
        "counts": dict(counts),
        "selection_manifest": str(manifest_path),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(output_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
