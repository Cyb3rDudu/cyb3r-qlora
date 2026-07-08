#!/usr/bin/env python3
"""
Build a compact subset from a merged chat corpus using abstract trace labels.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to merged corpus JSONL")
    parser.add_argument("--output", required=True, help="Path to write selected JSONL")
    parser.add_argument("--reasoning-target", type=int, default=25_000)
    parser.add_argument("--agent-policy-target", type=int, default=5_000)
    return parser.parse_args()


def get_trace_type(row: dict) -> str:
    metadata = row.get("metadata", {}) or {}
    return metadata.get("trace_type", "")


def get_pillar(row: dict) -> str:
    metadata = row.get("metadata", {}) or {}
    return metadata.get("pillar", "")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    counts = Counter()
    selected = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trace_type = get_trace_type(row)
            pillar = get_pillar(row)

            if pillar == "reasoning" and trace_type and counts["reasoning"] < args.reasoning_target:
                selected.append(row)
                counts["reasoning"] += 1
            elif pillar == "agent_policy" and counts["agent_policy"] < args.agent_policy_target:
                selected.append(row)
                counts["agent_policy"] += 1

            if (
                counts["reasoning"] >= args.reasoning_target
                and counts["agent_policy"] >= args.agent_policy_target
            ):
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"selected": len(selected), "counts": counts}, indent=2, default=dict))


if __name__ == "__main__":
    main()
