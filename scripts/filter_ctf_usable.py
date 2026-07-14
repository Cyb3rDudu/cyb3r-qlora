#!/usr/bin/env python3
"""filter_ctf_usable.py — classify CTF-pillar rows as usable vs junk.

Reads:  01-converted/justinwangx__CTFtime.jsonl
        01-converted/l3afai__dataset.jsonl
Writes: 01-converted/ctf_usable.jsonl         (kept rows, tagged with class)
        reports/ctf_junk_audit.json           (full statistics)

Classification tiers:
  KEEP:
    engagement_decision   — has Decision/STATE/pivot/next-action reasoning
    walkthrough_commands  — step-by-step with terminal/tool commands
    solution_steps        — structured solving narrative (Step 1, First, Then, ...)
    tool_trajectory       — multi-turn with tool_calls or tool role messages
  DROP:
    low_signal_qa         — generic "Analyze/Explain" wrapper + raw dump, no structure
    raw_writeup_dump      — unstructured prose dump with no actionable pattern
    too_short             — assistant < 200 chars (no real content)
    too_long              — > 16k chars (truncation risk at 4096 ctx)
    non_english           — majority non-ASCII (Chinese/mixed writeups)
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

CORPUS_ROOT = Path(os.path.expanduser("~/datasets/catdev-security"))
CONVERTED = CORPUS_ROOT / "01-converted"
REPORTS = CORPUS_ROOT / "reports"
OUT_FILE = CONVERTED / "ctf_usable.jsonl"
AUDIT_FILE = REPORTS / "ctf_junk_audit.json"

SOURCES = [
    "justinwangx__CTFtime.jsonl",
    "l3afai__dataset.jsonl",
]

# --- classifiers -------------------------------------------------------------

# Engagement-style: contains explicit decision/pivot language in a
# directive context (not just incidental mentions). Require at least 2 hits
# OR a strong single marker (STATE:, Decision:, ABORT, PIVOT) to qualify.
RE_DECISION_STRONG = re.compile(
    r"(STATE\s*:|Decision:|^Decision\b|ABORT\b|PIVOT\b|"
    r"stop.threshold|yield.rate|branch.ranking|re-plan|replan)",
    re.I | re.M,
)
RE_DECISION_WEAK = re.compile(
    r"\b(pivot|next.action|next.step|re-assess|reassess|"
    r"budget|threshold|defer|hypothes)\b",
    re.I,
)

# Walkthrough: real tool/terminal commands. Require MULTIPLE distinct
# tool invocations to qualify as a real walkthrough (filters out writeups
# that just show one incidental code block).
RE_TERMINAL = re.compile(
    r"(\$ |\.\/\w|root@|>>>|^> |nmap |gobuster |sqlmap |hydra |ffuf |"
    r"nikto |dirb |wfuzz |msfconsole|exploit\/|curl |wget |\bnc \b|netcat|"
    r"\bjohn \b|hashcat |enum4linux|smbclient|rpcclient|crackmapexec|\bcme \b|"
    r"ps1|powershell|cmd\.exe|whoami|uname|ifconfig|ip addr|"
    r"ssh \b|ftp \b|telnet|openssl|base64 --|cat /etc|cat /proc)",
    re.I | re.M,
)

# Solution steps: structured narrative
RE_STEPS = re.compile(
    r"(Step \d|^#{1,3}\s+\d|^#{1,3}\s+Step|^\d+\.\s+\w{4,}|First[,.]|"
    r"Then\s+I|Next[,.]|Finally[,.]|After that)",
    re.I | re.M,
)

# Generic wrapper user prompts (low-signal marker)
RE_GENERIC_USER = re.compile(
    r"^(Analyze the following security content and provide a structured assessment\.|"
    r"Explain the security topic: the following\. Cover what it is, how it works, detection, and mitigation\.)",
    re.M,
)

# Junk markers — almost always noise
RE_JUNK_ASST_START = re.compile(
    r"^(EVENT:|CHALLENGE:|CATEGORY:|#\s+\d+|##\s+\d+|ctf@|Hack\.|Hack |writeup)",
    re.I | re.M,
)

# Flag reference (shows real CTF content)
RE_FLAG = re.compile(r"flag\{|flag:|FLAG\{|HTB\{|THM\{", re.I)

# Code/exploit content
RE_CODE = re.compile(r"```|def |#include|import \w|function \w|class \w|public class")

# Non-English: high ratio of CJK characters
RE_CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]")


def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(RE_CJK.findall(text))
    return cjk / max(len(text), 1)


def classify(row: dict) -> tuple[str, dict]:
    """Return (tier, info) where tier is one of the KEEP/DROP classes."""
    msgs = row.get("messages", [])
    user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
    asst = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
    # multi-turn with tool role?
    has_tool_role = any(m.get("role") == "tool" for m in msgs)
    has_tool_calls = any(m.get("tool_calls") for m in msgs)
    total_chars = len(user) + len(asst)

    info = {
        "user_chars": len(user),
        "asst_chars": len(asst),
        "total_chars": total_chars,
        "turns": len(msgs),
        "has_tool_role": has_tool_role,
        "has_tool_calls": has_tool_calls,
    }

    # --- DROP rules (checked first) ---
    if len(asst) < 200:
        return "too_short", info
    if total_chars > 16_000:
        return "too_long", info
    if cjk_ratio(asst) > 0.15:
        return "non_english", info

    # --- KEEP rules ---
    # tool trajectory = multi-turn with real tool role/calls
    if has_tool_role or has_tool_calls:
        return "tool_trajectory", info

    has_decision_strong = bool(RE_DECISION_STRONG.search(asst))
    weak_hits = len(RE_DECISION_WEAK.findall(asst))
    has_decision = has_decision_strong or weak_hits >= 2
    # Require >=3 distinct terminal command hits for a real walkthrough
    terminal_hits = len(RE_TERMINAL.findall(asst))
    has_terminal = terminal_hits >= 3
    has_steps = bool(RE_STEPS.search(asst))
    has_flag = bool(RE_FLAG.search(asst))
    has_code = bool(RE_CODE.search(asst))
    generic_user = bool(RE_GENERIC_USER.search(user))
    junk_start = bool(RE_JUNK_ASST_START.search(asst))

    info.update({
        "has_decision": has_decision,
        "has_terminal": has_terminal,
        "terminal_hits": terminal_hits,
        "has_steps": has_steps,
        "has_flag": has_flag,
        "has_code": has_code,
        "generic_user": generic_user,
    })

    # Engagement decision is the highest-value pattern
    if has_decision:
        return "engagement_decision", info

    # Walkthrough: needs real tool commands (>=3 hits) plus structure
    if has_terminal and (has_steps or has_flag or has_code):
        return "walkthrough_commands", info

    # Structured solution: needs steps + flag/code/terminal
    if has_steps and (has_flag or has_terminal):
        return "solution_steps", info

    # Drop: generic wrapper + unstructured dump
    if generic_user and junk_start and not (has_steps or has_terminal):
        return "low_signal_qa", info

    # Drop: pure prose dump with no actionable structure
    if junk_start and not (has_steps or has_terminal or has_decision):
        return "raw_writeup_dump", info

    # Anything that didn't match any KEEP rule is junk by default
    return "low_signal_qa", info


KEEP_TIERS = {
    "engagement_decision",
    "walkthrough_commands",
    "solution_steps",
    "tool_trajectory",
}
DROP_TIERS = {
    "low_signal_qa",
    "raw_writeup_dump",
    "too_short",
    "too_long",
    "non_english",
}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    per_source = {}
    samples = {}
    kept_rows = []

    for src_name in SOURCES:
        src_path = CONVERTED / src_name
        if not src_path.exists():
            print(f"WARNING: {src_path} not found, skipping", file=sys.stderr)
            continue
        per_source[src_name] = Counter()
        with src_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                m = row.get("metadata") or {}
                if m.get("pillar") != "ctf":
                    continue
                tier, info = classify(row)
                counts[tier] += 1
                per_source[src_name][tier] += 1

                if tier not in samples:
                    samples[tier] = {
                        "source": src_name,
                        "user": info["user_chars"],
                        "asst": info["asst_chars"],
                        "content": next(
                            (msg.get("content", "") for msg in row["messages"] if msg.get("role") == "assistant"),
                            "",
                        )[:400],
                    }

                if tier in KEEP_TIERS:
                    # tag the row with its class for downstream selection
                    row.setdefault("metadata", {})["ctf_class"] = tier
                    kept_rows.append(row)

    # write kept rows
    with OUT_FILE.open("w") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = sum(counts.values())
    kept = sum(c for t, c in counts.items() if t in KEEP_TIERS)
    dropped = total - kept

    # print report
    print("=" * 72)
    print("  CTF USABILITY AUDIT")
    print("=" * 72)
    print(f"\nTotal CTF rows scanned: {total:,}")
    print(f"Kept:  {kept:,} ({100*kept/total:.1f}%)")
    print(f"Junk:  {dropped:,} ({100*dropped/total:.1f}%)")
    print()
    print(f"{'class':<25s} {'count':>8s} {'pct':>7s}  tier")
    print("-" * 55)
    for tier in sorted(KEEP_TIERS, key=lambda t: -counts[t]):
        c = counts[tier]
        print(f"  {tier:<23s} {c:>8,d} {100*c/total:>6.1f}%  KEEP ★★★")
    for tier in sorted(DROP_TIERS, key=lambda t: -counts[t]):
        c = counts[tier]
        print(f"  {tier:<23s} {c:>8,d} {100*c/total:>6.1f}%  DROP ✗")
    print("-" * 55)
    print(f"  {'TOTAL':<23s} {total:>8,d}")
    print()
    print(f"Per-source breakdown:")
    for src, c in per_source.items():
        src_total = sum(c.values())
        src_kept = sum(v for t, v in c.items() if t in KEEP_TIERS)
        print(f"  {src}: {src_kept:,}/{src_total:,} kept ({100*src_kept/src_total:.1f}%)")

    print(f"\nKept rows written to: {OUT_FILE}")
    print(f"Audit report:         {AUDIT_FILE}")

    # write audit json
    audit = {
        "total_scanned": total,
        "kept": kept,
        "dropped": dropped,
        "keep_pct": round(100 * kept / total, 1),
        "junk_pct": round(100 * dropped / total, 1),
        "counts": dict(counts),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "samples": samples,
        "output_file": str(OUT_FILE),
    }
    AUDIT_FILE.write_text(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
