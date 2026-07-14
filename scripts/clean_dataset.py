#!/usr/bin/env python3
"""clean_dataset.py — clean the cyb3r-dataset-agent-v2 train/eval splits.

Removes:
  1. Exact duplicate rows (normalized hash)
  2. Rows truncated at exactly 8000 chars (CTF writeup scraping artifact)
  3. Rows containing HTML/scrape page-chrome (GitHub README navigation, etc.)
     — uses TIGHT patterns that never match legitimate HTTP tool output
  4. Rows ending with a tool message (broken conversation flow)
  5. Rows with non-printable/garbled content

DOES NOT remove:
  - Rows mentioning "rate limit", "cloudflare", "captcha", "access denied"
    in natural English (these are legitimate security reasoning)
  - Rows containing HTML in tool outputs (legitimate HTTP responses)
  - Short assistant responses (legitimate one-line decisions like "ABORT")

Usage:
  # dry-run (shows what would be removed, writes nothing)
  python scripts/clean_dataset.py --dry-run

  # actually clean (writes to <output-dir>, preserves originals)
  python scripts/clean_dataset.py \\
    --input-dir /home/dudu/datasets/cyb3r-dataset-agent-v2 \\
    --output-dir /home/dudu/datasets/cyb3r-dataset-agent-v2-clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


# ─── detection patterns ─────────────────────────────────────────────────────

# TIGHT scrape-chrome detection: only matches HTML page chrome that NEVER
# appears in legitimate HTTP responses or security reasoning. Tuned so that
# real pentest tool output (which legitimately contains <html>, <title>,
# <meta name="generator" content="WordPress">, etc.) is NOT flagged, AND
# security reasoning that quotes "Attention Required! | Cloudflare" as an
# observation is NOT flagged.
RE_SCRAPED_CHROME = re.compile(
    r"github\.githubassets\.com"
    # GitHub theme chrome — only appears in <html data-color-mode=...> tag
    r"|<html[^>]*data-color-mode"
    r"|<html[^>]*data-light-theme"
    r"|<html[^>]*data-dark-theme"
    # SEO/preload boilerplate inside <head>
    r"|<link[^>]*dns-prefetch[^>]*github\.githubassets"
    r"|<link[^>]*preconnect[^>]*github\.githubassets"
    r"|<meta\s+name=[\"']color-scheme"
    r"|<link\s+rel=[\"']icon[\"']\s+href=[\"']data:image"
    # Cloudflare interstitial block page — requires the literal HTML title
    # tag, not just the phrase in prose
    r"|<title>\s*Attention Required!\s*\|\s*Cloudflare\s*</title>"
    r"|cf-browser-verification"
    r"|cf-mitigated",
    re.I | re.DOTALL,
)

# The exact char length at which the original CTF scraper truncated writeups.
# All rows with assistant content of EXACTLY this length are mid-sentence clips.
CLIPPED_LENGTH = 8000

# Non-printable/garbled content: > 5% non-text characters (excludes valid
# unicode like CJK, accented Latin, emoji).
RE_NONPRINT = re.compile(r"[^\x09\x0a\x0d\x20-\x7e\xc0-\xff\u0100-\uffff]")


# ─── helpers ────────────────────────────────────────────────────────────────

def row_hash(row: dict) -> str:
    """Stable normalized hash of message content (ignores metadata)."""
    msgs = row.get("messages", [])
    key = json.dumps(
        [{"role": m.get("role"), "content": m.get("content", "")} for m in msgs],
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(key.encode()).hexdigest()


def assistant_text(row: dict) -> str:
    return next(
        (m.get("content", "") for m in row["messages"] if m.get("role") == "assistant"),
        "",
    )


def all_text(row: dict) -> str:
    return " ".join(m.get("content", "") for m in row.get("messages", []))


def classify(row: dict) -> tuple[str | None, dict]:
    """Return (drop_reason or None, info dict). None means keep."""
    msgs = row.get("messages", [])
    roles = [m.get("role") for m in msgs]
    asst = assistant_text(row)
    text = all_text(row)
    info = {
        "asst_chars": len(asst),
        "total_chars": len(text),
        "turns": len(msgs),
    }

    # 1. truncated mid-sentence (exactly 8000 chars)
    if len(asst) == CLIPPED_LENGTH:
        return "clipped_8000", info

    # 2. scrape page-chrome (GitHub README nav, etc.) — TIGHT detection only
    if RE_SCRAPED_CHROME.search(text):
        return "scrape_chrome", info

    # 3. ends with tool message (broken conversation flow)
    if roles and roles[-1] == "tool":
        return "ends_with_tool", info

    # 4. garbled/non-printable dominant content
    if asst and len(RE_NONPRINT.findall(asst)) / max(len(asst), 1) > 0.05:
        return "garbled", info

    return None, info


def clean_rows(rows: list[dict]) -> tuple[list[dict], list[tuple[str, dict, dict]], Counter]:
    """Dedupe + classify. Returns (kept, dropped, reason_counts).

    dropped is a list of (reason, row, info) tuples for inspection.
    """
    kept: list[dict] = []
    dropped: list[tuple[str, dict, dict]] = []
    reasons: Counter = Counter()
    seen_hashes: set[str] = set()

    for row in rows:
        h = row_hash(row)

        # dedupe first (most common issue)
        if h in seen_hashes:
            reasons["duplicate"] += 1
            dropped.append(("duplicate", row, {"asst_chars": len(assistant_text(row))}))
            continue
        seen_hashes.add(h)

        # then quality filters
        reason, info = classify(row)
        if reason is not None:
            reasons[reason] += 1
            dropped.append((reason, row, info))
        else:
            kept.append(row)

    return kept, dropped, reasons


# ─── main ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", default="/home/dudu/datasets/cyb3r-dataset-agent-v2",
                   help="Source dataset directory (with train.jsonl + eval.jsonl)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: <input-dir>-clean)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be removed, write nothing")
    return p.parse_args()


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


def report(name: str, n_before: int, kept: list[dict], dropped: list, reasons: Counter) -> None:
    n_drop = len(dropped)
    print(f"\n{'=' * 72}")
    print(f"  {name.upper()}: {n_before:,} → {len(kept):,} kept, {n_drop:,} dropped ({100*n_drop/n_before:.1f}%)")
    print(f"{'=' * 72}")
    for reason, n in reasons.most_common():
        print(f"  {reason:20s} {n:>5,d}  ({100*n/n_before:.2f}%)")
    # show trace_type breakdown after cleaning
    tt = Counter(r.get("metadata", {}).get("trace_type", "?") for r in kept)
    print(f"\n  trace_type breakdown after cleaning:")
    for t, n in tt.most_common():
        print(f"    {t:35s} {n:>5,d}")


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir) if args.output_dir else in_dir.parent / f"{in_dir.name}-clean"

    if not (in_dir / "train.jsonl").exists():
        sys.exit(f"ERROR: {in_dir}/train.jsonl not found")

    print(f"input:  {in_dir}")
    print(f"output: {out_dir}")
    print(f"mode:   {'DRY-RUN (no files written)' if args.dry_run else 'WRITE'}")

    results = {}
    for split in ("train", "eval"):
        path = in_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"\n(skip {split}: not found)")
            continue
        rows = load_jsonl(path)
        kept, dropped, reasons = clean_rows(rows)
        report(split, len(rows), kept, dropped, reasons)
        results[split] = (rows, kept, dropped, reasons)

    # sample inspection: show 2 examples of each drop reason from train
    train_dropped = results.get("train", (None, None, None, None))[2]
    if train_dropped:
        by_reason = {}
        for reason, row, info in train_dropped:
            by_reason.setdefault(reason, []).append((row, info))
        print(f"\n{'=' * 72}")
        print("  SAMPLE DROPPED ROWS (for sanity-check)")
        print(f"{'=' * 72}")
        for reason in sorted(by_reason.keys()):
            samples = by_reason[reason][:2]
            print(f"\n  --- {reason} ({len(by_reason[reason])} total) ---")
            for row, info in samples:
                asst = assistant_text(row)
                tt = row.get("metadata", {}).get("trace_type", "?")
                preview = asst[:200].replace("\n", " ")
                print(f"    [{tt}] {preview}...")

    # write if not dry-run
    if not args.dry_run:
        for split, (rows, kept, dropped, reasons) in results.items():
            write_jsonl(out_dir / f"{split}.jsonl", kept)
            print(f"\nwrote {out_dir / f'{split}.jsonl'} ({len(kept):,} rows)")
        # copy manifest with cleaning stats
        manifest_path = in_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest["cleaned"] = {
                "tool": "scripts/clean_dataset.py",
                "original_train": len(results["train"][0]) if "train" in results else 0,
                "cleaned_train": len(results["train"][1]) if "train" in results else 0,
                "original_eval": len(results["eval"][0]) if "eval" in results else 0,
                "cleaned_eval": len(results["eval"][1]) if "eval" in results else 0,
                "drop_reasons_train": dict(results["train"][3]) if "train" in results else {},
                "drop_reasons_eval": dict(results["eval"][3]) if "eval" in results else {},
            }
            (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            print(f"wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
