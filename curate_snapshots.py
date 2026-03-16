#!/usr/bin/env python3
"""
Rich-Context Data Extraction — extracts (snapshot, decision) pairs from
raw Claude Code witness session JSONL files.

Instead of multi-turn conversations, this produces single-turn examples:
the user message is a structured patrol snapshot (matching serve.py's
gather_patrol_context_rich() format), and the assistant response is a
JSON tool call.

Algorithm:
1. Parse each session's JSONL into (tool_call, tool_result) pairs
2. Classify each call as GATHER or ACTION
3. Accumulate GATHER results into snapshot sections
4. On each ACTION: emit a training example (snapshot → JSON tool call)

Usage:
    python curate_snapshots.py --output-dir ./dataset/snapshots_v1
    python curate_snapshots.py --session-dirs ~/.claude/projects/*-witness/ --output-dir ./dataset/snapshots_v1
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Optional

from shared import find_session_files
from snapshot_format import (
    format_snapshot,
    is_gather_command,
    is_action_command,
    classify_gather_section,
    SECTIONS,
)
from curate import (
    classify_bash_command,
    load_session,
    TOOL_PATTERNS,
)

SYSTEM_PROMPT = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""


def extract_bash_command(record: dict) -> Optional[str]:
    """Extract bash command from an assistant record's tool_use blocks."""
    msg = record.get("message", {})
    content = msg.get("content", [])
    if not isinstance(content, list):
        return None

    for item in content:
        if item.get("type") == "tool_use" and item.get("name") == "Bash":
            return item.get("input", {}).get("command", "")
    return None


def extract_tool_result(record: dict) -> Optional[str]:
    """Extract tool result text from a user record containing tool_result."""
    msg = record.get("message", {})
    content = msg.get("content", [])
    if not isinstance(content, list):
        return None

    for item in content:
        if item.get("type") == "tool_result":
            result_content = item.get("content", "")
            if isinstance(result_content, list):
                result_content = "\n".join(
                    c.get("text", "") for c in result_content if c.get("type") == "text"
                )
            return str(result_content)[:2000]
    return None


def extract_snapshot_pairs(records: list) -> list:
    """
    Extract (snapshot, tool_call) pairs from a session's records.

    Walks through the session, accumulating GATHER results into snapshot
    sections. When an ACTION is encountered, emits the current snapshot
    paired with the action as a training example.
    """
    # Current context buffer — maps section names to accumulated content
    context = {name: "" for name in SECTIONS}
    # Default state
    context["State"] = "patrol_count: 0, idle_cycles: 0, last_action: none"

    pairs = []
    pending_cmd = None  # (command_string, tool_use_id)

    for record in records:
        rtype = record.get("type")

        if rtype == "assistant":
            cmd = extract_bash_command(record)
            if cmd:
                pending_cmd = cmd

        elif rtype == "user" and pending_cmd is not None:
            result = extract_tool_result(record)
            cmd = pending_cmd
            pending_cmd = None

            if result is None:
                continue

            if is_gather_command(cmd):
                # Accumulate into the appropriate section
                section = classify_gather_section(cmd)
                if section:
                    # For tmux has-session, empty result means alive
                    if "tmux has-session" in cmd and not result:
                        session_name = "unknown"
                        if "-t" in cmd:
                            parts = cmd.split()
                            for i, p in enumerate(parts):
                                if p == "-t" and i + 1 < len(parts):
                                    session_name = parts[i + 1]
                        result = f"{session_name}: alive"
                    if result:
                        existing = context[section]
                        if existing:
                            context[section] = existing + "\n" + result
                        else:
                            context[section] = result

            elif is_action_command(cmd):
                # Map to structured tool call
                mapped = classify_bash_command(cmd)
                if mapped:
                    tool_name, tool_args = mapped
                    tool_call = {"tool": tool_name, "args": tool_args}
                else:
                    tool_call = {"tool": "bash", "args": {"command": cmd[:300]}}

                # Build snapshot from current context
                snapshot = format_snapshot(context)

                # Only emit if we have at least one non-empty section
                non_empty = sum(
                    1 for s in SECTIONS
                    if context.get(s, "").strip() and context[s] != "N/A"
                )
                if non_empty >= 1:
                    pairs.append((snapshot, tool_call))

                # Reset volatile sections after action (keep infra/state)
                context["Inbox"] = ""
                context["Cleanup Wisps"] = ""

    return pairs


def dedup_pairs(pairs: list) -> list:
    """Remove duplicate (snapshot, tool_call) pairs by content hash."""
    seen = set()
    unique = []
    for snapshot, tool_call in pairs:
        # Hash on tool + first 200 chars of snapshot
        key = hashlib.md5(
            (json.dumps(tool_call, sort_keys=True) + snapshot[:200]).encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append((snapshot, tool_call))
    return unique


def pairs_to_training(pairs: list) -> list:
    """Convert (snapshot, tool_call) pairs to chat-format training examples."""
    examples = []
    for snapshot, tool_call in pairs:
        example = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": snapshot},
            {"role": "assistant", "content": json.dumps(tool_call)},
        ]
        examples.append(example)
    return examples




def main():
    parser = argparse.ArgumentParser(
        description="Extract (snapshot, decision) pairs from witness sessions"
    )
    parser.add_argument("--session-dirs", nargs="+", default=None,
                        help="Session directories to scan")
    parser.add_argument("--output-dir", default="./dataset/snapshots_v1",
                        help="Output directory for training data")
    parser.add_argument("--min-quality", type=int, default=1,
                        help="Min non-empty snapshot sections for inclusion")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/eval split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true",
                        help="Print stats without writing files")
    args = parser.parse_args()

    # Default: all witness session dirs
    if args.session_dirs is None:
        claude_projects = os.path.expanduser("~/.claude/projects/")
        args.session_dirs = glob.glob(os.path.join(claude_projects, "*-witness"))
        if not args.session_dirs:
            # Fallback: scan all project dirs
            args.session_dirs = glob.glob(os.path.join(claude_projects, "*"))

    print(f"Scanning {len(args.session_dirs)} session directories...")
    files = find_session_files(args.session_dirs)
    print(f"Found {len(files)} session files in size range")

    # Process all sessions
    all_pairs = []
    sessions_with_pairs = 0

    for i, path in enumerate(files):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i+1}/{len(files)}...", file=sys.stderr)

        try:
            records = load_session(path)
            pairs = extract_snapshot_pairs(records)
            if pairs:
                sessions_with_pairs += 1
                all_pairs.extend(pairs)
        except Exception as e:
            print(f"  ERROR: {path}: {e}", file=sys.stderr)

    print(f"\nExtraction results:")
    print(f"  Sessions scanned: {len(files)}")
    print(f"  Sessions with pairs: {sessions_with_pairs}")
    print(f"  Raw pairs extracted: {len(all_pairs)}")

    # Dedup
    unique_pairs = dedup_pairs(all_pairs)
    print(f"  After dedup: {len(unique_pairs)}")

    # Tool distribution
    tool_counts = Counter(p[1]["tool"] for p in unique_pairs)
    print(f"\nTool distribution:")
    for tool, count in tool_counts.most_common():
        print(f"  {tool:30s} {count:5d} ({count/len(unique_pairs)*100:.1f}%)")

    # Snapshot quality (section coverage)
    section_counts = Counter()
    for snapshot, _ in unique_pairs:
        for section in SECTIONS:
            if f"## {section}\n" in snapshot:
                content_after = snapshot.split(f"## {section}\n")[1].split("\n\n##")[0]
                if content_after.strip() and content_after.strip() != "N/A":
                    section_counts[section] += 1
    print(f"\nSection coverage (across {len(unique_pairs)} pairs):")
    for section in SECTIONS:
        count = section_counts.get(section, 0)
        print(f"  {section:20s} {count:5d} ({count/len(unique_pairs)*100:.1f}%)")

    if args.stats_only:
        return

    # Convert to training format
    examples = pairs_to_training(unique_pairs)

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(examples)
    split_idx = int(len(examples) * args.split)
    train = examples[:split_idx]
    eval_set = examples[split_idx:]

    print(f"\nSplit: {len(train)} train, {len(eval_set)} eval")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "train.jsonl"), "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(os.path.join(args.output_dir, "eval.jsonl"), "w") as f:
        for ex in eval_set:
            f.write(json.dumps(ex) + "\n")

    # Metadata
    metadata = {
        "total_sessions": len(files),
        "sessions_with_pairs": sessions_with_pairs,
        "raw_pairs": len(all_pairs),
        "unique_pairs": len(unique_pairs),
        "train_count": len(train),
        "eval_count": len(eval_set),
        "tool_distribution": dict(tool_counts),
        "section_coverage": dict(section_counts),
        "seed": args.seed,
        "split": args.split,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput written to {args.output_dir}/")
    print(f"  train.jsonl  ({len(train)} examples)")
    print(f"  eval.jsonl   ({len(eval_set)} examples)")
    print(f"  metadata.json")

    # Show samples
    print(f"\nSample examples:")
    for ex in train[:3]:
        ctx = ex[1]["content"][:150].replace("\n", " ")
        tool = json.loads(ex[2]["content"])
        print(f"  Context: {ctx}...")
        print(f"  Tool: {tool['tool']}, Args: {json.dumps(tool.get('args',{}))[:80]}")
        print()


if __name__ == "__main__":
    main()
