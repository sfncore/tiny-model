#!/usr/bin/env python3
"""
Dataset Preprocessor — Multi-format output + class balancing.

Takes the curated dataset from curate.py and produces training-ready
data in three output formats, with class balancing applied.

Usage:
    python preprocess.py [--input-dir DIR] [--output-dir DIR]
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

# The witness tools that matter for structured output
WITNESS_TOOLS = {
    "gt_polecat_list", "gt_polecat_nuke", "gt_peek", "gt_session_status",
    "gt_nudge", "gt_mail_inbox", "gt_mail_read", "gt_mail_send",
    "gt_patrol_report", "gt_handoff", "gt_escalate",
    "bd_show", "bd_list", "bd_close", "bd_children",
    "check_git_state", "check_tmux_session",
    "bash",  # keep as escape hatch
}

SYSTEM_PROMPT = """You are a Witness agent monitoring worker agents (polecats) in a multi-agent workspace.

Your responsibilities:
1. HEALTH MONITORING: Check if polecats are stuck, crashed, or healthy
2. ESCALATION: Nudge stuck polecats, escalate to Mayor after repeated failures
3. PRE-KILL VERIFICATION: Check git state before nuking polecats
4. CLEANUP: Send MERGE_READY to refinery, then nuke completed polecats
5. INFRASTRUCTURE: Verify deacon and refinery sessions are alive

Decision rules:
- If no polecats active and infrastructure healthy → report idle patrol
- If a polecat is idle with no progress → nudge it
- If 3+ nudges with no response → escalate to Mayor
- If polecat completed work and git is clean → nuke it
- If polecat has unpushed work → escalate for recovery, do NOT nuke
- If polecat is in crash loop (3+ restarts) → force nuke and escalate
- After extraordinary actions (force nuke, escalation) → hand off to fresh session"""

SYSTEM_PROMPT_JSON = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""


# -------------------------------------------------------------------
# Conversation classification
# -------------------------------------------------------------------

def classify_conversation(conv: list) -> str:
    """Classify a conversation by its dominant decision pattern."""
    blob = " ".join(m.get("content", "") or "" for m in conv)

    has_nuke = "gt_polecat_nuke" in blob
    has_nudge = "gt_nudge" in blob
    has_escalate = ("gt_escalate" in blob or
                    ("gt_mail_send" in blob and
                     any(kw in blob.upper() for kw in ["ESCALAT", "RECOVERY_NEEDED"])))
    has_force = '"force": true' in blob or "--force" in blob
    has_polecats = ("Polecats\n" in blob and "No polecats" not in blob)

    action_count = sum([has_nuke, has_nudge, has_escalate])

    if action_count >= 2:
        return "mixed"
    elif has_escalate:
        return "escalate"
    elif has_force:
        return "nuke_force"
    elif has_nuke:
        return "nuke_clean"
    elif has_nudge:
        return "nudge"
    elif has_polecats:
        return "monitor"
    else:
        return "idle"


def balance_classes(conversations: list, seed: int = 42) -> list:
    """
    Balance classes by oversampling rare patterns and undersampling idle.
    Target: no class more than 3x any other class.
    """
    by_class = {}
    for conv in conversations:
        cls = classify_conversation(conv)
        by_class.setdefault(cls, []).append(conv)

    # Find the median class size
    sizes = sorted(len(v) for v in by_class.values())
    target = sizes[len(sizes) // 2]  # median
    target = max(target, 50)  # at least 50 per class

    rng = random.Random(seed)
    balanced = []

    for cls, convs in by_class.items():
        if len(convs) >= target:
            # Undersample (but keep at least target)
            balanced.extend(rng.sample(convs, min(len(convs), target * 2)))
        else:
            # Oversample
            balanced.extend(convs)
            extra_needed = target - len(convs)
            balanced.extend(rng.choices(convs, k=extra_needed))

    rng.shuffle(balanced)
    return balanced


# -------------------------------------------------------------------
# Format converters
# -------------------------------------------------------------------

def to_format_a(conv: list) -> list:
    """Format A: inline tags. Free text + <tool_call>fn(args)</tool_call>"""
    result = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in conv:
        if m["role"] == "system":
            continue  # replaced by our system prompt
        elif m["role"] == "user":
            content = m.get("content", "")
            if content and "<tool_result>" not in content:
                result.append({"role": "user", "content": content})
            elif "<tool_result>" in content:
                result.append({"role": "user", "content": content})
        elif m["role"] == "assistant":
            result.append({"role": "assistant", "content": m.get("content", "") or ""})

    return result


def to_format_b(conv: list) -> list:
    """Format B: pure JSON output. Model outputs only {"tool": ..., "args": ...}"""
    result = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]

    for m in conv:
        if m["role"] == "system":
            continue
        elif m["role"] == "user":
            content = m.get("content", "")
            if "<tool_result>" in content:
                # Strip tags, just give raw result
                inner = content.replace("<tool_result>", "").replace("</tool_result>", "").strip()
                result.append({"role": "user", "content": inner})
            elif content:
                result.append({"role": "user", "content": content})
        elif m["role"] == "assistant":
            content = m.get("content", "") or ""
            # Extract tool calls and convert to JSON
            calls = re.findall(r'<tool_call>(\w+)\((.*?)\)</tool_call>', content, re.DOTALL)
            if calls:
                for fn_name, fn_args in calls:
                    try:
                        args = json.loads(fn_args)
                    except json.JSONDecodeError:
                        args = {"raw": fn_args}
                    result.append({
                        "role": "assistant",
                        "content": json.dumps({"tool": fn_name, "args": args})
                    })
            else:
                # Text-only response — model says "no action" or summarizes
                text = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                if text:
                    result.append({
                        "role": "assistant",
                        "content": json.dumps({"tool": "none", "args": {}, "note": text[:200]})
                    })

    return result


def to_format_c(conv: list) -> list:
    """Format C: FunctionGemma-style control tokens."""
    result = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in conv:
        if m["role"] == "system":
            continue
        elif m["role"] == "user":
            content = m.get("content", "")
            if "<tool_result>" in content:
                # Convert to fn_response format
                inner = content.replace("<tool_result>", "").replace("</tool_result>", "").strip()
                result.append({"role": "user", "content": f"<fn_response>{inner}</fn_response>"})
            elif content:
                result.append({"role": "user", "content": content})
        elif m["role"] == "assistant":
            content = m.get("content", "") or ""
            # Convert tool_call tags to fn_call tags
            converted = re.sub(
                r'<tool_call>(\w+)\((.*?)\)</tool_call>',
                lambda m: f'<fn_call>{m.group(1)}{m.group(2)}</fn_call>',
                content,
                flags=re.DOTALL
            )
            if converted.strip():
                result.append({"role": "assistant", "content": converted})

    return result


# -------------------------------------------------------------------
# Chunking for small context windows
# -------------------------------------------------------------------

def chunk_conversation(conv: list, max_tokens: int = 2048, chars_per_token: int = 4) -> list:
    """Split a conversation into chunks that fit within a token budget."""
    max_chars = max_tokens * chars_per_token
    chunks = []
    current = [conv[0]] if conv and conv[0]["role"] == "system" else []
    system_msg = current[0] if current else None
    current_size = sum(len(m.get("content", "") or "") for m in current)

    for m in conv:
        if m["role"] == "system":
            continue
        msg_size = len(m.get("content", "") or "")

        if current_size + msg_size > max_chars and len(current) > 1:
            # Save current chunk
            chunks.append(current[:])
            # Start new chunk with system prompt
            current = [system_msg] if system_msg else []
            current_size = sum(len(m.get("content", "") or "") for m in current)

        current.append(m)
        current_size += msg_size

    if len(current) > 1:  # more than just system prompt
        chunks.append(current)

    return chunks


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess curated dataset into multi-format training data")
    parser.add_argument("--input-dir", default="./dataset",
                        help="Input directory from curate.py")
    parser.add_argument("--output-dir", default="./dataset",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balance", action="store_true", default=True,
                        help="Apply class balancing to training data")
    parser.add_argument("--chunk-size", type=int, default=2048,
                        help="Token budget for chunked variant")
    args = parser.parse_args()

    # Load curated data (nanochat format — has the inline tags)
    train_convs = []
    with open(os.path.join(args.input_dir, "train_nanochat.jsonl")) as f:
        for line in f:
            train_convs.append(json.loads(line))

    eval_convs = []
    with open(os.path.join(args.input_dir, "eval_nanochat.jsonl")) as f:
        for line in f:
            eval_convs.append(json.loads(line))

    print(f"Loaded: {len(train_convs)} train, {len(eval_convs)} eval")

    # Classify
    train_classes = Counter(classify_conversation(c) for c in train_convs)
    print(f"\nTrain class distribution (before balancing):")
    for cls, count in train_classes.most_common():
        print(f"  {cls:15s} {count:5d} ({count/len(train_convs)*100:5.1f}%)")

    # Balance training data
    if args.balance:
        train_balanced = balance_classes(train_convs, seed=args.seed)
        balanced_classes = Counter(classify_conversation(c) for c in train_balanced)
        print(f"\nTrain class distribution (after balancing):")
        for cls, count in balanced_classes.most_common():
            print(f"  {cls:15s} {count:5d} ({count/len(train_balanced)*100:5.1f}%)")
    else:
        train_balanced = train_convs

    # Generate all three formats
    for fmt_name, converter in [("format_a", to_format_a),
                                 ("format_b", to_format_b),
                                 ("format_c", to_format_c)]:
        fmt_dir = os.path.join(args.output_dir, fmt_name)
        os.makedirs(fmt_dir, exist_ok=True)

        # Full conversations
        with open(os.path.join(fmt_dir, "train.jsonl"), "w") as f:
            for conv in train_balanced:
                converted = converter(conv)
                f.write(json.dumps(converted) + "\n")

        with open(os.path.join(fmt_dir, "eval.jsonl"), "w") as f:
            for conv in eval_convs:
                converted = converter(conv)
                f.write(json.dumps(converted) + "\n")

        # Chunked variant (for 2K context models)
        chunked_dir = os.path.join(fmt_dir, "chunked_2k")
        os.makedirs(chunked_dir, exist_ok=True)

        train_chunks = []
        for conv in train_balanced:
            converted = converter(conv)
            chunks = chunk_conversation(converted, max_tokens=args.chunk_size)
            train_chunks.extend(chunks)

        eval_chunks = []
        for conv in eval_convs:
            converted = converter(conv)
            chunks = chunk_conversation(converted, max_tokens=args.chunk_size)
            eval_chunks.extend(chunks)

        with open(os.path.join(chunked_dir, "train.jsonl"), "w") as f:
            for chunk in train_chunks:
                f.write(json.dumps(chunk) + "\n")

        with open(os.path.join(chunked_dir, "eval.jsonl"), "w") as f:
            for chunk in eval_chunks:
                f.write(json.dumps(chunk) + "\n")

        # Stats
        train_tokens = sum(
            sum(len(m.get("content", "") or "") for m in converter(c)) / 4
            for c in train_balanced
        )
        print(f"\n{fmt_name}:")
        print(f"  Full:    {len(train_balanced)} train, {len(eval_convs)} eval")
        print(f"  Chunked: {len(train_chunks)} train, {len(eval_chunks)} eval")
        print(f"  Est train tokens: {train_tokens:,.0f}")

        # Sample one conversation
        sample = converter(train_balanced[0])
        print(f"  Sample ({len(sample)} messages):")
        for m in sample[:4]:
            content = (m.get("content") or "")[:100]
            print(f"    [{m['role']:9s}] {content}")

    print(f"\nOutput written to {args.output_dir}/format_{{a,b,c}}/")


if __name__ == "__main__":
    main()
