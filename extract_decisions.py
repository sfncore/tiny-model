#!/usr/bin/env python3
"""
Extract decision-focused training pairs from multi-turn conversations.

Problem: Multi-turn SFT trains on ALL turns, so the model learns "none" is the
most common response (42.6% of first turns). When eval gives a single context
message, the model defaults to "none".

Fix: Extract (context → action) pairs where each example ends with a meaningful
tool call. Include balanced "none" examples for idle scenarios.

Usage:
    python extract_decisions.py
    python extract_decisions.py --max-context-turns 6 --none-ratio 0.2
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


# Action tools — these are the ones we want the model to learn to call
ACTION_TOOLS = {
    "gt_nudge", "gt_polecat_nuke", "gt_peek", "gt_session_status",
    "gt_polecat_list", "gt_mail_inbox", "gt_mail_read", "gt_mail_send",
    "gt_patrol_report", "gt_handoff", "gt_escalate",
    "check_git_state", "check_tmux_session",
}

# Info-gathering tools — keep some but don't prioritize
INFO_TOOLS = {"bd_show", "bd_list", "bd_close", "bd_children", "bash", "none"}

SYSTEM_PROMPT_JSON = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""


def extract_tool(content: str) -> str | None:
    """Extract tool name from a Format B assistant message."""
    try:
        parsed = json.loads(content)
        return parsed.get("tool")
    except (json.JSONDecodeError, TypeError):
        return None


def extract_decision_pairs(conv: list, max_context_turns: int = 8) -> list:
    """
    Extract (context → decision) pairs from a multi-turn conversation.

    For each assistant turn with an action tool, create a training example
    containing the preceding context + that assistant turn.
    """
    pairs = []
    messages = []  # running list of (role, content)

    for m in conv:
        role = m.get("role", "")
        content = m.get("content", "") or ""

        if role == "system":
            continue  # We'll add our own system prompt

        if role == "user":
            messages.append({"role": "user", "content": content})

        elif role == "assistant":
            tool = extract_tool(content)
            if tool and tool in ACTION_TOOLS:
                # Create a training pair: context → this action
                # Take last N messages as context
                context = messages[-max_context_turns:] if len(messages) > max_context_turns else messages[:]
                if context:  # need at least one user message
                    pair = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]
                    pair.extend(context)
                    pair.append({"role": "assistant", "content": content})
                    pairs.append(pair)

            # Add to running context regardless
            messages.append({"role": "assistant", "content": content})

    return pairs


def extract_none_pairs(conv: list, max_context_turns: int = 8) -> list:
    """
    Extract examples where "none" is the correct answer.
    Only take the first "none" turn from each conversation (idle patrol pattern).
    """
    pairs = []
    messages = []

    for m in conv:
        role = m.get("role", "")
        content = m.get("content", "") or ""

        if role == "system":
            continue
        if role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool = extract_tool(content)
            if tool == "none" and messages:
                # Only take the first none per conversation
                context = messages[-max_context_turns:] if len(messages) > max_context_turns else messages[:]
                pair = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]
                pair.extend(context)
                pair.append({"role": "assistant", "content": content})
                pairs.append(pair)
                return pairs  # Only one per conversation
            messages.append({"role": "assistant", "content": content})

    return pairs


def main():
    parser = argparse.ArgumentParser(description="Extract decision-focused training pairs")
    parser.add_argument("--input", default="./dataset/format_b/chunked_2k/train.jsonl",
                        help="Input training JSONL")
    parser.add_argument("--eval-input", default="./dataset/format_b/chunked_2k/eval.jsonl",
                        help="Input eval JSONL")
    parser.add_argument("--output-dir", default="./dataset/format_b_decisions",
                        help="Output directory")
    parser.add_argument("--max-context-turns", type=int, default=8,
                        help="Max context turns to include before decision")
    parser.add_argument("--none-ratio", type=float, default=0.15,
                        help="Target ratio of 'none' examples (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Process train and eval sets
    for split, input_path in [("train", args.input), ("eval", args.eval_input)]:
        print(f"\n{'='*60}")
        print(f"Processing {split}: {input_path}")
        print(f"{'='*60}")

        with open(input_path) as f:
            conversations = [json.loads(line) for line in f]
        print(f"  Loaded {len(conversations)} conversations")

        # Extract action pairs
        action_pairs = []
        for conv in conversations:
            action_pairs.extend(extract_decision_pairs(conv, args.max_context_turns))

        # Extract none pairs
        none_pairs = []
        for conv in conversations:
            none_pairs.extend(extract_none_pairs(conv, args.max_context_turns))

        print(f"  Extracted {len(action_pairs)} action pairs")
        print(f"  Extracted {len(none_pairs)} none pairs")

        # Balance: include enough "none" to hit target ratio
        n_action = len(action_pairs)
        n_none_target = int(n_action * args.none_ratio / (1 - args.none_ratio))
        n_none_target = min(n_none_target, len(none_pairs))

        if n_none_target < len(none_pairs):
            none_sample = rng.sample(none_pairs, n_none_target)
        else:
            none_sample = none_pairs

        all_pairs = action_pairs + none_sample
        rng.shuffle(all_pairs)

        print(f"  Final: {len(all_pairs)} pairs ({len(action_pairs)} action + {len(none_sample)} none)")

        # Tool distribution
        tool_counts = Counter()
        for pair in all_pairs:
            last_msg = pair[-1]
            if last_msg["role"] == "assistant":
                tool = extract_tool(last_msg["content"])
                tool_counts[tool or "UNKNOWN"] += 1

        print(f"\n  Tool distribution (decision turns only):")
        for tool, count in tool_counts.most_common():
            print(f"    {tool:30s} {count:5d} ({count/len(all_pairs)*100:.1f}%)")

        # Conversation length stats
        lengths = [len(p) - 1 for p in all_pairs]  # -1 for system prompt
        print(f"\n  Turns per example: min={min(lengths)}, median={sorted(lengths)[len(lengths)//2]}, max={max(lengths)}")

        # Write output
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{split}.jsonl")
        with open(output_path, "w") as f:
            for pair in all_pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"\n  Written to {output_path}")

    print(f"\n{'='*60}")
    print("Done! Use with train.py:")
    print(f"  python train.py --model smollm2-135m --format b --epochs 3 \\")
    print(f"    --dataset-dir {args.output_dir} --use-chunked=false \\")
    print(f"    --max-length 2048")
    print(f"\nNote: pass the output dir as dataset-dir and DON'T use --use-chunked")
    print(f"since these are already short, focused examples.")


if __name__ == "__main__":
    main()
