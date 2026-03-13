#!/usr/bin/env python3
"""
Extract decision-focused training pairs from multi-turn conversations.

Problem: Multi-turn SFT trains on ALL turns, so the model learns "none" is the
most common response (42.6% of first turns). When eval gives a single context
message, the model defaults to "none".

Fix: Extract (context → action) pairs where each example ends with a meaningful
tool call. Include balanced "none" examples for idle scenarios.

Handles both formats:
  - Format B (pre-converted): assistant content is JSON like {"tool": "...", "args": {...}}
  - OpenAI function-calling: assistant has tool_calls array with function.name/arguments

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


def extract_tool_from_message(m: dict) -> tuple[str | None, str | None]:
    """
    Extract tool name and Format B JSON from a message.

    Returns (tool_name, format_b_json) or (None, None).

    Handles:
      1. Format B: content is already JSON like {"tool": "...", "args": {...}}
      2. OpenAI function-calling: tool_calls array with function.name/arguments
    """
    # Try Format B first (content is JSON)
    content = m.get("content", "") or ""
    if content:
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and "tool" in parsed:
                return parsed["tool"], content
        except (json.JSONDecodeError, TypeError):
            pass

    # Try OpenAI function-calling format
    tool_calls = m.get("tool_calls", [])
    if tool_calls:
        tc = tool_calls[0]  # Take first tool call
        func = tc.get("function", {})
        name = func.get("name", "")
        if name:
            try:
                args = json.loads(func.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}
            format_b = json.dumps({"tool": name, "args": args})
            return name, format_b

    return None, None


def build_context_message(m: dict) -> str:
    """
    Build a context string from a message for the user context window.
    Combines assistant text + tool results into a coherent context turn.
    """
    content = m.get("content", "") or ""
    return content


def extract_decision_pairs(conv: list, max_context_turns: int = 8) -> list:
    """
    Extract (context → decision) pairs from a multi-turn conversation.

    For each assistant turn with an action tool, create a training example
    containing the preceding context + that assistant turn as Format B JSON.
    """
    pairs = []
    context_messages = []  # running context of user + tool messages

    for m in conv:
        role = m.get("role", "")

        if role == "system":
            continue

        if role == "user":
            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "user", "content": content})

        elif role == "tool":
            # Tool responses become user context (the witness sees tool output)
            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "user", "content": content})

        elif role == "assistant":
            tool_name, format_b = extract_tool_from_message(m)

            if tool_name and tool_name in ACTION_TOOLS:
                # Create a training pair: context → this action
                context = context_messages[-max_context_turns:] if len(context_messages) > max_context_turns else context_messages[:]
                if context:
                    pair = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]
                    pair.extend(context)
                    pair.append({"role": "assistant", "content": format_b})
                    pairs.append(pair)

            # Add assistant text to context if it has content
            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "assistant", "content": content})

    return pairs


def extract_none_pairs(conv: list, max_context_turns: int = 8) -> list:
    """
    Extract examples where "none" is the correct answer.

    In raw sessions, idle patrols show as assistant messages with no tool calls
    following a context that shows everything is healthy. We look for patterns
    where the witness saw healthy state and took no action.
    """
    pairs = []
    context_messages = []

    # Track consecutive user/tool messages without an action tool call
    idle_contexts = []

    for m in conv:
        role = m.get("role", "")

        if role == "system":
            continue
        if role == "user":
            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "user", "content": content})
        elif role == "tool":
            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_name, _ = extract_tool_from_message(m)

            # If assistant has text but no action tool → potential idle/none
            if not tool_name or tool_name == "none":
                content = (m.get("content", "") or "").strip()
                # Look for idle patrol indicators
                if context_messages and _looks_idle(context_messages):
                    context = context_messages[-max_context_turns:] if len(context_messages) > max_context_turns else context_messages[:]
                    none_json = json.dumps({"tool": "none", "args": {}})
                    pair = [{"role": "system", "content": SYSTEM_PROMPT_JSON}]
                    pair.extend(context)
                    pair.append({"role": "assistant", "content": none_json})
                    pairs.append(pair)
                    return pairs  # Only one per conversation

            content = build_context_message(m)
            if content.strip():
                context_messages.append({"role": "assistant", "content": content})

    return pairs


def _looks_idle(context: list) -> bool:
    """Heuristic: does the recent context suggest an idle/healthy state?"""
    last_few = context[-3:]
    text = " ".join(m.get("content", "") for m in last_few).lower()
    idle_signals = ["no polecats", "idle", "no unread", "patrol_count", "healthy",
                    "no molecules", "nothing on hook", "all clear", "quiet"]
    return any(signal in text for signal in idle_signals)


def main():
    parser = argparse.ArgumentParser(description="Extract decision-focused training pairs")
    parser.add_argument("--input", default="./dataset/curated/train_openai.jsonl",
                        help="Input training JSONL")
    parser.add_argument("--eval-input", default="./dataset/curated/eval_openai.jsonl",
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
            msgs = conv.get("messages", conv) if isinstance(conv, dict) else conv
            action_pairs.extend(extract_decision_pairs(msgs, args.max_context_turns))

        # Extract none pairs
        none_pairs = []
        for conv in conversations:
            msgs = conv.get("messages", conv) if isinstance(conv, dict) else conv
            none_pairs.extend(extract_none_pairs(msgs, args.max_context_turns))

        print(f"  Extracted {len(action_pairs)} action pairs")
        print(f"  Extracted {len(none_pairs)} none pairs")

        # Balance: include enough "none" to hit target ratio
        n_action = len(action_pairs)
        if n_action == 0:
            print("  WARNING: No action pairs found!")
            n_none_target = len(none_pairs)
        else:
            n_none_target = int(n_action * args.none_ratio / (1 - args.none_ratio))
            n_none_target = min(n_none_target, len(none_pairs))

        if n_none_target < len(none_pairs):
            none_sample = rng.sample(none_pairs, n_none_target)
        else:
            none_sample = none_pairs

        all_pairs = action_pairs + none_sample
        rng.shuffle(all_pairs)

        print(f"  Final: {len(all_pairs)} pairs ({len(action_pairs)} action + {len(none_sample)} none)")

        if not all_pairs:
            print("  Skipping output — no pairs extracted")
            continue

        # Tool distribution
        tool_counts = Counter()
        for pair in all_pairs:
            last_msg = pair[-1]
            if last_msg["role"] == "assistant":
                try:
                    parsed = json.loads(last_msg["content"])
                    tool_counts[parsed.get("tool", "UNKNOWN")] += 1
                except (json.JSONDecodeError, TypeError):
                    tool_counts["UNKNOWN"] += 1

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
    print(f"    --dataset-dir {args.output_dir}")
    print(f"\nNote: pass the output dir as dataset-dir.")


if __name__ == "__main__":
    main()
