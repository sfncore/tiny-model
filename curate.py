#!/usr/bin/env python3
"""
Witness Session Curation Pipeline

Extracts clean training data from raw Claude Code witness session JSONL files.
Produces conversation datasets in both nanochat CustomJSON and OpenAI ChatML formats.

Usage:
    python curate.py [--session-dir DIR] [--output-dir DIR] [--min-actions N] [--split RATIO]
"""

import argparse
import json
import glob
import os
import sys
import hashlib
import random
from pathlib import Path
from typing import Optional

from shared import find_session_files

# -------------------------------------------------------------------
# Tool call mapping: maps raw bash commands to structured tool calls
# -------------------------------------------------------------------

TOOL_PATTERNS = [
    # (substring_match, tool_name, arg_extractor)
    ("gt polecat list",    "gt_polecat_list",    lambda cmd: extract_polecat_list_args(cmd)),
    ("gt polecat nuke",    "gt_polecat_nuke",    lambda cmd: extract_polecat_nuke_args(cmd)),
    ("gt peek",            "gt_peek",            lambda cmd: extract_peek_args(cmd)),
    ("gt session status",  "gt_session_status",  lambda cmd: extract_session_status_args(cmd)),
    ("gt nudge",           "gt_nudge",           lambda cmd: extract_nudge_args(cmd)),
    ("gt mail inbox",      "gt_mail_inbox",      lambda _: {}),
    ("gt mail read",       "gt_mail_read",       lambda cmd: extract_mail_read_args(cmd)),
    ("gt mail send",       "gt_mail_send",       lambda cmd: extract_mail_send_args(cmd)),
    ("gt patrol report",   "gt_patrol_report",   lambda cmd: extract_patrol_report_args(cmd)),
    ("gt patrol new",      "gt_patrol_report",   lambda _: {"summary": "New patrol cycle"}),
    ("gt handoff",         "gt_handoff",         lambda cmd: extract_handoff_args(cmd)),
    ("gt escalate",        "gt_escalate",        lambda cmd: extract_escalate_args(cmd)),
    ("bd show",            "bd_show",            lambda cmd: extract_bd_show_args(cmd)),
    ("bd list",            "bd_list",            lambda cmd: extract_bd_list_args(cmd)),
    ("bd close",           "bd_close",           lambda cmd: extract_bd_close_args(cmd)),
    ("bd children",        "bd_children",        lambda cmd: extract_bd_children_args(cmd)),
    ("bd gate check",      "bd_list",            lambda _: {"type": "timer", "status": "open"}),
    ("bd mol squash",      "bd_close",           lambda cmd: extract_first_id_arg(cmd)),
    ("bd mol wisp",        "gt_patrol_report",   lambda _: {"summary": "Create new patrol wisp"}),
    ("bd mol current",     "bd_list",            lambda _: {"type": "molecule", "status": "in_progress"}),
    ("tmux has-session",   "check_tmux_session", lambda cmd: extract_tmux_args(cmd)),
    ("git status",         "check_git_state",    lambda cmd: extract_git_state_args(cmd)),
    ("git log",            "check_git_state",    lambda cmd: extract_git_state_args(cmd)),
]


def extract_polecat_list_args(cmd):
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p == "list" and i + 1 < len(parts):
            return {"rig": parts[i + 1]}
    return {"rig": "gastown"}


def extract_polecat_nuke_args(cmd):
    force = "--force" in cmd
    parts = cmd.replace("--force", "").split()
    for i, p in enumerate(parts):
        if p == "nuke" and i + 1 < len(parts):
            return {"target": parts[i + 1], "force": force}
    return {"target": "unknown", "force": force}


def extract_peek_args(cmd):
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p == "peek" or (p == "gt" and i + 1 < len(parts) and parts[i + 1] == "peek"):
            continue
        if "/" in p:
            lines = 30
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                lines = int(parts[i + 1])
            return {"target": p, "lines": lines}
    return {"target": "unknown", "lines": 30}


def extract_session_status_args(cmd):
    parts = cmd.split()
    for p in parts:
        if "/" in p and p not in ("gt", "session", "status"):
            return {"target": p}
    return {"target": "unknown"}


def extract_nudge_args(cmd):
    # gt nudge gastown/furiosa "message here"
    parts = cmd.split(None, 3)
    if len(parts) >= 4:
        return {"target": parts[2], "message": parts[3].strip('"').strip("'")}
    elif len(parts) >= 3:
        return {"target": parts[2], "message": ""}
    return {"target": "unknown", "message": ""}


def extract_mail_read_args(cmd):
    parts = cmd.split()
    for p in parts:
        if p.startswith("hq-") or (len(p) > 3 and "-" in p and p not in ("gt", "mail", "read")):
            return {"mail_id": p}
    return {"mail_id": "unknown"}


def extract_mail_send_args(cmd):
    parts = cmd.split()
    recipient = ""
    subject = ""
    body = ""
    for i, p in enumerate(parts):
        if p == "send" and i + 1 < len(parts):
            recipient = parts[i + 1]
        if p == "-s" and i + 1 < len(parts):
            # Collect subject (may be quoted)
            subject = _extract_quoted(cmd, cmd.index("-s") + 3)
        if p == "-m" and i + 1 < len(parts):
            body = _extract_quoted(cmd, cmd.index("-m") + 3)
    return {"recipient": recipient, "subject": subject, "body": body}


def extract_patrol_report_args(cmd):
    summary = ""
    if "--summary" in cmd:
        idx = cmd.index("--summary")
        summary = _extract_quoted(cmd, idx + 10)
    return {"summary": summary}


def extract_handoff_args(cmd):
    subject = ""
    message = ""
    if "-s" in cmd:
        idx = cmd.index("-s")
        subject = _extract_quoted(cmd, idx + 3)
    if "-m" in cmd:
        idx = cmd.index("-m")
        message = _extract_quoted(cmd, idx + 3)
    return {"subject": subject, "message": message}


def extract_escalate_args(cmd):
    severity = "HIGH"
    message = ""
    if "-s" in cmd:
        parts = cmd.split()
        for i, p in enumerate(parts):
            if p == "-s" and i + 1 < len(parts):
                severity = parts[i + 1]
    # Message is the rest
    if '"' in cmd:
        message = _extract_quoted(cmd, cmd.index('"'))
    return {"severity": severity, "message": message}


def extract_bd_show_args(cmd):
    parts = cmd.split()
    for p in parts:
        if p.startswith("gt-") or p.startswith("hq-"):
            return {"bead_id": p.rstrip(",")}
    return {"bead_id": "unknown"}


def extract_bd_list_args(cmd):
    result = {}
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p.startswith("--status="):
            result["status"] = p.split("=", 1)[1]
        elif p.startswith("--type="):
            result["type"] = p.split("=", 1)[1]
        elif p.startswith("--label"):
            if "=" in p:
                result["label"] = p.split("=", 1)[1]
            elif i + 1 < len(parts):
                result["label"] = parts[i + 1]
        elif p.startswith("--assignee"):
            if "=" in p:
                result["assignee"] = p.split("=", 1)[1]
            elif i + 1 < len(parts):
                result["assignee"] = parts[i + 1]
    return result


def extract_bd_close_args(cmd):
    return extract_first_id_arg(cmd)


def extract_bd_children_args(cmd):
    return extract_first_id_arg(cmd)


def extract_first_id_arg(cmd):
    parts = cmd.split()
    for p in parts:
        if p.startswith("gt-") or p.startswith("hq-"):
            return {"bead_id": p.rstrip(",")}
    return {"bead_id": "unknown"}


def extract_tmux_args(cmd):
    sessions = []
    # Parse "tmux has-session -t name && ... || ..." chains
    parts = cmd.split("&&")
    for part in parts:
        part = part.strip()
        subparts = part.split("||")
        for sub in subparts:
            sub = sub.strip()
            if "has-session" in sub and "-t" in sub:
                tokens = sub.split()
                for i, t in enumerate(tokens):
                    if t == "-t" and i + 1 < len(tokens):
                        sessions.append(tokens[i + 1])
    return {"sessions": sessions if sessions else ["unknown"]}


def extract_git_state_args(cmd):
    # Try to find the path
    if "cd " in cmd:
        path_start = cmd.index("cd ") + 3
        path = cmd[path_start:].split("&&")[0].strip()
        return {"polecat_path": path}
    return {"polecat_path": "unknown"}


def _extract_quoted(text, start):
    """Extract a quoted or unquoted string starting at position start."""
    text = text[start:].strip()
    if text.startswith('"'):
        end = text.find('"', 1)
        if end > 0:
            return text[1:end]
    if text.startswith("'"):
        end = text.find("'", 1)
        if end > 0:
            return text[1:end]
    # Unquoted — take until next flag or end
    return text.split(" -")[0].strip().strip('"').strip("'")


def classify_bash_command(cmd: str) -> Optional[tuple]:
    """Map a bash command to a structured tool call. Returns (tool_name, args) or None."""
    for pattern, tool_name, extractor in TOOL_PATTERNS:
        if pattern in cmd:
            try:
                args = extractor(cmd)
                return (tool_name, args)
            except Exception:
                return (tool_name, {})
    return None


# -------------------------------------------------------------------
# Session parsing
# -------------------------------------------------------------------

def load_session(path: str) -> list:
    """Load a JSONL session file into a list of records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def is_boilerplate_user_message(content: str) -> bool:
    """Check if a user message is startup boilerplate (not real conversation)."""
    boilerplate_markers = [
        "Run `gt prime",
        "gt prime --hook",
        "begin patrol",
        "[from deacon] HEALTH_CHECK",
    ]
    for marker in boilerplate_markers:
        if marker in content:
            return True
    return False


def extract_conversations(records: list) -> list:
    """
    Extract clean multi-turn conversations from raw session records.
    Returns a list of conversation dicts with 'messages' key.
    """
    messages = []

    for r in records:
        rtype = r.get("type")

        # Skip non-conversation records
        if rtype in ("file-history-snapshot", "progress", "system"):
            continue

        if rtype == "user":
            msg = r.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                # Skip startup boilerplate but keep it as context indicator
                if is_boilerplate_user_message(content):
                    # Still include it — it's the "situation" the model needs to respond to
                    messages.append({
                        "role": "user",
                        "content": _truncate_system_context(content)
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": content
                    })

            elif isinstance(content, list):
                # Tool results come as list items in user messages
                for item in content:
                    if item.get("type") == "tool_result":
                        tool_id = item.get("tool_use_id", "")
                        result_content = item.get("content", "")
                        if isinstance(result_content, list):
                            result_content = "\n".join(
                                c.get("text", "") for c in result_content if c.get("type") == "text"
                            )
                        is_error = item.get("is_error", False)

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "content": str(result_content)[:2000],  # Truncate verbose output
                            "is_error": is_error
                        })
                    elif item.get("type") == "text":
                        txt = item.get("text", "").strip()
                        if txt:
                            messages.append({
                                "role": "user",
                                "content": txt
                            })

        elif rtype == "assistant":
            msg = r.get("message", {})
            content = msg.get("content", [])

            if not isinstance(content, list):
                continue

            # Collect all parts of this assistant turn
            text_parts = []
            tool_calls = []

            for item in content:
                itype = item.get("type")

                if itype == "thinking":
                    # Skip thinking blocks for v1 (input→action pairs only)
                    pass

                elif itype == "text":
                    text = item.get("text", "").strip()
                    if text:
                        text_parts.append(text)

                elif itype == "tool_use":
                    tool_name = item.get("name", "")
                    tool_input = item.get("input", {})
                    tool_id = item.get("id", "")

                    if tool_name == "Bash":
                        # Map bash commands to structured tool calls
                        cmd = tool_input.get("command", "")
                        mapped = classify_bash_command(cmd)
                        if mapped:
                            fn_name, fn_args = mapped
                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": fn_name,
                                    "arguments": json.dumps(fn_args)
                                }
                            })
                        else:
                            # Unmapped bash command — skip (cat, ls, etc.)
                            tool_calls.append({
                                "id": tool_id,
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": json.dumps({"command": cmd[:500]})
                                }
                            })

                    elif tool_name in ("Read", "Glob", "Grep", "Write", "Edit"):
                        # File operations — less relevant for witness but keep for completeness
                        tool_calls.append({
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name.lower(),
                                "arguments": json.dumps(tool_input)
                            }
                        })

            # Build the assistant message
            if tool_calls and text_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts),
                    "tool_calls": tool_calls
                })
            elif tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                })
            elif text_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(text_parts)
                })

    return messages


def _truncate_system_context(text: str, max_len: int = 500) -> str:
    """Truncate long system context injections, keeping the actionable part."""
    if len(text) <= max_len:
        return text
    # Keep the first line (usually the instruction) and truncate
    lines = text.split("\n")
    result = lines[0]
    for line in lines[1:]:
        if len(result) + len(line) + 1 > max_len:
            break
        result += "\n" + line
    return result


# -------------------------------------------------------------------
# Quality scoring
# -------------------------------------------------------------------

def score_session(messages: list) -> dict:
    """Score a curated conversation for training quality."""
    scores = {
        "total_messages": len(messages),
        "user_messages": sum(1 for m in messages if m["role"] == "user"),
        "assistant_messages": sum(1 for m in messages if m["role"] == "assistant"),
        "tool_results": sum(1 for m in messages if m["role"] == "tool"),
        "tool_calls": 0,
        "witness_tool_calls": 0,
        "decision_actions": set(),
        "quality_score": 0.0,
    }

    witness_tools = {
        "gt_polecat_list", "gt_polecat_nuke", "gt_peek", "gt_session_status",
        "gt_nudge", "gt_mail_inbox", "gt_mail_read", "gt_mail_send",
        "gt_patrol_report", "gt_handoff", "gt_escalate",
        "bd_show", "bd_list", "bd_close", "bd_children",
        "check_git_state", "check_tmux_session",
    }

    action_tools = {
        "gt_polecat_nuke", "gt_nudge", "gt_mail_send", "gt_escalate",
        "gt_handoff", "gt_patrol_report",
    }

    for m in messages:
        if m["role"] == "assistant" and "tool_calls" in m:
            for tc in m["tool_calls"]:
                scores["tool_calls"] += 1
                fn_name = tc.get("function", {}).get("name", "")
                if fn_name in witness_tools:
                    scores["witness_tool_calls"] += 1
                if fn_name in action_tools:
                    scores["decision_actions"].add(fn_name)

    # Quality heuristic:
    # - More witness tool calls = better
    # - More diverse actions = better
    # - Minimum viable conversation length
    n_actions = len(scores["decision_actions"])
    scores["decision_actions"] = list(scores["decision_actions"])

    if scores["total_messages"] < 4:
        scores["quality_score"] = 0.0
    else:
        scores["quality_score"] = (
            min(scores["witness_tool_calls"] / 5, 1.0) * 0.4 +
            min(n_actions / 3, 1.0) * 0.4 +
            min(scores["total_messages"] / 20, 1.0) * 0.2
        )

    return scores


# -------------------------------------------------------------------
# Output formatters
# -------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Witness agent monitoring worker agents (polecats) in the Gas Town multi-agent workspace.

Your responsibilities:
1. HEALTH MONITORING: Check if polecats are stuck, crashed, or healthy
2. ESCALATION: Nudge stuck polecats, escalate to Mayor after repeated failures
3. PRE-KILL VERIFICATION: Check git state before nuking polecats
4. CLEANUP: Send MERGE_READY to refinery, then nuke completed polecats
5. INFRASTRUCTURE: Verify deacon and refinery sessions are alive

Decision rules:
- If a polecat is idle >30min with no progress → nudge it
- If 3+ nudges with no response → escalate to Mayor
- If polecat sent POLECAT_DONE and git is clean → MERGE_READY then nuke
- If polecat has unpushed work → escalate for recovery, do NOT nuke
- If polecat is in crash loop (3+ restarts) → force nuke and escalate
- If all infrastructure is healthy and no polecats active → report idle patrol
- After extraordinary actions (force nuke, escalation) → hand off to fresh session"""


def to_nanochat_format(messages: list) -> list:
    """Convert to nanochat CustomJSON format: list of {role, content} dicts."""
    result = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in messages:
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})

        elif role == "assistant":
            content = m.get("content", "") or ""
            if "tool_calls" in m:
                # Encode tool calls inline (nanochat style)
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    call_str = f'<tool_call>{fn["name"]}({fn["arguments"]})</tool_call>'
                    content = (content + "\n" + call_str).strip()
            if content:
                result.append({"role": "assistant", "content": content})

        elif role == "tool":
            # Tool results as user messages in nanochat format
            tool_content = m.get("content", "")
            if m.get("is_error"):
                tool_content = f"ERROR: {tool_content}"
            result.append({
                "role": "user",
                "content": f'<tool_result>{tool_content}</tool_result>'
            })

    return result


def to_openai_format(messages: list) -> dict:
    """Convert to OpenAI ChatML format for fine-tuning."""
    result = [{"role": "system", "content": SYSTEM_PROMPT}]

    for m in messages:
        role = m["role"]

        if role == "user":
            result.append({"role": "user", "content": m["content"]})

        elif role == "assistant":
            msg = {"role": "assistant"}
            if m.get("content"):
                msg["content"] = m["content"]
            else:
                msg["content"] = None
            if "tool_calls" in m:
                msg["tool_calls"] = m["tool_calls"]
            result.append(msg)

        elif role == "tool":
            result.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id", ""),
                "content": m.get("content", "")
            })

    return {"messages": result}


# -------------------------------------------------------------------
# Main pipeline
# -------------------------------------------------------------------



def process_session(path: str) -> Optional[dict]:
    """Process a single session file into curated training data."""
    try:
        records = load_session(path)
        messages = extract_conversations(records)
        scores = score_session(messages)

        if scores["quality_score"] < 0.1:
            return None

        session_id = os.path.basename(path).replace(".jsonl", "")

        return {
            "session_id": session_id,
            "source_path": path,
            "messages": messages,
            "scores": scores,
            "nanochat": to_nanochat_format(messages),
            "openai": to_openai_format(messages),
        }
    except Exception as e:
        print(f"  ERROR processing {path}: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(description="Curate witness session data for fine-tuning")
    parser.add_argument("--session-dir", action="append", default=None,
                        help="Session directory (can specify multiple)")
    parser.add_argument("--output-dir", default="./dataset",
                        help="Output directory for curated data")
    parser.add_argument("--min-quality", type=float, default=0.2,
                        help="Minimum quality score to include (0.0-1.0)")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train/eval split ratio")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for split")
    parser.add_argument("--max-sessions", type=int, default=0,
                        help="Max sessions to process (0=all)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Only print stats, don't write output")
    args = parser.parse_args()

    # Default session directories: all witness sessions across all rigs
    if args.session_dir is None:
        claude_projects = os.path.expanduser("~/.claude/projects/")
        args.session_dir = glob.glob(os.path.join(claude_projects, "*-witness"))

    print(f"Scanning {len(args.session_dir)} session directories...")
    files = find_session_files(args.session_dir)
    print(f"Found {len(files)} session files in size range")

    if args.max_sessions:
        files = files[:args.max_sessions]

    # Process all sessions
    curated = []
    skipped = 0
    for i, path in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(files)}...")
        result = process_session(path)
        if result and result["scores"]["quality_score"] >= args.min_quality:
            curated.append(result)
        else:
            skipped += 1

    print(f"\nResults:")
    print(f"  Processed: {len(files)}")
    print(f"  Kept:      {len(curated)}")
    print(f"  Skipped:   {skipped} (below quality threshold {args.min_quality})")

    if not curated:
        print("No sessions passed quality filter!")
        return

    # Print quality distribution
    scores = [c["scores"]["quality_score"] for c in curated]
    scores.sort(reverse=True)
    print(f"\nQuality distribution:")
    print(f"  Top:    {scores[0]:.3f}")
    print(f"  Median: {scores[len(scores)//2]:.3f}")
    print(f"  Bottom: {scores[-1]:.3f}")

    # Action distribution
    all_actions = {}
    for c in curated:
        for a in c["scores"]["decision_actions"]:
            all_actions[a] = all_actions.get(a, 0) + 1
    print(f"\nAction distribution across {len(curated)} sessions:")
    for action, count in sorted(all_actions.items(), key=lambda x: -x[1]):
        print(f"  {action:30s} {count:4d} sessions")

    # Message stats
    total_msgs = sum(c["scores"]["total_messages"] for c in curated)
    total_tool_calls = sum(c["scores"]["witness_tool_calls"] for c in curated)
    print(f"\nTotal messages: {total_msgs}")
    print(f"Total witness tool calls: {total_tool_calls}")
    print(f"Avg messages/session: {total_msgs / len(curated):.1f}")
    print(f"Avg tool calls/session: {total_tool_calls / len(curated):.1f}")

    if args.stats_only:
        return

    # Split train/eval
    random.seed(args.seed)
    random.shuffle(curated)
    split_idx = int(len(curated) * args.split)
    train = curated[:split_idx]
    eval_set = curated[split_idx:]

    print(f"\nSplit: {len(train)} train, {len(eval_set)} eval")

    # Write output
    os.makedirs(args.output_dir, exist_ok=True)

    # nanochat CustomJSON format (one conversation per line)
    with open(os.path.join(args.output_dir, "train_nanochat.jsonl"), "w") as f:
        for c in train:
            f.write(json.dumps(c["nanochat"]) + "\n")

    with open(os.path.join(args.output_dir, "eval_nanochat.jsonl"), "w") as f:
        for c in eval_set:
            f.write(json.dumps(c["nanochat"]) + "\n")

    # OpenAI ChatML format
    with open(os.path.join(args.output_dir, "train_openai.jsonl"), "w") as f:
        for c in train:
            f.write(json.dumps(c["openai"]) + "\n")

    with open(os.path.join(args.output_dir, "eval_openai.jsonl"), "w") as f:
        for c in eval_set:
            f.write(json.dumps(c["openai"]) + "\n")

    # Metadata
    metadata = {
        "total_sessions_scanned": len(files),
        "sessions_kept": len(curated),
        "train_count": len(train),
        "eval_count": len(eval_set),
        "min_quality": args.min_quality,
        "split_ratio": args.split,
        "seed": args.seed,
        "total_messages": total_msgs,
        "total_witness_tool_calls": total_tool_calls,
        "action_distribution": all_actions,
        "session_dirs": [str(d) for d in args.session_dir],
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nOutput written to {args.output_dir}/")
    print(f"  train_nanochat.jsonl  ({len(train)} conversations)")
    print(f"  eval_nanochat.jsonl   ({len(eval_set)} conversations)")
    print(f"  train_openai.jsonl    ({len(train)} conversations)")
    print(f"  eval_openai.jsonl     ({len(eval_set)} conversations)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()
