#!/usr/bin/env python3
"""
Shared snapshot formatting module for rich-context witness patrol.

Defines section headers, per-section truncation limits, and the assembly
function. Imported by both curate_snapshots.py (training data extraction)
and serve.py (live inference context).

Key invariant: the snapshot format produced here is the ONLY format the model
sees — if curate and serve disagree, the model gets train/serve mismatch.
"""

# Section headers — order matters (model sees them in this order)
SECTIONS = [
    "Polecats",
    "Inbox",
    "Cleanup Wisps",
    "Infrastructure",
    "Active Work",
    "State",
]

# Per-section character limits (truncation for 2048 token budget)
# Total budget: ~6000 chars (~1500 tokens) for context, leaving ~500 tokens
# for system prompt and generation.
SECTION_LIMITS = {
    "Polecats": 800,
    "Inbox": 1200,
    "Cleanup Wisps": 400,
    "Infrastructure": 200,
    "Active Work": 600,
    "State": 200,
}

DEFAULT_LIMIT = 400


def truncate_section(content: str, limit: int) -> str:
    """Truncate section content to fit within character limit."""
    if len(content) <= limit:
        return content
    # Truncate at last newline before limit, add indicator
    cut = content[:limit]
    last_nl = cut.rfind("\n")
    if last_nl > limit // 2:
        cut = cut[:last_nl]
    return cut.rstrip() + "\n[... truncated]"


def format_snapshot(sections: dict) -> str:
    """
    Assemble a structured patrol snapshot from section data.

    Args:
        sections: dict mapping section names to content strings.
                  Missing sections are shown as "N/A".
                  Sections with value "[error] command timed out" are kept as-is.

    Returns:
        Formatted snapshot string with ## headers.
    """
    parts = []
    for name in SECTIONS:
        content = sections.get(name, "N/A")
        if not content or content.strip() == "":
            content = "N/A"

        limit = SECTION_LIMITS.get(name, DEFAULT_LIMIT)
        content = truncate_section(content.strip(), limit)

        parts.append(f"## {name}\n{content}")

    return "\n\n".join(parts)


# Classify commands as GATHER vs ACTION for curate_snapshots.py
GATHER_COMMANDS = [
    "gt polecat list",
    "gt mail inbox",
    "gt mail read",
    "bd list",
    "bd show",
    "bd children",
    "bd gate check",
    "bd mol current",
    "cat state.json",
    "tmux has-session",
    "tmux list-sessions",
    "gt session status",
    "gt status",
    "gt peek",
    "check_git_state",
    "git status",
    "git log",
    "gt hook",
]

ACTION_COMMANDS = [
    "gt nudge",
    "gt polecat nuke",
    "gt mail send",
    "gt escalate",
    "gt patrol report",
    "gt patrol new",
    "gt handoff",
    "bd close",
    "bd mol squash",
]


def is_gather_command(cmd: str) -> bool:
    """Check if a bash command is a context-gathering operation."""
    for pattern in GATHER_COMMANDS:
        if pattern in cmd:
            return True
    return False


def is_action_command(cmd: str) -> bool:
    """Check if a bash command is a decision/action operation."""
    for pattern in ACTION_COMMANDS:
        if pattern in cmd:
            return True
    return False


def classify_gather_section(cmd: str) -> str | None:
    """
    Map a gather command to the snapshot section it populates.
    Returns section name or None if unknown.
    """
    if "gt polecat list" in cmd or "gt peek" in cmd:
        return "Polecats"
    if "gt mail inbox" in cmd or "gt mail read" in cmd:
        return "Inbox"
    if "bd list" in cmd:
        if "cleanup" in cmd or "label=cleanup" in cmd:
            return "Cleanup Wisps"
        if "in_progress" in cmd or "status=in_progress" in cmd:
            return "Active Work"
        return "Active Work"
    if "bd show" in cmd or "bd children" in cmd or "bd mol" in cmd:
        return "Active Work"
    if "bd gate check" in cmd:
        return "Active Work"
    if "tmux has-session" in cmd or "tmux list-sessions" in cmd:
        return "Infrastructure"
    if "gt session status" in cmd or "gt status" in cmd:
        return "Infrastructure"
    if "cat state.json" in cmd:
        return "State"
    if "git status" in cmd or "git log" in cmd or "check_git_state" in cmd:
        return "Active Work"
    if "gt hook" in cmd:
        return "Infrastructure"
    return None
