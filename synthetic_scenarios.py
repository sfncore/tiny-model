#!/usr/bin/env python3
"""
Generate synthetic single-turn training examples for scenario-based decisions.

The model defaults to "none" on short contexts because 62% of single-turn
training examples are "none". This script creates targeted examples where
short inputs map to the correct action tools.

Each scenario template has variations to prevent overfitting to exact phrasing.
"""

import json
import random
import os

from snapshot_format import format_snapshot

SYSTEM_PROMPT = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""

RIGS = ["gastown", "bcc", "hq", "zfc"]
POLECATS = ["furiosa", "nux", "rust", "guzzle", "nitro", "chrome", "refinery"]
WISPS = ["hq-wisp-2v214", "bcc-wisp-x8k3m", "gt-kvo.6", "hq-wisp-ragxshg", "bcc-wisp-gb0j5u"]
BEADS = ["bcc-8rlwh", "gt-4tp", "hq-3mn2", "zfc-9qw1", "bcc-wisp-2fgi4p"]
BRANCHES = ["feature/fix-auth", "feature/add-tests", "integration/beads-ide", "fix/type-errors", "feature/command-palette"]


def make_idle_patrol():
    """Idle patrol → gt_patrol_report or none"""
    rig = random.choice(RIGS)
    crew = random.choice(["crew/zfc", "crew/bcc", "crew/hq"])
    n_old = random.choice([0, 1, 3, 5])
    templates = [
        "Polecats\n\nNo active polecats.",
        "Polecats\n\nNo polecats running.\n\nAll sessions idle.",
        f"Polecats\n\nNo active polecats.\n\nDeacon: alive\nRefinery: running\n{rig} rig quiet.",
        "Patrol check: no polecats active, infrastructure healthy.",
        f"Polecats\n\n(none active)\n\nInbox: 0 unread\n{rig} rig idle.",
        # Real CLI output formats from gt polecat list + gt mail inbox
        "Polecats\n\nNo polecats found.",
        f"Polecats\n\nNo polecats found.\n\nInbox\n\n📬 Inbox: gastown/{crew} (0 messages, 0 unread)\n  (no messages)",
        f"Polecats\n\nNo polecats found.\n\nInbox\n\n📬 Inbox: gastown/{crew} ({n_old} messages, 0 unread)\n  (no messages)",
        f"Polecats\n\nNo polecats found.\n\nInbox\n\n📬 Inbox: gastown/{crew} (1 messages, 0 unread)\n  (no messages)",
        # Variations with infrastructure healthy
        f"Polecats\n\nNo polecats found.\n\nInbox\n\n📬 Inbox: gastown/{crew} (0 messages, 0 unread)\n  (no messages)\n\nDeacon: alive\nRefinery: running",
    ]
    tools = [
        {"tool": "gt_patrol_report", "args": {"status": "idle", "note": "No active polecats. Rig quiet."}},
        {"tool": "none", "args": {}},
        {"tool": "gt_patrol_report", "args": {"status": "idle"}},
        {"tool": "none", "args": {}},
        {"tool": "none", "args": {}},
    ]
    return random.choice(templates), random.choice(tools)


def make_healthy_polecat():
    """Active healthy polecat → gt_peek or gt_session_status (NOT nudge)"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    bead = random.choice(BEADS)
    recent_mins = random.choice([1, 2, 3, 5])
    templates = [
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\nWorking on {bead}. Last activity: {recent_mins} minutes ago. Making progress.",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n  ○ {rig}/{random.choice(POLECATS)}  done\n    {random.choice(WISPS)}\n\nActive output detected. {polecat} is healthy.",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\nLast commit: {recent_mins}min ago. Context usage: 34%. Healthy.",
        f"Polecat status:\n  {rig}/{polecat}: active (working on {bead})\n  Session: {wisp}\n  Last activity: {recent_mins}m ago, producing output normally.",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\n{polecat} is actively working on {bead}. Recent output visible. No issues.",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\nHealthy. Making progress on {bead}. {recent_mins}min since last activity.",
        # Bare status — no negative signal means healthy (DO NOT nudge)
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n  ○ {rig}/{random.choice(POLECATS)}  done\n    {random.choice(WISPS)}",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}",
    ]
    tools = [
        {"tool": "gt_peek", "args": {"target": f"{rig}/{polecat}", "lines": 30}},
        {"tool": "gt_peek", "args": {"target": f"{rig}/{polecat}"}},
        {"tool": "gt_session_status", "args": {"session": wisp}},
        {"tool": "gt_polecat_list", "args": {"rig": rig}},
        {"tool": "gt_patrol_report", "args": {"status": "active", "note": f"{polecat} healthy and working on {bead}."}},
    ]
    return random.choice(templates), random.choice(tools)


def make_stuck_polecat():
    """Stuck polecat → gt_nudge"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    idle_mins = random.choice([15, 20, 30, 45, 60])
    templates = [
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\nLast activity: {idle_mins} minutes ago. No progress detected.",
        f"Polecats\n\n  ● {rig}/{polecat}  idle\n    {wisp}\n\nNo output for {idle_mins} minutes. May be stuck.",
        f"Polecat {rig}/{polecat} appears stuck.\nSession: {wisp}\nLast activity: {idle_mins}min ago\nNo new output or commits.",
        f"Polecats\n\n  ● {rig}/{polecat}  working\n    {wisp}\n\nStale for {idle_mins}m. Context window may be full.",
    ]
    tools = [
        {"tool": "gt_nudge", "args": {"target": f"{rig}/{polecat}", "message": "Are you still working? No progress detected."}},
        {"tool": "gt_nudge", "args": {"target": f"{rig}/{polecat}"}},
        {"tool": "gt_peek", "args": {"target": f"{rig}/{polecat}", "lines": 50}},
    ]
    # Weighted: nudge is the primary correct action
    weights = [3, 3, 1]
    return random.choice(templates), random.choices(tools, weights=weights, k=1)[0]


def make_completed_polecat():
    """Completed polecat with clean git → gt_polecat_nuke or check_git_state"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    branch = random.choice(BRANCHES)
    templates = [
        f"Polecats\n\n  ○ {rig}/{polecat}  done\n    {wisp}\n\nAll work pushed. Branch merged.",
        f"Polecats\n\n  ○ {rig}/{polecat}  done\n    {wisp}\n\nBranch {branch} merged to main. Git clean.",
        f"Polecat {rig}/{polecat} completed.\nSession: {wisp}\nGit state: clean, branch merged.\nReady for cleanup.",
        f"Polecats\n\n  ○ {rig}/{polecat}  done\n    {wisp}\n\nPOLECAT_DONE received. Branch {branch} merged. No uncommitted changes.",
    ]
    tools = [
        {"tool": "gt_polecat_nuke", "args": {"target": f"{rig}/{polecat}"}},
        {"tool": "check_git_state", "args": {"session": wisp}},
        {"tool": "gt_polecat_nuke", "args": {"target": f"{rig}/{polecat}", "force": False}},
    ]
    weights = [3, 2, 2]
    return random.choice(templates), random.choices(tools, weights=weights, k=1)[0]


def make_crash_loop():
    """Crash-looping polecat → gt_mail_send (escalate) or gt_polecat_nuke --force"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    restarts = random.choice([3, 4, 5, 6])
    crash_reason = random.choice(["segfault", "OOM", "context overflow", "API timeout", "module not found"])
    templates = [
        f"Polecats\n\n  ✗ {rig}/{polecat}  crashed\n    {wisp}\n\nRestart count: {restarts}. Last crash: {crash_reason}.",
        f"Polecat {rig}/{polecat} in crash loop.\nSession: {wisp}\nRestarts: {restarts}\nCrash: {crash_reason}\nNot recovering.",
        f"Polecats\n\n  ✗ {rig}/{polecat}  crashed\n    {wisp}\n\n{restarts} restarts in 30 minutes. Crash: {crash_reason}. Needs intervention.",
        f"ALERT: {rig}/{polecat} crash loop detected.\nRestarts: {restarts}\nLast error: {crash_reason}\nSession {wisp} unstable.",
    ]
    tools = [
        {"tool": "gt_mail_send", "args": {"recipient": "mayor/", "subject": f"ESCALATION: {rig}/{polecat} crash loop", "body": f"{polecat} has crashed {restarts} times ({crash_reason}). Needs intervention."}},
        {"tool": "gt_polecat_nuke", "args": {"target": f"{rig}/{polecat}", "force": True}},
        {"tool": "gt_mail_send", "args": {"recipient": "mayor/", "subject": f"ALERT: {rig}/{polecat} crash loop ({restarts} restarts)", "body": f"Crash reason: {crash_reason}. Escalating for intervention."}},
    ]
    weights = [3, 2, 2]
    return random.choice(templates), random.choices(tools, weights=weights, k=1)[0]


def make_unpushed_work():
    """Completed but unpushed work → gt_mail_send (escalate) or check_git_state"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    branch = random.choice(BRANCHES)
    n_commits = random.choice([1, 2, 3, 5])
    templates = [
        f"Polecats\n\n  ○ {rig}/{polecat}  done\n    {wisp}\n\nGit state: {n_commits} unpushed commits on branch {branch}.",
        f"Polecat {rig}/{polecat} completed but has unpushed work.\nSession: {wisp}\nBranch: {branch}\n{n_commits} local commits not pushed.",
        f"Polecats\n\n  ○ {rig}/{polecat}  done\n    {wisp}\n\nWARNING: {n_commits} unpushed commits on {branch}. DO NOT nuke.",
        f"ALERT: {rig}/{polecat} done but unpushed.\n{n_commits} commits on {branch} not pushed to remote.\nSession: {wisp}\nRecovery needed before cleanup.",
    ]
    tools = [
        {"tool": "gt_mail_send", "args": {"recipient": "mayor/", "subject": f"RECOVERY_NEEDED: {rig}/{polecat} has unpushed work", "body": f"{n_commits} unpushed commits on {branch}. DO NOT nuke."}},
        {"tool": "check_git_state", "args": {"session": wisp}},
        {"tool": "gt_mail_send", "args": {"recipient": "mayor/", "subject": f"ESCALATION: unpushed work on {rig}/{polecat}", "body": f"Branch {branch} has {n_commits} unpushed commits. Needs recovery."}},
    ]
    weights = [3, 2, 2]
    return random.choice(templates), random.choices(tools, weights=weights, k=1)[0]


def make_check_infrastructure():
    """Infrastructure check → check_tmux_session or gt_session_status"""
    rig = random.choice(RIGS)
    templates = [
        f"Infrastructure check requested.\n\nDeacon: unknown\nRefinery: unknown",
        f"Patrol: need to verify {rig} infrastructure.\nDeacon status: unknown\nRefinery status: unknown",
        f"Check deacon and refinery health for {rig}.",
        f"Infrastructure status unknown.\nDeacon: ?\nRefinery: ?\nNeed to verify sessions are alive.",
    ]
    tools = [
        {"tool": "check_tmux_session", "args": {"session": "deacon"}},
        {"tool": "gt_session_status", "args": {}},
        {"tool": "check_tmux_session", "args": {"session": "refinery"}},
        {"tool": "bash", "args": {"command": "tmux list-sessions 2>/dev/null"}},
    ]
    return random.choice(templates), random.choice(tools)


def make_mail_check():
    """Mail check → gt_mail_inbox"""
    n_unread = random.choice([1, 3, 5, 12, 27])
    templates = [
        "Check inbox for new messages.",
        f"📬 Inbox: {n_unread} unread messages.",
        "Patrol step: check mail for handoffs or escalations.",
        f"Mail check. {n_unread} unread in witness inbox.",
        "Check inbox before next patrol cycle.",
    ]
    tools = [
        {"tool": "gt_mail_inbox", "args": {}},
        {"tool": "gt_mail_read", "args": {"mail_id": "1"}},
    ]
    weights = [4, 1]
    return random.choice(templates), random.choices(tools, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Rich-context snapshot scenarios (P6.3)
# ---------------------------------------------------------------------------

MAIL_IDS = ["hq-m3k9x", "bcc-m7j2p", "gt-m1r4v", "zfc-m8n6q", "hq-m5t2w"]
AGENTS = ["gastown/witness", "hq/mayor", "bcc/foreman", "zfc/witness"]


def _rich_snapshot(**overrides) -> str:
    """Build a rich snapshot with defaults and optional overrides."""
    rig = overrides.pop("_rig", random.choice(RIGS))
    defaults = {
        "Polecats": "No polecats found.",
        "Inbox": "No unread messages.",
        "Cleanup Wisps": "None",
        "Infrastructure": "Deacon: alive\nRefinery: running",
        "Active Work": "None",
        "State": f"patrol_count: {random.randint(1, 50)}, idle_cycles: {random.randint(0, 5)}, last_action: none",
    }
    defaults.update(overrides)
    return format_snapshot(defaults)


def _maybe_timeout(content: str) -> str:
    """Randomly replace content with timeout error (10% chance)."""
    if random.random() < 0.1:
        return "[error] command timed out"
    return content


def make_rich_polecat_done_inbox():
    """POLECAT_DONE mail in inbox + agent bead → gt_mail_send (MERGE_READY)"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    bead = random.choice(BEADS)
    mail_id = random.choice(MAIL_IDS)
    branch = random.choice(BRANCHES)

    inbox = (
        f"1 unread\n"
        f"  {mail_id}  from:{rig}/{polecat}  POLECAT_DONE\n"
        f"--- {mail_id} ---\n"
        f"All work on {bead} complete. Branch {branch} pushed. Git clean."
    )
    polecats = f"  ○ {rig}/{polecat}  done\n    {random.choice(WISPS)}"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=polecats,
        Inbox=_maybe_timeout(inbox),
        **{"Active Work": f"{bead} (in_progress)"},
    )
    tool = {"tool": "gt_mail_send", "args": {
        "recipient": f"{rig}/refinery",
        "subject": f"MERGE_READY: {bead}",
        "body": f"Polecat {polecat} completed {bead}. Branch {branch} pushed and clean."
    }}
    return snapshot, tool


def make_rich_lifecycle_shutdown():
    """LIFECYCLE:Shutdown mail → check_git_state or gt_polecat_nuke"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    mail_id = random.choice(MAIL_IDS)

    inbox = (
        f"1 unread\n"
        f"  {mail_id}  from:system  LIFECYCLE:Shutdown\n"
        f"--- {mail_id} ---\n"
        f"Agent {rig}/{polecat} shutting down. Session {wisp} ending."
    )
    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=_maybe_timeout(polecats),
        Inbox=inbox,
    )
    tools = [
        {"tool": "check_git_state", "args": {"session": wisp}},
        {"tool": "gt_polecat_nuke", "args": {"target": f"{rig}/{polecat}"}},
    ]
    return snapshot, random.choice(tools)


def make_rich_merged_notification():
    """MERGED notification + cleanup wisp → bd_close"""
    rig = random.choice(RIGS)
    bead = random.choice(BEADS)
    wisp = random.choice(WISPS)
    mail_id = random.choice(MAIL_IDS)
    branch = random.choice(BRANCHES)

    inbox = (
        f"1 unread\n"
        f"  {mail_id}  from:{rig}/refinery  MERGED\n"
        f"--- {mail_id} ---\n"
        f"Branch {branch} merged to main. Bead {bead} work complete."
    )
    cleanup = f"  {wisp}  cleanup  {bead} (merge cleanup)"

    snapshot = _rich_snapshot(
        _rig=rig,
        Inbox=inbox,
        **{"Cleanup Wisps": _maybe_timeout(cleanup)},
    )
    tool = {"tool": "bd_close", "args": {"bead_id": wisp}}
    return snapshot, tool


def make_rich_help_request():
    """HELP request in inbox from polecat → gt_mail_read or gt_mail_send"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    bead = random.choice(BEADS)
    mail_id = random.choice(MAIL_IDS)

    help_reasons = [
        "Need access to production database credentials",
        "Build failing on dependency resolution",
        "Cannot push to remote — permission denied",
        "Tests timing out, need guidance on timeout config",
        "Merge conflict on main that I cannot resolve",
    ]
    reason = random.choice(help_reasons)

    inbox = (
        f"1 unread\n"
        f"  {mail_id}  from:{rig}/{polecat}  HELP: {reason[:40]}\n"
        f"--- {mail_id} ---\n"
        f"Working on {bead}. {reason}. Requesting witness assistance."
    )
    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=polecats,
        Inbox=inbox,
    )
    tools = [
        {"tool": "gt_mail_send", "args": {
            "recipient": "mayor/",
            "subject": f"HELP from {rig}/{polecat}: {reason[:30]}",
            "body": f"Polecat {polecat} needs help: {reason}. Working on {bead}."
        }},
        {"tool": "gt_mail_read", "args": {"mail_id": mail_id}},
    ]
    return snapshot, random.choices(tools, weights=[3, 1], k=1)[0]


def make_rich_zombie_polecat():
    """Running agent_state but dead session → gt_mail_send (RECOVERY)"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    bead = random.choice(BEADS)

    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}\n\nSession {wisp} not found in tmux. Agent state shows running."
    active = f"{bead} (in_progress, assigned to {rig}/{polecat})"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=_maybe_timeout(polecats),
        Infrastructure="Deacon: alive\nRefinery: running",
        **{"Active Work": active},
    )
    tool = {"tool": "gt_mail_send", "args": {
        "recipient": "mayor/",
        "subject": f"RECOVERY: zombie polecat {rig}/{polecat}",
        "body": f"Agent {polecat} shows running but session {wisp} is dead. Bead {bead} needs reassignment."
    }}
    return snapshot, tool


def make_rich_stale_spawn():
    """Polecat in spawning state > 10 min → gt_escalate"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    mins = random.choice([12, 15, 20, 30])

    polecats = f"  ◐ {rig}/{polecat}  spawning\n    {wisp}\n\nSpawning for {mins} minutes. No activity detected."

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=polecats,
    )
    tool = {"tool": "gt_escalate", "args": {
        "severity": "HIGH",
        "message": f"{rig}/{polecat} stuck spawning for {mins}min. Session {wisp} may need restart."
    }}
    return snapshot, tool


def make_rich_orphaned_bead():
    """In-progress bead with no matching polecat → gt_mail_send"""
    rig = random.choice(RIGS)
    bead = random.choice(BEADS)
    branch = random.choice(BRANCHES)

    active = f"{bead} (in_progress, branch {branch})\n  No assigned polecat found."

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats="No polecats found.",
        **{"Active Work": _maybe_timeout(active)},
    )
    tool = {"tool": "gt_mail_send", "args": {
        "recipient": "mayor/",
        "subject": f"ORPHANED: bead {bead} has no polecat",
        "body": f"Bead {bead} is in_progress on branch {branch} but no polecat is assigned. Needs reassignment."
    }}
    return snapshot, tool


def make_rich_infra_down():
    """Both deacon and refinery down → gt_mail_send (escalate)"""
    rig = random.choice(RIGS)

    infra = "Deacon: dead\nRefinery: [error] session not found"

    snapshot = _rich_snapshot(
        _rig=rig,
        Infrastructure=infra,
    )
    tool = {"tool": "gt_mail_send", "args": {
        "recipient": "mayor/",
        "subject": f"ESCALATION: {rig} infrastructure down",
        "body": f"Both deacon and refinery are down on {rig}. Immediate attention required."
    }}
    return snapshot, tool


def make_rich_combined_stuck_help_dead():
    """Combined: stuck polecat + HELP mail + dead deacon → gt_mail_send (escalate)"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    mail_id = random.choice(MAIL_IDS)

    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}\n\nLast activity: 40 minutes ago. No progress detected."
    inbox = (
        f"1 unread\n"
        f"  {mail_id}  from:{rig}/{polecat}  HELP: stuck on merge conflict\n"
        f"--- {mail_id} ---\n"
        f"I'm stuck on a merge conflict and cannot proceed."
    )
    infra = "Deacon: dead\nRefinery: running"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=polecats,
        Inbox=inbox,
        Infrastructure=infra,
    )
    tool = {"tool": "gt_mail_send", "args": {
        "recipient": "mayor/",
        "subject": f"ESCALATION: {rig}/{polecat} stuck + deacon down",
        "body": f"Polecat {polecat} stuck 40min with merge conflict. Deacon is also dead. Multiple issues on {rig}."
    }}
    return snapshot, tool


def make_rich_idle_patrol():
    """Rich-format idle patrol → none or gt_patrol_report"""
    rig = random.choice(RIGS)
    idle_cycles = random.choice([0, 1, 3, 5, 8])

    snapshot = _rich_snapshot(
        _rig=rig,
        State=f"patrol_count: {random.randint(5, 50)}, idle_cycles: {idle_cycles}, last_action: none",
    )
    tools = [
        {"tool": "none", "args": {}},
        {"tool": "gt_patrol_report", "args": {"status": "idle", "note": f"Rig {rig} quiet. No polecats active."}},
        {"tool": "none", "args": {}},
    ]
    return snapshot, random.choice(tools)


def make_rich_healthy_polecat():
    """Rich-format healthy polecat → gt_peek or none"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    bead = random.choice(BEADS)
    recent_mins = random.choice([1, 2, 3, 5])

    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}\n\nLast activity: {recent_mins} minutes ago. Making progress on {bead}."
    active = f"{bead} (in_progress, assigned to {rig}/{polecat})"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=_maybe_timeout(polecats),
        **{"Active Work": active},
    )
    tools = [
        {"tool": "gt_peek", "args": {"target": f"{rig}/{polecat}", "lines": 30}},
        {"tool": "none", "args": {}},
        {"tool": "gt_patrol_report", "args": {"status": "active", "note": f"{polecat} healthy."}},
    ]
    return snapshot, random.choice(tools)


def make_rich_stuck_polecat():
    """Rich-format stuck polecat → gt_nudge"""
    rig = random.choice(RIGS)
    polecat = random.choice(POLECATS)
    wisp = random.choice(WISPS)
    bead = random.choice(BEADS)
    idle_mins = random.choice([15, 20, 30, 45])

    polecats = f"  ● {rig}/{polecat}  working\n    {wisp}\n\nLast activity: {idle_mins} minutes ago. No progress detected."
    active = f"{bead} (in_progress, assigned to {rig}/{polecat})"

    snapshot = _rich_snapshot(
        _rig=rig,
        Polecats=polecats,
        **{"Active Work": active},
    )
    tool = {"tool": "gt_nudge", "args": {
        "target": f"{rig}/{polecat}",
        "message": "Are you still working? No progress detected."
    }}
    return snapshot, tool


SCENARIO_GENERATORS = [
    (make_idle_patrol, 3.0),          # upweight — must learn "quiet rig → none"
    (make_healthy_polecat, 3.0),    # upweight — must distinguish from stuck
    (make_stuck_polecat, 2.0),      # upweight rare but important
    (make_completed_polecat, 2.0),   # upweight
    (make_crash_loop, 2.5),          # upweight — hardest scenario
    (make_unpushed_work, 2.5),       # upweight — hardest scenario
    (make_check_infrastructure, 1.5),
    (make_mail_check, 1.0),
]

# Rich-context snapshot scenarios (P6.3)
RICH_SCENARIO_GENERATORS = [
    (make_rich_idle_patrol, 3.0),
    (make_rich_healthy_polecat, 3.0),
    (make_rich_stuck_polecat, 2.0),
    (make_rich_polecat_done_inbox, 2.5),
    (make_rich_lifecycle_shutdown, 2.0),
    (make_rich_merged_notification, 2.0),
    (make_rich_help_request, 2.5),
    (make_rich_zombie_polecat, 2.5),
    (make_rich_stale_spawn, 2.0),
    (make_rich_orphaned_bead, 2.0),
    (make_rich_infra_down, 2.5),
    (make_rich_combined_stuck_help_dead, 2.0),
]


def generate_examples(n: int = 200, seed: int = 42,
                      fmt: str = "legacy") -> list:
    """Generate n synthetic scenario examples.

    Args:
        n: Number of examples to generate.
        seed: Random seed.
        fmt: "legacy" for original format, "rich" for snapshot format,
             "both" for a mix of both.
    """
    rng = random.Random(seed)
    random.seed(seed)

    if fmt == "rich":
        gen_list = RICH_SCENARIO_GENERATORS
    elif fmt == "both":
        gen_list = SCENARIO_GENERATORS + RICH_SCENARIO_GENERATORS
    else:
        gen_list = SCENARIO_GENERATORS

    generators, weights = zip(*gen_list)
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]

    examples = []
    for _ in range(n):
        gen = rng.choices(generators, weights=probs, k=1)[0]
        user_msg, tool_call = gen()

        example = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": json.dumps(tool_call)},
        ]
        examples.append(example)

    return examples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic scenario training data")
    parser.add_argument("--n", type=int, default=200, help="Number of examples to generate")
    parser.add_argument("--output", default="./dataset/format_b_decisions/format_b/synthetic_scenarios.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--format", choices=["legacy", "rich", "both"], default="legacy",
                        help="Scenario format: legacy (original), rich (snapshot), both")
    parser.add_argument("--merge", action="store_true",
                        help="Merge with existing train.jsonl and write combined output")
    args = parser.parse_args()

    examples = generate_examples(args.n, args.seed, fmt=args.format)

    # Stats
    from collections import Counter
    tool_counts = Counter()
    for ex in examples:
        tool = json.loads(ex[-1]["content"]).get("tool", "")
        tool_counts[tool] += 1

    print(f"Generated {len(examples)} synthetic examples")
    print(f"\nTool distribution:")
    for tool, count in tool_counts.most_common():
        print(f"  {tool:30s} {count:5d} ({count/len(examples)*100:.1f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    if args.merge:
        # Read existing train data and merge
        train_path = os.path.join(os.path.dirname(args.output), "train.jsonl")
        with open(train_path) as f:
            existing = [json.loads(line) for line in f]
        print(f"\nMerging with {len(existing)} existing examples")

        combined = existing + examples
        random.seed(args.seed)
        random.shuffle(combined)

        merged_path = os.path.join(os.path.dirname(args.output), "train_with_synthetic.jsonl")
        with open(merged_path, "w") as f:
            for ex in combined:
                f.write(json.dumps(ex) + "\n")
        print(f"Written {len(combined)} merged examples to {merged_path}")
    else:
        with open(args.output, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"\nWritten to {args.output}")

    # Show sample
    print(f"\nSample examples:")
    for ex in examples[:3]:
        user = ex[1]["content"][:100].replace("\n", " ")
        tool = json.loads(ex[-1]["content"])
        print(f"  User: {user}")
        print(f"  Tool: {tool['tool']}, Args: {json.dumps(tool.get('args',{}))[:80]}")
        print()


if __name__ == "__main__":
    main()
