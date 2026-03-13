#!/usr/bin/env python3
"""
Witness patrol serving shim — loads a fine-tuned SmolLM2-135M model and runs
patrol cycles: gather context → inference → execute tool call → backoff.

Replaces the Claude Code witness agent for patrol decisions.

Usage:
    python serve.py --checkpoint ./checkpoints/smollm2-135m_fmtb_full_ep3_800hardened_v2/
    python serve.py --checkpoint ./checkpoints/... --shadow        # observe only
    python serve.py --checkpoint ./checkpoints/... --once          # single cycle
    python serve.py --checkpoint ./checkpoints/... --interval 60   # fixed interval
"""

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from snapshot_format import format_snapshot

SYSTEM_PROMPT = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""

BACKOFF_MIN = 30
BACKOFF_MAX = 300

log = logging.getLogger("witness-shim")


# ---------------------------------------------------------------------------
# Model loading & inference (from evaluate.py)
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str):
    """Load model and tokenizer from checkpoint."""
    log.info("Loading model from %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Loaded: %.1fM params", n_params / 1e6)
    return model, tokenizer


def model_decide(model, tokenizer, context: str) -> dict:
    """Run inference on patrol context, return parsed tool call dict."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    ).strip()

    log.debug("Raw output (%dms): %s", latency_ms, generated)

    parsed = parse_json_output(generated)
    if parsed is None:
        log.warning("Failed to parse JSON from model output: %s", generated)
        return {"tool": "none", "args": {}, "_raw": generated, "_latency_ms": latency_ms}

    parsed["_raw"] = generated
    parsed["_latency_ms"] = latency_ms
    return parsed


def parse_json_output(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Patrol context gathering
# ---------------------------------------------------------------------------

def run_cmd(cmd: str, timeout: int = 15) -> str:
    """Run a shell command and return stdout (or error string)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else f"[error] {r.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "[error] command timed out"


def gather_patrol_context() -> str:
    """Gather patrol context via gt CLI commands (legacy 2-command version)."""
    polecats = run_cmd("gt polecat list --all")
    inbox = run_cmd("gt mail inbox --unread")

    parts = ["Polecats", "", polecats or "No active polecats."]
    if inbox and inbox != "[error] command timed out":
        parts += ["", "Inbox", "", inbox]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

STATE_FILE = os.path.expanduser("~/.claude/witness_state.json")


def load_state() -> dict:
    """Load persistent patrol state from disk."""
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"patrol_count": 0, "idle_cycles": 0, "last_action": "none"}


def save_state(state: dict, decision: dict) -> None:
    """Update and persist patrol state after a decision."""
    state["patrol_count"] += 1
    if decision.get("tool", "none") != "none":
        state["idle_cycles"] = 0
    else:
        state["idle_cycles"] += 1
    state["last_action"] = decision.get("tool", "none")
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# Rich context gathering (10 commands → structured snapshot)
# ---------------------------------------------------------------------------

def gather_patrol_context_rich(rig: str) -> str:
    """Gather rich patrol context via 10 gt CLI commands."""
    timeout = 10

    # 1. Polecat status
    polecats = run_cmd(f"gt polecat list {shlex.quote(rig)}", timeout=timeout)

    # 2. Inbox summary
    inbox = run_cmd("gt mail inbox --unread", timeout=timeout)

    # 3. Read unread messages (max 3)
    inbox_detail = ""
    if inbox and "[error]" not in inbox:
        # Extract mail IDs from inbox output (lines with IDs like hq-xxx)
        mail_ids = re.findall(r'\b([a-z]+-[a-z0-9]{4,})\b', inbox)
        for mid in mail_ids[:3]:
            msg = run_cmd(f"gt mail read {shlex.quote(mid)}", timeout=timeout)
            if msg and "[error]" not in msg:
                inbox_detail += f"\n--- {mid} ---\n{msg[:200]}"

    inbox_full = inbox or "No unread messages."
    if inbox_detail:
        inbox_full += inbox_detail

    # 4. Cleanup wisps
    cleanup = run_cmd("bd list --label=cleanup --status=open", timeout=timeout)

    # 5. Refinery status
    refinery = run_cmd(
        f"gt session status {shlex.quote(rig)}/refinery", timeout=timeout
    )

    # 6. Deacon health
    deacon = run_cmd("tmux has-session -t hq-deacon 2>/dev/null && echo alive || echo dead",
                     timeout=timeout)

    # 7. Active beads
    active_beads = run_cmd("bd list --status=in_progress", timeout=timeout)

    # 8. Timer gates
    timer_gates = run_cmd("bd gate check --type=timer", timeout=timeout)

    # Build infrastructure summary
    deacon_status = "alive" if deacon and "alive" in deacon else "dead"
    refinery_status = refinery if refinery and "[error]" not in refinery else "unknown"
    infra = f"Deacon: {deacon_status}\nRefinery: {refinery_status}"

    # Build active work summary
    active_parts = []
    if active_beads and active_beads != "[error] command timed out":
        active_parts.append(active_beads)
    if timer_gates and timer_gates != "[error] command timed out":
        active_parts.append(f"Timer gates: {timer_gates}")
    active_work = "\n".join(active_parts) if active_parts else "None"

    # State
    state = load_state()
    state_str = (
        f"patrol_count: {state['patrol_count']}, "
        f"idle_cycles: {state['idle_cycles']}, "
        f"last_action: {state['last_action']}"
    )

    sections = {
        "Polecats": polecats or "No active polecats.",
        "Inbox": inbox_full,
        "Cleanup Wisps": cleanup if cleanup and "[error]" not in cleanup else "None",
        "Infrastructure": infra,
        "Active Work": active_work,
        "State": state_str,
    }

    return format_snapshot(sections)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def execute_tool(decision: dict, shadow: bool = False) -> str:
    """Map a tool-call dict to a gt CLI command and execute it."""
    tool = decision.get("tool", "none")
    args = decision.get("args", {})

    cmd = _build_command(tool, args)
    if cmd is None:
        log.info("No-op: tool=%s", tool)
        return ""

    if shadow:
        log.info("[SHADOW] Would run: %s", cmd)
        return f"[shadow] {cmd}"

    log.info("Executing: %s", cmd)
    result = run_cmd(cmd, timeout=30)
    log.info("Result: %s", result[:200])
    return result


def _build_command(tool: str, args: dict) -> str | None:
    """Build a shell command string from tool name and args. Returns None for no-op."""
    if tool == "none":
        return None

    if tool == "gt_nudge":
        target = args.get("target", "")
        message = args.get("message", "")
        if not target:
            return None
        cmd = f"gt nudge {shlex.quote(target)}"
        if message:
            cmd += f" -m {shlex.quote(message)}"
        return cmd

    if tool == "gt_polecat_nuke":
        target = args.get("target", "")
        if not target:
            return None
        cmd = f"gt polecat nuke {shlex.quote(target)}"
        if args.get("force"):
            cmd += " --force"
        return cmd

    if tool == "gt_peek":
        target = args.get("target", "")
        if not target:
            return None
        cmd = f"gt peek {shlex.quote(target)}"
        if args.get("lines"):
            cmd += f" --lines {int(args['lines'])}"
        return cmd

    if tool == "gt_mail_inbox":
        return "gt mail inbox"

    if tool == "gt_mail_read":
        mail_id = args.get("mail_id", "")
        if not mail_id:
            return "gt mail inbox"
        return f"gt mail read {shlex.quote(str(mail_id))}"

    if tool == "gt_mail_send":
        recipient = args.get("recipient", "")
        subject = args.get("subject", "")
        body = args.get("body", "")
        if not recipient:
            return None
        cmd = f"gt mail send {shlex.quote(recipient)}"
        if subject:
            cmd += f" -s {shlex.quote(subject)}"
        if body:
            cmd += f" -m {shlex.quote(body)}"
        return cmd

    if tool == "gt_patrol_report":
        status = args.get("status", "ok")
        note = args.get("note", "")
        cmd = f"gt patrol report --summary {shlex.quote(status)}"
        if note:
            cmd += f" --note {shlex.quote(note)}"
        return cmd

    if tool == "check_tmux_session":
        session = args.get("session", "")
        if not session:
            return None
        return f"tmux has-session -t {shlex.quote(session)}"

    if tool == "gt_session_status":
        return "gt status --fast"

    if tool == "gt_polecat_list":
        return "gt polecat list"

    if tool == "gt_escalate":
        severity = args.get("severity", "HIGH")
        message = args.get("message", "")
        cmd = f"gt escalate -s {shlex.quote(severity)}"
        if message:
            cmd += f" {shlex.quote(message)}"
        return cmd

    if tool == "gt_handoff":
        target = args.get("target", "")
        if not target:
            return None
        return f"gt handoff {shlex.quote(target)}"

    if tool == "check_git_state":
        session = args.get("session", "")
        if session:
            return f"tmux send-keys -t {shlex.quote(session)} 'git status' Enter"
        return "git status"

    if tool == "bash":
        command = args.get("command", "")
        if not command:
            return None
        return command

    log.warning("Unknown tool: %s", tool)
    return None


# ---------------------------------------------------------------------------
# Main patrol loop
# ---------------------------------------------------------------------------

def patrol_loop(model, tokenizer, *, shadow: bool = False,
                once: bool = False, fixed_interval: int | None = None,
                rig: str | None = None):
    """Run the patrol loop with exponential backoff."""
    interval = fixed_interval or BACKOFF_MIN
    cycle = 0
    use_rich = rig is not None

    log.info("Starting patrol loop (shadow=%s, once=%s, interval=%s, rig=%s, rich=%s)",
             shadow, once, fixed_interval or "adaptive", rig, use_rich)

    try:
        while True:
            cycle += 1
            ts = time.strftime("%H:%M:%S")

            # 1. Gather context
            if use_rich:
                context = gather_patrol_context_rich(rig)
            else:
                context = gather_patrol_context()
            ctx_summary = context[:120].replace("\n", " ")
            log.info("[%s] cycle=%d context=%s", ts, cycle, ctx_summary)

            # 2. Inference
            decision = model_decide(model, tokenizer, context)
            tool = decision.get("tool", "none")
            latency = decision.get("_latency_ms", 0)
            log.info("[%s] decision: tool=%s latency=%.0fms", ts, tool, latency)

            # 3. Execute
            result = execute_tool(decision, shadow=shadow)
            if result:
                log.info("[%s] result: %s", ts, result[:200])

            # 4. Update state (rich mode only)
            if use_rich:
                state = load_state()
                save_state(state, decision)

            # 5. Backoff
            if once:
                log.info("Single cycle complete, exiting.")
                break

            if fixed_interval:
                interval = fixed_interval
            elif tool != "none" and not shadow:
                interval = BACKOFF_MIN  # reset on real action only
            else:
                interval = min(interval * 2, BACKOFF_MAX)  # exponential backoff

            log.info("[%s] sleeping %ds", ts, interval)
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Interrupted, shutting down.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Witness patrol shim — tiny model serving loop"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--rig", type=str, default=None,
                        help="Rig name for rich context gathering (enables 10-command snapshot)")
    parser.add_argument("--shadow", action="store_true",
                        help="Shadow mode: log decisions but do not execute")
    parser.add_argument("--once", action="store_true",
                        help="Run a single patrol cycle then exit")
    parser.add_argument("--interval", type=int, default=None,
                        help="Fixed sleep interval in seconds (disables adaptive backoff)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model, tokenizer = load_model(args.checkpoint)
    patrol_loop(model, tokenizer, shadow=args.shadow,
                once=args.once, fixed_interval=args.interval, rig=args.rig)


if __name__ == "__main__":
    main()
