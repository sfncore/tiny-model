#!/usr/bin/env python3
"""
Evaluation script for fine-tuned witness models.

Measures:
1. JSON validity rate — does the model produce parseable JSON?
2. Tool accuracy — does it pick the right tool for the scenario?
3. Inference latency — CPU wall-clock time per generation
4. Format compliance — does the output match expected schema?

Usage:
    python evaluate.py --checkpoint ./checkpoints/smollm2-135m_fmtb_full_ep3/final
    python evaluate.py --checkpoint ./checkpoints/smollm2-135m_fmtb_full_ep3/final --eval-set ./dataset/format_b/chunked_2k/eval.jsonl
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from snapshot_format import format_snapshot
from shared import load_model


VALID_TOOLS = {
    "gt_polecat_list", "gt_polecat_nuke", "gt_peek", "gt_session_status",
    "gt_nudge", "gt_mail_inbox", "gt_mail_read", "gt_mail_send",
    "gt_patrol_report", "gt_handoff", "gt_escalate",
    "bd_show", "bd_list", "bd_close", "bd_children",
    "check_git_state", "check_tmux_session",
    "bash", "none",
}

# ---------------------------------------------------------------------------
# Legacy scenarios (original 8, flat format)
# ---------------------------------------------------------------------------

SCENARIOS_LEGACY = [
    {
        "name": "idle_patrol",
        "messages": [
            {"role": "user", "content": "Polecats\n\nNo active polecats."},
        ],
        "expected_tools": {"gt_patrol_report", "none"},
        "description": "No polecats active → should report idle or do nothing",
    },
    {
        "name": "healthy_polecat",
        "messages": [
            {"role": "user", "content": "Polecats\n\n  ● gastown/furiosa  working\n    hq-wisp-2v214\n  ○ gastown/nux  done\n    gt-kvo.6\n\nLast activity: 2 minutes ago. Making progress on bcc-8rlwh. Healthy."},
        ],
        "expected_tools": {"gt_peek", "gt_session_status", "gt_polecat_list", "gt_patrol_report"},
        "description": "Active polecat → should peek or check status",
    },
    {
        "name": "stuck_polecat",
        "messages": [
            {"role": "user", "content": "Polecats\n\n  ● gastown/furiosa  working\n    hq-wisp-2v214\n\nLast activity: 45 minutes ago. No progress detected."},
        ],
        "expected_tools": {"gt_nudge", "gt_peek", "gt_session_status"},
        "description": "Stuck polecat → should nudge or investigate",
    },
    {
        "name": "completed_polecat",
        "messages": [
            {"role": "user", "content": "Polecats\n\n  ○ gastown/furiosa  done\n    hq-wisp-2v214\n\nAll work pushed. Branch merged."},
        ],
        "expected_tools": {"gt_polecat_nuke", "check_git_state", "gt_mail_send"},
        "description": "Completed polecat with clean git → should nuke or verify git",
    },
    {
        "name": "crash_loop",
        "messages": [
            {"role": "user", "content": "Polecats\n\n  ✗ gastown/furiosa  crashed\n    hq-wisp-2v214\n\nRestart count: 4. Last crash: segfault."},
        ],
        "expected_tools": {"gt_escalate", "gt_polecat_nuke", "gt_mail_send"},
        "description": "Crash-looping polecat → should escalate or force nuke",
    },
    {
        "name": "unpushed_work",
        "messages": [
            {"role": "user", "content": "Polecats\n\n  ○ gastown/furiosa  done\n    hq-wisp-2v214\n\nGit state: 3 unpushed commits on branch feature/fix-auth."},
        ],
        "expected_tools": {"gt_escalate", "gt_mail_send", "check_git_state"},
        "description": "Completed but unpushed work → should escalate, NOT nuke",
    },
    {
        "name": "check_infrastructure",
        "messages": [
            {"role": "user", "content": "Infrastructure check requested.\n\nDeacon: unknown\nRefinery: unknown"},
        ],
        "expected_tools": {"check_tmux_session", "gt_session_status", "bash"},
        "description": "Infrastructure check → should verify deacon/refinery sessions",
    },
    {
        "name": "mail_check",
        "messages": [
            {"role": "user", "content": "Check inbox for new messages."},
        ],
        "expected_tools": {"gt_mail_inbox", "gt_mail_read"},
        "description": "Mail check → should check inbox",
    },
]


def _snap(**sections) -> str:
    """Helper to build rich snapshot for eval scenarios."""
    defaults = {
        "Polecats": "No polecats found.",
        "Inbox": "No unread messages.",
        "Cleanup Wisps": "None",
        "Infrastructure": "Deacon: alive\nRefinery: running",
        "Active Work": "None",
        "State": "patrol_count: 12, idle_cycles: 2, last_action: none",
    }
    defaults.update(sections)
    return format_snapshot(defaults)


# ---------------------------------------------------------------------------
# Rich-snapshot scenarios (all 16: original 8 rewritten + 8 new)
# ---------------------------------------------------------------------------

SCENARIOS_RICH = [
    # --- Original 8, rewritten in snapshot format ---
    {
        "name": "idle_patrol",
        "messages": [
            {"role": "user", "content": _snap()},
        ],
        "expected_tools": {"gt_patrol_report", "none"},
        "description": "No polecats active, everything healthy → idle or report",
    },
    {
        "name": "healthy_polecat",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ● gastown/furiosa  working\n    hq-wisp-2v214\n\nLast activity: 2 minutes ago. Making progress on bcc-8rlwh.",
                **{"Active Work": "bcc-8rlwh (in_progress, assigned to gastown/furiosa)"},
            )},
        ],
        "expected_tools": {"gt_peek", "gt_session_status", "gt_polecat_list", "gt_patrol_report", "none"},
        "description": "Active healthy polecat → peek, monitor, or do nothing",
    },
    {
        "name": "stuck_polecat",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ● gastown/furiosa  working\n    hq-wisp-2v214\n\nLast activity: 45 minutes ago. No progress detected.",
                **{"Active Work": "bcc-8rlwh (in_progress, assigned to gastown/furiosa)"},
            )},
        ],
        "expected_tools": {"gt_nudge", "gt_peek"},
        "description": "Stuck polecat 45min → should nudge",
    },
    {
        "name": "completed_polecat",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ○ gastown/furiosa  done\n    hq-wisp-2v214\n\nAll work pushed. Branch merged.",
            )},
        ],
        "expected_tools": {"gt_polecat_nuke", "check_git_state", "gt_mail_send", "bd_close", "gt_escalate", "gt_patrol_report"},
        "description": "Completed polecat, clean git → nuke, close bead, verify git, or report",
    },
    {
        "name": "crash_loop",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ✗ gastown/furiosa  crashed\n    hq-wisp-2v214\n\nRestart count: 4. Last crash: segfault.",
            )},
        ],
        "expected_tools": {"gt_escalate", "gt_polecat_nuke", "gt_mail_send"},
        "description": "Crash-looping polecat → escalate or force nuke",
    },
    {
        "name": "unpushed_work",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ○ gastown/furiosa  done\n    hq-wisp-2v214\n\nGit state: 3 unpushed commits on branch feature/fix-auth.",
            )},
        ],
        "expected_tools": {"gt_escalate", "gt_mail_send", "check_git_state"},
        "description": "Completed but unpushed work → escalate, NOT nuke",
    },
    {
        "name": "check_infrastructure",
        "messages": [
            {"role": "user", "content": _snap(
                Infrastructure="Deacon: dead\nRefinery: running",
            )},
        ],
        "expected_tools": {"gt_mail_send", "gt_escalate"},
        "description": "Deacon down (refinery ok) → should escalate",
    },
    {
        "name": "idle_with_mail",
        "messages": [
            {"role": "user", "content": _snap(
                Inbox="2 unread\n  hq-m5t2w  from:hq/mayor  Routine check-in\n--- hq-m5t2w ---\nRoutine status check. How is your rig?",
            )},
        ],
        "expected_tools": {"gt_mail_send", "gt_patrol_report", "none"},
        "description": "Idle rig with routine mail → respond or report idle",
    },
    # --- New 8 scenarios ---
    {
        "name": "polecat_done_inbox",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ○ gastown/furiosa  done\n    hq-wisp-2v214",
                Inbox="1 unread\n  hq-m3k9x  from:gastown/furiosa  POLECAT_DONE\n--- hq-m3k9x ---\nAll work on bcc-8rlwh complete. Branch feature/fix-auth pushed. Git clean.",
                **{"Active Work": "bcc-8rlwh (in_progress)"},
            )},
        ],
        "expected_tools": {"gt_mail_send"},
        "description": "POLECAT_DONE in inbox → send MERGE_READY to refinery",
    },
    {
        "name": "lifecycle_shutdown",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ● gastown/nux  working\n    bcc-wisp-x8k3m",
                Inbox="1 unread\n  gt-m1r4v  from:system  LIFECYCLE:Shutdown\n--- gt-m1r4v ---\nAgent gastown/nux shutting down. Session bcc-wisp-x8k3m ending.",
            )},
        ],
        "expected_tools": {"check_git_state", "gt_polecat_nuke"},
        "description": "Lifecycle shutdown → verify git state or nuke",
    },
    {
        "name": "merged_notification",
        "messages": [
            {"role": "user", "content": _snap(
                Inbox="1 unread\n  bcc-m7j2p  from:gastown/refinery  MERGED\n--- bcc-m7j2p ---\nBranch feature/add-tests merged to main. Bead gt-4tp work complete.",
                **{"Cleanup Wisps": "  hq-wisp-ragxshg  cleanup  gt-4tp (merge cleanup)"},
            )},
        ],
        "expected_tools": {"bd_close"},
        "description": "MERGED notification + cleanup wisp → close cleanup bead",
    },
    {
        "name": "help_request",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ● bcc/rust  working\n    gt-kvo.6",
                Inbox="1 unread\n  zfc-m8n6q  from:bcc/rust  HELP: Build failing on deps\n--- zfc-m8n6q ---\nWorking on hq-3mn2. Build failing on dependency resolution. Requesting witness assistance.",
            )},
        ],
        "expected_tools": {"gt_mail_send", "gt_mail_read"},
        "description": "HELP request → escalate to mayor or read details",
    },
    {
        "name": "zombie_polecat",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ● gastown/chrome  working\n    bcc-wisp-gb0j5u\n\nSession bcc-wisp-gb0j5u not found in tmux. Agent state shows running.",
                **{"Active Work": "zfc-9qw1 (in_progress, assigned to gastown/chrome)"},
            )},
        ],
        "expected_tools": {"gt_mail_send", "gt_escalate"},
        "description": "Zombie polecat (running state, dead session) → escalate",
    },
    {
        "name": "stale_spawn",
        "messages": [
            {"role": "user", "content": _snap(
                Polecats="  ◐ hq/nitro  spawning\n    hq-wisp-2v214\n\nSpawning for 20 minutes. No activity detected.",
            )},
        ],
        "expected_tools": {"gt_escalate", "gt_mail_send"},
        "description": "Stale spawn >10min → escalate",
    },
    {
        "name": "orphaned_bead",
        "messages": [
            {"role": "user", "content": _snap(
                **{"Active Work": "bcc-8rlwh (in_progress, branch integration/beads-ide)\n  No assigned polecat found."},
            )},
        ],
        "expected_tools": {"gt_mail_send"},
        "description": "Orphaned bead with no polecat → notify mayor",
    },
    {
        "name": "infra_down",
        "messages": [
            {"role": "user", "content": _snap(
                Infrastructure="Deacon: dead\nRefinery: [error] session not found",
            )},
        ],
        "expected_tools": {"gt_mail_send", "gt_escalate"},
        "description": "Deacon + refinery both down → escalate",
    },
]

# Default: use rich scenarios
SCENARIOS = SCENARIOS_RICH



def generate_response(model, tokenizer, messages: list, system_prompt: str,
                      max_new_tokens: int = 150) -> tuple:
    """Generate a response and return (text, latency_ms)."""
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    prompt = tokenizer.apply_chat_template(
        full_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_len = inputs["input_ids"].shape[1]
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(
        out[0][input_len:],
        skip_special_tokens=True
    )
    return generated.strip(), latency


def _extract_json_balanced(text: str) -> dict | None:
    """Extract a JSON object using balanced brace matching."""
    for i, ch in enumerate(text):
        if ch == '{':
            depth = 0
            for j in range(i, len(text)):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[i:j+1])
                    except json.JSONDecodeError:
                        break  # This opening brace didn't work, try next
    return None


def parse_json_output(text: str) -> dict | None:
    """Try to extract a JSON object from model output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Use balanced brace matching to handle nested objects
    return _extract_json_balanced(text)


def evaluate_scenarios(model, tokenizer, system_prompt: str) -> dict:
    """Run scenario-based evaluation."""
    results = []

    print(f"\n{'='*70}")
    print("SCENARIO EVALUATION")
    print(f"{'='*70}")

    for scenario in SCENARIOS:
        output, latency = generate_response(model, tokenizer, scenario["messages"], system_prompt)
        parsed = parse_json_output(output)

        is_valid_json = parsed is not None
        tool_name = parsed.get("tool", "") if parsed else ""
        is_valid_tool = tool_name in VALID_TOOLS
        is_correct_tool = tool_name in scenario["expected_tools"]
        has_args = isinstance(parsed.get("args"), dict) if parsed else False

        result = {
            "scenario": scenario["name"],
            "description": scenario["description"],
            "output": output[:200],
            "parsed": parsed,
            "valid_json": is_valid_json,
            "valid_tool": is_valid_tool,
            "correct_tool": is_correct_tool,
            "has_args": has_args,
            "latency_ms": latency,
        }
        results.append(result)

        status = "OK" if (is_valid_json and is_correct_tool) else "FAIL" if not is_valid_json else "WRONG"
        print(f"\n  [{status:5s}] {scenario['name']}")
        print(f"         Expected: {scenario['expected_tools']}")
        print(f"         Got tool: {tool_name!r}")
        print(f"         Latency:  {latency:.0f}ms")
        print(f"         Output:   {output[:120]}")

    return results


def evaluate_eval_set(model, tokenizer, eval_path: str, system_prompt: str,
                      max_examples: int = 50) -> dict:
    """Evaluate on held-out eval set."""
    print(f"\n{'='*70}")
    print(f"EVAL SET: {eval_path}")
    print(f"{'='*70}")

    with open(eval_path) as f:
        eval_convs = [json.loads(line) for line in f]

    if max_examples:
        eval_convs = eval_convs[:max_examples]

    valid_json_count = 0
    valid_tool_count = 0
    correct_schema_count = 0
    latencies = []
    total_turns = 0

    for i, conv in enumerate(eval_convs):
        # Build conversation context and evaluate all assistant turns
        messages = []
        for m in conv:
            if m["role"] == "system":
                continue
            elif m["role"] == "user":
                messages.append(m)
            elif m["role"] == "assistant":
                # This is a turn we can evaluate
                if not messages:
                    continue

                output, latency = generate_response(
                    model, tokenizer, messages, system_prompt
                )
                latencies.append(latency)

                parsed = parse_json_output(output)
                if parsed is not None:
                    valid_json_count += 1
                    tool = parsed.get("tool", "")
                    if tool in VALID_TOOLS:
                        valid_tool_count += 1
                    if isinstance(parsed.get("args"), dict):
                        correct_schema_count += 1

                total_turns += 1
                # Add actual assistant message to maintain conversation context
                messages.append(m)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(eval_convs)} conversations...")

    return {
        "total_turns": total_turns,
        "valid_json": valid_json_count,
        "valid_json_rate": valid_json_count / max(total_turns, 1),
        "valid_tool": valid_tool_count,
        "valid_tool_rate": valid_tool_count / max(total_turns, 1),
        "correct_schema": correct_schema_count,
        "correct_schema_rate": correct_schema_count / max(total_turns, 1),
        "mean_latency_ms": sum(latencies) / max(len(latencies), 1),
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": sorted(latencies)[min(int(len(latencies) * 0.95), len(latencies) - 1)] if latencies else 0,
    }


def evaluate_regression(model, tokenizer, system_prompt: str) -> dict:
    """Run regression: evaluate original 8 scenarios in BOTH legacy and rich format."""
    print(f"\n{'='*70}")
    print("REGRESSION: Legacy vs Rich format (original 8 scenarios)")
    print(f"{'='*70}")

    # Only compare the first 8 scenarios which exist in both formats
    legacy_names = {s["name"] for s in SCENARIOS_LEGACY}
    rich_by_name = {s["name"]: s for s in SCENARIOS_RICH}

    results = []
    for legacy_s in SCENARIOS_LEGACY:
        name = legacy_s["name"]
        if name not in rich_by_name:
            continue

        rich_s = rich_by_name[name]

        # Legacy
        out_l, lat_l = generate_response(model, tokenizer, legacy_s["messages"], system_prompt)
        parsed_l = parse_json_output(out_l)
        tool_l = parsed_l.get("tool", "") if parsed_l else ""
        correct_l = tool_l in legacy_s["expected_tools"]

        # Rich
        out_r, lat_r = generate_response(model, tokenizer, rich_s["messages"], system_prompt)
        parsed_r = parse_json_output(out_r)
        tool_r = parsed_r.get("tool", "") if parsed_r else ""
        correct_r = tool_r in rich_s["expected_tools"]

        match = "MATCH" if correct_l == correct_r else "DRIFT"
        print(f"\n  [{match:5s}] {name}")
        print(f"         Legacy: tool={tool_l!r} correct={correct_l}")
        print(f"         Rich:   tool={tool_r!r} correct={correct_r}")

        results.append({
            "scenario": name,
            "legacy_tool": tool_l,
            "legacy_correct": correct_l,
            "rich_tool": tool_r,
            "rich_correct": correct_r,
            "match": correct_l == correct_r,
        })

    n_match = sum(1 for r in results if r["match"])
    print(f"\n  Regression: {n_match}/{len(results)} scenarios match between formats")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned witness model")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--eval-set", type=str, default=None,
                        help="Path to eval JSONL (optional, runs eval set evaluation)")
    parser.add_argument("--max-eval", type=int, default=50,
                        help="Max eval examples to process")
    parser.add_argument("--max-tokens", type=int, default=150,
                        help="Max new tokens to generate")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results JSON")
    parser.add_argument("--scenarios", choices=["rich", "legacy", "both"],
                        default="rich",
                        help="Which scenario set to run (default: rich)")
    parser.add_argument("--regression", action="store_true",
                        help="Run regression: compare legacy vs rich format on original 8")
    args = parser.parse_args()

    system_prompt = """You are a Witness agent. You respond ONLY with JSON tool calls.

For each turn, output exactly one JSON object:
{"tool": "<tool_name>", "args": {<arguments>}}

If no action is needed, output:
{"tool": "none", "args": {}}

Available tools: gt_polecat_list, gt_polecat_nuke, gt_peek, gt_session_status, gt_nudge, gt_mail_inbox, gt_mail_read, gt_mail_send, gt_patrol_report, gt_handoff, gt_escalate, bd_show, bd_list, bd_close, bd_children, check_git_state, check_tmux_session, bash"""

    model, tokenizer = load_model(args.checkpoint)

    # Select scenario set
    if args.scenarios == "legacy":
        scenarios = SCENARIOS_LEGACY
    elif args.scenarios == "both":
        # Deduplicate by name, preferring rich
        seen = set()
        scenarios = []
        for s in SCENARIOS_RICH:
            seen.add(s["name"])
            scenarios.append(s)
        for s in SCENARIOS_LEGACY:
            if s["name"] not in seen:
                scenarios.append(s)
    else:
        scenarios = SCENARIOS_RICH

    # Temporarily swap SCENARIOS for evaluate_scenarios
    global SCENARIOS
    SCENARIOS = scenarios

    # Scenario evaluation (always run)
    scenario_results = evaluate_scenarios(model, tokenizer, system_prompt)

    # Regression mode (optional)
    regression_results = None
    if args.regression:
        regression_results = evaluate_regression(model, tokenizer, system_prompt)

    # Eval set evaluation (optional)
    eval_set_results = None
    if args.eval_set:
        eval_set_results = evaluate_eval_set(
            model, tokenizer, args.eval_set, system_prompt,
            max_examples=args.max_eval
        )

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Scenario summary
    n_scenarios = len(scenario_results)
    n_valid = sum(1 for r in scenario_results if r["valid_json"])
    n_correct = sum(1 for r in scenario_results if r["correct_tool"])
    avg_lat = sum(r["latency_ms"] for r in scenario_results) / n_scenarios

    print(f"\nScenarios ({n_scenarios} total, format={args.scenarios}):")
    print(f"  JSON valid:    {n_valid}/{n_scenarios} ({n_valid/n_scenarios*100:.0f}%)")
    print(f"  Correct tool:  {n_correct}/{n_scenarios} ({n_correct/n_scenarios*100:.0f}%)")
    print(f"  Avg latency:   {avg_lat:.0f}ms")

    if regression_results:
        n_match = sum(1 for r in regression_results if r["match"])
        print(f"\nRegression ({len(regression_results)} scenarios):")
        print(f"  Format match:  {n_match}/{len(regression_results)}")

    if eval_set_results:
        print(f"\nEval set ({eval_set_results['total_turns']} turns):")
        print(f"  JSON valid:    {eval_set_results['valid_json_rate']*100:.0f}%")
        print(f"  Valid tool:    {eval_set_results['valid_tool_rate']*100:.0f}%")
        print(f"  Schema ok:     {eval_set_results['correct_schema_rate']*100:.0f}%")
        print(f"  Latency p50:   {eval_set_results['p50_latency_ms']:.0f}ms")
        print(f"  Latency p95:   {eval_set_results['p95_latency_ms']:.0f}ms")

    # Save results
    output_path = args.output or os.path.join(
        os.path.dirname(args.checkpoint), "eval_results.json"
    )
    all_results = {
        "checkpoint": args.checkpoint,
        "scenario_format": args.scenarios,
        "scenario_summary": {
            "n_scenarios": n_scenarios,
            "valid_json_rate": n_valid / n_scenarios,
            "correct_tool_rate": n_correct / n_scenarios,
            "avg_latency_ms": avg_lat,
        },
        "scenarios": scenario_results,
    }
    if regression_results:
        all_results["regression"] = regression_results
    if eval_set_results:
        all_results["eval_set"] = eval_set_results

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
