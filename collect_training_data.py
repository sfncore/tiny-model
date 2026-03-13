#!/usr/bin/env python3
"""
Gas Town Patrol Session Exporter

Self-contained script (stdlib only) that collects raw session JSONL files
from all patrol roles (witness, polecats, deacon, refinery, mayor) into a
.tar.gz archive for training data curation. Does NOT process or transform
the data — just bundles the raw sessions so we can curate (and re-curate)
from them later.

Usage:
    python3 collect_training_data.py
    python3 collect_training_data.py --output ~/my-export.tar.gz
    python3 collect_training_data.py --session-dirs ~/.claude/projects/*-witness/
    python3 collect_training_data.py --dry-run

Requires: Python 3.8+, no pip installs.
"""

import argparse
import glob
import hashlib
import json
import os
import platform
import sys
import tarfile
import time
from io import BytesIO


def find_session_files(session_dirs, min_size=5000, max_size=5_000_000):
    """Find all witness session JSONL files within size bounds."""
    files = []
    for d in session_dirs:
        d = os.path.expanduser(d)
        if not os.path.isdir(d):
            continue
        for path in glob.glob(os.path.join(d, "*.jsonl")):
            try:
                size = os.path.getsize(path)
                if min_size <= size <= max_size:
                    files.append(path)
            except OSError:
                continue
    return sorted(files)


def file_sha256(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def quick_session_stats(path):
    """Quick stats from a session file without full parsing."""
    n_lines = 0
    n_assistant = 0
    n_bash = 0
    first_ts = None
    last_ts = None

    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Grab timestamps
                ts = record.get("timestamp")
                if ts:
                    if first_ts is None:
                        first_ts = ts
                    last_ts = ts

                if record.get("type") == "assistant":
                    n_assistant += 1
                    msg = record.get("message", {})
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if (item.get("type") == "tool_use"
                                    and item.get("name") == "Bash"):
                                n_bash += 1
    except Exception:
        pass

    return {
        "lines": n_lines,
        "assistant_turns": n_assistant,
        "bash_calls": n_bash,
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
    }


PATROL_ROLES = ["witness", "polecats", "deacon", "refinery", "mayor"]


def classify_dir_role(dirname):
    """Classify a project directory into a patrol role, or None."""
    for role in PATROL_ROLES:
        if role in dirname:
            return role
    return None


def discover_session_dirs():
    """Auto-discover session directories for all patrol roles."""
    claude_projects = os.path.expanduser("~/.claude/projects/")
    if not os.path.isdir(claude_projects):
        return []

    dirs = []
    for role in PATROL_ROLES:
        dirs.extend(glob.glob(os.path.join(claude_projects, f"*-{role}*")))

    if dirs:
        return sorted(set(dirs))

    # Fallback: scan all project dirs
    return glob.glob(os.path.join(claude_projects, "*"))


def main():
    parser = argparse.ArgumentParser(
        description="Export raw Gas Town patrol sessions for training data curation"
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.expanduser("~/gastown-patrol-export.tar.gz"),
        help="Output archive path (default: ~/gastown-patrol-export.tar.gz)",
    )
    parser.add_argument(
        "--session-dirs", nargs="+", default=None,
        help="Session directories to scan (default: auto-discover)",
    )
    parser.add_argument(
        "--min-size", type=int, default=5000,
        help="Min session file size in bytes (default: 5000)",
    )
    parser.add_argument(
        "--max-size", type=int, default=5_000_000,
        help="Max session file size in bytes (default: 5000000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be collected without writing archive",
    )
    parser.add_argument(
        "--no-hash", action="store_true",
        help="Skip per-file SHA-256 (faster, archive still gzip-checksummed)",
    )
    args = parser.parse_args()

    # Discover session directories
    if args.session_dirs is None:
        args.session_dirs = discover_session_dirs()

    if not args.session_dirs:
        print("No session directories found.")
        print("Check that ~/.claude/projects/ exists and contains patrol sessions.")
        sys.exit(1)

    print(f"Scanning {len(args.session_dirs)} session directories...")
    files = find_session_files(args.session_dirs, args.min_size, args.max_size)
    print(f"Found {len(files)} session files ({args.min_size//1000}KB-{args.max_size//1000}KB)")

    if not files:
        print("No session files found in size range.")
        sys.exit(1)

    # Collect file info + quick stats
    file_entries = []
    total_bytes = 0

    for i, path in enumerate(files):
        if (i + 1) % 200 == 0:
            print(f"  Scanning {i + 1}/{len(files)}...", file=sys.stderr)

        size = os.path.getsize(path)
        total_bytes += size

        # Derive a stable archive name from the path:
        # ~/.claude/projects/<project>/<session>.jsonl -> <project>/<session>.jsonl
        parts = path.split(os.sep)
        try:
            proj_idx = parts.index("projects")
            archive_name = os.path.join(*parts[proj_idx + 1:])
        except ValueError:
            archive_name = os.path.basename(path)

        stats = quick_session_stats(path)

        file_entries.append({
            "source_path": path,
            "archive_name": archive_name,
            "size_bytes": size,
            "sha256": None if args.no_hash else file_sha256(path),
            "stats": stats,
        })

    # Summary
    total_mb = total_bytes / (1024 * 1024)
    total_bash = sum(e["stats"]["bash_calls"] for e in file_entries)
    total_assistant = sum(e["stats"]["assistant_turns"] for e in file_entries)

    # Classify entries by role
    role_counts = {r: 0 for r in PATROL_ROLES}
    role_bytes = {r: 0 for r in PATROL_ROLES}
    for e in file_entries:
        role = classify_dir_role(e["archive_name"])
        if role:
            role_counts[role] += 1
            role_bytes[role] += e["size_bytes"]

    print(f"\n{'=' * 55}")
    print(f"EXPORT SUMMARY")
    print(f"{'=' * 55}")
    print(f"  Session files:       {len(file_entries)}")
    print(f"  Total raw size:      {total_mb:.1f} MB")
    print(f"  Assistant turns:     {total_assistant}")
    print(f"  Bash tool calls:     {total_bash}")

    print(f"\n  By role:")
    for role in PATROL_ROLES:
        count = role_counts[role]
        mb = role_bytes[role] / (1024 * 1024)
        if count > 0:
            print(f"    {role:12s} {count:5d} sessions  ({mb:.1f} MB)")

    if args.dry_run:
        print(f"\n  (dry run — no archive written)")
        return

    # Build manifest
    manifest = {
        "version": 2,
        "description": "Gas Town patrol session export for training data curation",
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "total_files": len(file_entries),
        "total_bytes": total_bytes,
        "total_assistant_turns": total_assistant,
        "total_bash_calls": total_bash,
        "roles": {r: role_counts[r] for r in PATROL_ROLES if role_counts[r]},
        "files": [
            {
                "name": e["archive_name"],
                "role": classify_dir_role(e["archive_name"]),
                "size": e["size_bytes"],
                "sha256": e["sha256"],
                "lines": e["stats"]["lines"],
                "assistant_turns": e["stats"]["assistant_turns"],
                "bash_calls": e["stats"]["bash_calls"],
                "first_timestamp": e["stats"]["first_timestamp"],
                "last_timestamp": e["stats"]["last_timestamp"],
            }
            for e in file_entries
        ],
    }

    # Write tar.gz
    output_path = os.path.expanduser(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"\nWriting archive...")

    with tarfile.open(output_path, "w:gz") as tar:
        # Add manifest first
        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo(name="manifest.json")
        info.size = len(manifest_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, BytesIO(manifest_bytes))

        # Add session files
        for i, entry in enumerate(file_entries):
            if (i + 1) % 200 == 0:
                print(f"  Archiving {i + 1}/{len(file_entries)}...",
                      file=sys.stderr)
            tar.add(entry["source_path"], arcname=entry["archive_name"])

    archive_mb = os.path.getsize(output_path) / (1024 * 1024)
    ratio = archive_mb / total_mb * 100 if total_mb > 0 else 0

    print(f"\nWrote {output_path}")
    print(f"  Archive size: {archive_mb:.1f} MB ({ratio:.0f}% of raw)")
    print(f"  Contains: {len(file_entries)} sessions + manifest.json")
    print(f"\nShare this file to contribute witness training data!")


if __name__ == "__main__":
    main()
