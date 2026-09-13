"""Deterministic repository validation profiles; never sends gameplay input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def run_step(name: str, command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    return {"name": name, "command": command, "returncode": result.returncode,
            "duration_seconds": round(time.monotonic() - started, 3),
            "stdout": result.stdout[-4000:], "stderr": result.stderr[-4000:],
            "success": result.returncode == 0}

def git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None

def main(argv: list[str] | None = None) -> int:
    started_at = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("fast", "full", "live-preflight"), required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    steps = [
        ("git diff --check", ["git", "diff", "--check"]),
        ("compileall", [sys.executable, "-m", "compileall", "-q", "src", "tests", "scripts"]),
        ("import validation", [sys.executable, "-c", "import pvz_deeplearning, pvz_env, pvz_runtime, stable_baselines3, sb3_contrib"]),
        ("unit tests", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"]),
        ("harness contract", [sys.executable, "scripts/check_harness.py"]),
    ]
    if args.profile in {"full", "live-preflight"}:
        steps.extend([
            ("CLI help smoke", [sys.executable, "-m", "pvz_deeplearning.cli", "--help"]),
            ("CLI doctor import smoke", [sys.executable, "-m", "pvz_deeplearning.cli", "doctor"]),
        ])
    if args.profile == "live-preflight":
        steps.append(("read-only live readiness doctor", [sys.executable, "-m", "pvz_deeplearning.cli", "doctor"]))
    results = [run_step(name, command) for name, command in steps]
    payload = {"profile": args.profile, "success": all(step["success"] for step in results),
               "started_at": started_at.isoformat(),
               "finished_at": datetime.now(timezone.utc).isoformat(),
               "git_sha": git_value("rev-parse", "HEAD"), "git_branch": git_value("branch", "--show-current"),
               "python": {"executable": sys.executable, "version": platform.python_version()}, "steps": results}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload["success"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
