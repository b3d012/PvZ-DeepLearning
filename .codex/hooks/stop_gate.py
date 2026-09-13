"""Run one bounded offline validation continuation after repository changes."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VALIDATED_PYTHON = r"E:\Dev\miniconda3\envs\pvz-rl\python.exe"

def main() -> int:
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if status.returncode != 0 or not status.stdout.strip():
        return 0
    interpreter = os.environ.get("PVZ_PYTHON", VALIDATED_PYTHON)
    result = subprocess.run(
        [interpreter, "scripts/validate.py", "--profile", "fast"], cwd=ROOT, check=False
    )
    if result.returncode == 0:
        return 0
    if os.environ.get("PVZ_STOP_GATE_CONTINUATION") == "1":
        print("Stop validation failed twice; reporting failure without another continuation.", file=sys.stderr)
        return 0
    print("Stop validation failed; one bounded investigation continuation is allowed.", file=sys.stderr)
    return 1

if __name__ == "__main__":
    raise SystemExit(main())
