"""Block destructive Git commands before a Codex tool executes them."""

from __future__ import annotations

import json
import sys
from typing import Any

BLOCKED = (
    "git reset --hard", "git clean -f", "git clean -fd", "git clean -d -f",
    "git push --force", "git push -f", "git rebase", "git filter-branch",
    "git filter-repo", "git branch -D", "git checkout -- .", "git restore .",
    "git restore --staged",
)

def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [part for item in value.values() for part in _strings(item)]
    if isinstance(value, list):
        return [part for item in value for part in _strings(item)]
    return []

def main() -> int:
    raw = sys.stdin.read()
    try:
        payload: Any = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        payload = {"raw": raw}
    haystack = "\n".join(_strings(payload)).lower()
    matched = next((pattern for pattern in BLOCKED if pattern in haystack), None)
    if matched:
        print(f"BLOCKED: destructive Git operation matched '{matched}'.", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
