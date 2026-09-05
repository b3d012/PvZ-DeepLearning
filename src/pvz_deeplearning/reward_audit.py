"""Reward-component aggregation and simple reward-hacking diagnostics."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def audit_records(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    components: defaultdict[str, float] = defaultdict(float)
    episode_rewards: defaultdict[str, float] = defaultdict(float)
    actions: Counter[str] = Counter()
    events: list[dict[str, Any]] = []
    for record in records:
        outcome = record.get("outcome")
        if outcome is None and isinstance(record.get("reward"), dict):
            outcome = record["reward"]
        if isinstance(outcome, dict):
            reward_components = outcome.get("components", record.get("reward_components", {}))
            total = float(outcome.get("reward", record.get("reward", 0.0)))
        else:
            reward_components = record.get("reward_components", {})
            total = float(record.get("reward", 0.0))
        episode = str(record.get("episode_id", "unknown"))
        episode_rewards[episode] += total
        for name, value in reward_components.items():
            components[name] += float(value)
        action = record.get("action") or {}
        actions[str(action.get("action_type", record.get("action_type", "unknown")))] += 1
        events.append({"episode_id": episode, "step_index": record.get("step_index"), "reward": total})
    total_reward = sum(episode_rewards.values())
    dominant = max(components, key=lambda key: abs(components[key]), default=None)
    dominant_share = abs(components[dominant]) / sum(abs(x) for x in components.values()) if dominant else 0.0
    warnings = []
    if dominant_share > 0.9 and len(components) > 1:
        warnings.append(f"component {dominant!r} contributes {dominant_share:.1%} of absolute shaped reward")
    return {
        "total_reward": total_reward, "components": dict(sorted(components.items())),
        "episodes": dict(sorted(episode_rewards.items())), "actions": dict(actions),
        "largest_positive_events": sorted(events, key=lambda x: x["reward"], reverse=True)[:10],
        "largest_negative_events": sorted(events, key=lambda x: x["reward"])[:10],
        "warnings": warnings,
    }
