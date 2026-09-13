"""Adapters for frozen harness baselines under the Phase 4 evaluator."""

from __future__ import annotations

from typing import Any

from pvz_env.baselines import RandomPolicyConfig, RandomValidActionPolicy, SimpleHeuristicPolicy


class HarnessBaselineSelector:
    def __init__(self, name: str, environment: Any, seed: int = 0) -> None:
        self.environment = environment
        if name == "random-valid":
            self.policy = RandomValidActionPolicy(RandomPolicyConfig(seed))
        elif name == "scripted-heuristic":
            self.policy = SimpleHeuristicPolicy()
        else:
            raise ValueError("unknown harness baseline")

    def __call__(self, observation: Any, action_mask: Any) -> int:
        state = getattr(self.environment, "current_state", None)
        return self.policy.select_action(observation, action_mask, state=state).action_index
