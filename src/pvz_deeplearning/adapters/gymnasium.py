"""Gymnasium adapter preserving game termination versus technical truncation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np

from pvz_env.actions import ACTION_COUNT
from pvz_env.environment import EpisodeConfig, PvZEnvironment
from pvz_env.observation import OBSERVATION_SPEC
from pvz_env.rewards import OutcomeReason


class PvZGymEnv(gym.Env[np.ndarray, int]):
    """Thin adapter; reset preparation remains an injected harness-layer concern."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        environment: PvZEnvironment,
        episode_factory: Callable[[], EpisodeConfig],
        *,
        prepare_reset: Callable[[EpisodeConfig], None] | None = None,
        before_step: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.environment = environment
        self.episode_factory = episode_factory
        self.prepare_reset = prepare_reset
        self.before_step = before_step
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=OBSERVATION_SPEC.flat_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(ACTION_COUNT)
        self._mask = np.zeros(ACTION_COUNT, dtype=np.bool_)
        self._mask[0] = True
        self.current_state: Any = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        config = self.episode_factory()
        if self.prepare_reset is not None:
            self.prepare_reset(config)
        result = self.environment.reset(config)
        self.current_state = result.state
        self._mask = result.action_mask.copy()
        return result.observation.copy(), self._info(result.state, "reset", None)

    def step(self, action: int):
        if self.before_step is not None:
            self.before_step()
        result = self.environment.step(int(action))
        outcome = result.outcome
        assert outcome is not None
        snapshot = result.after or result.before
        if snapshot is None:
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            self._mask = np.zeros(self.action_space.n, dtype=np.bool_)
            self._mask[0] = True
        else:
            observation = snapshot.observation.copy()
            self._mask = snapshot.action_mask.copy()
            self.current_state = snapshot.state
        reason = outcome.reason.value if outcome.reason else None
        game_terminal = outcome.reason in (OutcomeReason.WIN, OutcomeReason.LOSS)
        terminated = bool(outcome.terminated and game_terminal)
        truncated = bool(outcome.truncated or (outcome.terminated and not game_terminal))
        info = self._info(snapshot.state if snapshot else None, reason, result)
        info["reward_components"] = dict(outcome.components)
        info["technical_truncation"] = bool(truncated and reason not in {"max_steps"})
        return observation, float(outcome.reward), terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    @staticmethod
    def _info(state: Any, reason: str | None, result: Any) -> dict[str, Any]:
        return {
            "outcome_reason": reason,
            "adventure_level": getattr(state, "adventure_level", None),
            "wave": getattr(getattr(state, "wave", None), "spawned_waves", None),
            "action_legal": getattr(result, "action_legal", None),
        }
