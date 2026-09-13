"""Small deterministic masked environment for safe ML integration tests."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from pvz_deeplearning.harness import EXPECTED_HARNESS_CONTRACT


class MockPvZEnv(gym.Env[np.ndarray, int]):
    """A sparse toy objective with the exact Observation/Action v1 dimensions."""

    def __init__(self, episode_steps: int = 16) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(5534,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(541)
        self.episode_steps = episode_steps
        self._step = 0
        self._target = 1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._step = 0
        self._target = int(self.np_random.integers(1, 9))
        return self._observation(), {"mock": True}

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(EXPECTED_HARNESS_CONTRACT["action_count"], dtype=np.bool_)
        mask[0] = True
        mask[1:10] = True
        return mask

    def step(self, action: int):
        if not self.action_masks()[int(action)]:
            raise ValueError("masked action selected")
        reward = 1.0 if int(action) == self._target else -0.01
        self._step += 1
        terminated = self._step >= self.episode_steps
        return self._observation(), reward, terminated, False, {
            "mock": True, "wave": self._step // 4, "technical_truncation": False
        }

    def _observation(self) -> np.ndarray:
        obs = np.zeros(5534, dtype=np.float32)
        obs[0] = self._step / self.episode_steps
        obs[1 + self._target] = 1.0
        return obs
