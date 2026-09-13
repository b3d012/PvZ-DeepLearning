from types import SimpleNamespace
import unittest

import numpy as np
from pvz_env.environment import EpisodeConfig
from pvz_env.rewards import OutcomeReason

from pvz_deeplearning.adapters import PvZGymEnv


def snapshot(level=5):
    state = SimpleNamespace(adventure_level=level, wave=SimpleNamespace(spawned_waves=2))
    return SimpleNamespace(state=state, observation=np.zeros(5534, dtype=np.float32),
                           action_mask=np.ones(541, dtype=np.bool_))


class FakeHarnessEnv:
    def __init__(self, reason, terminated, truncated):
        self.reason, self.terminated, self.truncated = reason, terminated, truncated
    def reset(self, config):
        snap = snapshot()
        return SimpleNamespace(observation=snap.observation, action_mask=snap.action_mask, state=snap.state)
    def step(self, action):
        outcome = SimpleNamespace(reason=self.reason, terminated=self.terminated, truncated=self.truncated,
                                  reward=1.0, components={"terminal": 1.0})
        return SimpleNamespace(outcome=outcome, after=snapshot(), before=None, action_legal=True)


class AdapterTests(unittest.TestCase):
    def make(self, reason, terminated=False, truncated=False):
        return PvZGymEnv(FakeHarnessEnv(reason, terminated, truncated), lambda: EpisodeConfig("x"))

    def test_spaces_reset_and_mask(self):
        env = self.make(None)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (5534,)); self.assertEqual(env.action_space.n, 541)
        self.assertEqual(env.action_masks().dtype, np.bool_)

    def test_natural_outcome_terminates(self):
        _, _, terminated, truncated, _ = self.make(OutcomeReason.WIN, True).step(0)
        self.assertTrue(terminated); self.assertFalse(truncated)

    def test_technical_outcome_truncates(self):
        _, _, terminated, truncated, info = self.make(OutcomeReason.STATE_UNAVAILABLE, False, True).step(0)
        self.assertFalse(terminated); self.assertTrue(truncated); self.assertTrue(info["technical_truncation"])

    def test_horizon_not_technical(self):
        _, _, _, truncated, info = self.make(OutcomeReason.MAX_STEPS, False, True).step(0)
        self.assertTrue(truncated); self.assertFalse(info["technical_truncation"])
