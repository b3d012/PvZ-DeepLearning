import unittest
import numpy as np

from pvz_deeplearning.evaluation import evaluate_masked
from pvz_deeplearning.mock_env import MockPvZEnv


class MockEvaluationTests(unittest.TestCase):
    def test_exact_contract_and_mask(self):
        env = MockPvZEnv(4); obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape, (5534,)); self.assertEqual(env.action_space.n, 541)
        self.assertEqual(env.action_masks().dtype, np.bool_)

    def test_masked_action_rejected(self):
        env = MockPvZEnv(); env.reset()
        with self.assertRaises(ValueError): env.step(540)

    def test_evaluation_statistics(self):
        env = MockPvZEnv(3)
        records, summary = evaluate_masked(env, lambda _o, _m: 0, 3, seed=9)
        self.assertEqual(len(records), 3); self.assertEqual(summary["episodes"], 3)
        self.assertTrue(summary["mock"]); self.assertEqual(summary["technical_truncations"], 0)
