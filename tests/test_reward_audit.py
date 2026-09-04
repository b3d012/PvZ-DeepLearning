import unittest

from pvz_deeplearning.reward_audit import audit_records


class RewardAuditTests(unittest.TestCase):
    def test_component_totals_and_actions(self):
        report = audit_records([
            {"episode_id": "a", "step_index": 0, "reward": {"reward": .1, "components": {"wave": .1}}, "action": {"action_type": "wait"}},
            {"episode_id": "a", "step_index": 1, "reward": {"reward": 1., "components": {"terminal": 1.}}, "action": {"action_type": "plant"}},
        ])
        self.assertAlmostEqual(report["total_reward"], 1.1)
        self.assertEqual(report["components"], {"terminal": 1.0, "wave": .1})
        self.assertEqual(report["actions"], {"wait": 1, "plant": 1})

    def test_flat_harness_transition_v2(self):
        report = audit_records([{"episode_id": "v2", "step_index": 0, "reward": .01,
            "reward_components": {"wave_progress": .01}, "action": {"action_type": "wait"}}])
        self.assertEqual(report["components"], {"wave_progress": .01})
