import unittest
from types import SimpleNamespace

from pvz_deeplearning.training import extract_sb3_metrics


class TrainingMetricsTests(unittest.TestCase):
    def test_stable_logger_values_are_mapped_and_non_numeric_ignored(self):
        logger = SimpleNamespace(name_to_value={
            "train/policy_gradient_loss": -0.1,
            "train/value_loss": 0.2,
            "train/approx_kl": "0.03",
            "train/learning_rate": object(),
            "unrelated": 4,
        })
        self.assertEqual(extract_sb3_metrics(logger), {
            "policy_gradient_loss": -0.1, "value_loss": 0.2, "approx_kl": 0.03,
        })
