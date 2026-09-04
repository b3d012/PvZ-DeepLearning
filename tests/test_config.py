import tempfile
import unittest
from pathlib import Path

from pvz_deeplearning.config import LevelProfile, load_experiment, load_level, load_model


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_tracked_configs_validate(self):
        self.assertEqual(load_level(ROOT / "configs/levels/adventure_1_5.yaml").adventure_level, 5)
        self.assertEqual(load_model(ROOT / "configs/models/mlp_small.yaml").net_arch, (128, 128))
        self.assertEqual(load_experiment(ROOT / "configs/experiments/mock_smoke.yaml").mode, "mock")

    def test_unknown_reset_strategy_rejected(self):
        with self.assertRaises(ValueError):
            LevelProfile("x", 1, (True,) * 6, .25, 2, True, "blind_clicks", (), "r", "m")

    def test_bad_yaml_shape_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.yaml"
            path.write_text("- not\n- a mapping\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_level(path)
