import tempfile
import unittest
from pathlib import Path

from pvz_deeplearning.algorithms import get_backend
from pvz_deeplearning.config import ModelConfig
from pvz_deeplearning.mock_env import MockPvZEnv


class AlgorithmSmokeTests(unittest.TestCase):
    def test_build_learn_save_load_predict_with_mask(self):
        env = MockPvZEnv(8); backend = get_backend("maskable_ppo")
        config = ModelConfig("tiny", (16,))
        model = backend.build(env, model_config=config, seed=1, device="cpu",
            hyperparameters={"n_steps": 8, "batch_size": 4, "n_epochs": 1, "verbose": 0})
        backend.learn(model, 16)
        obs, _ = env.reset(seed=2)
        action = backend.predict(model, obs, env.action_masks())
        self.assertTrue(env.action_masks()[action])
        with tempfile.TemporaryDirectory() as folder:
            path = backend.save(model, Path(folder) / "model")
            loaded = backend.load(path, environment=env, device="cpu")
            self.assertTrue(env.action_masks()[backend.predict(loaded, obs, env.action_masks())])
