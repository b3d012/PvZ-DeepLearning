import unittest

from pvz_deeplearning.tuning import suggest_maskable_ppo


class FakeTrial:
    def suggest_categorical(self, name, values): return values[0]
    def suggest_float(self, name, low, high, log=False): return low


class TuningTests(unittest.TestCase):
    def test_search_space_is_valid(self):
        params = suggest_maskable_ppo(FakeTrial())
        self.assertEqual(params["n_steps"] % params["batch_size"], 0)
        self.assertGreater(params["learning_rate"], 0)
