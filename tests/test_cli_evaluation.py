"""Offline tests for explicit mock/live evaluation selection."""

from argparse import Namespace
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pvz_deeplearning import cli
from pvz_deeplearning.mock_env import MockPvZEnv


def arguments(**changes):
    values = dict(policy="random-valid", checkpoint=None, episodes=1, seed=3,
                  device="cpu", config=None, yes=False, output=None)
    values.update(changes)
    return Namespace(**values)


class Bundle:
    def __init__(self):
        self.gym_environment = MockPvZEnv(1)
        self.closed = False

    def close(self):
        self.closed = True


class CliEvaluationTests(unittest.TestCase):
    def test_mock_evaluation_remains_default_and_needs_no_confirmation(self):
        self.assertEqual(cli.command_evaluate(arguments()), 0)

    def test_live_evaluation_requires_yes_before_factory(self):
        experiment = SimpleNamespace(mode="live")
        with patch.object(cli, "_resolve_config", return_value=(experiment, None, None)):
            with self.assertRaisesRegex(SystemExit, "requires --yes"):
                cli.command_evaluate(arguments(config="live.yaml"))

    def test_live_evaluation_uses_factory_and_closes_bundle(self):
        experiment = SimpleNamespace(mode="live")
        bundle = Bundle()
        with patch.object(cli, "_resolve_config", return_value=(experiment, None, None)):
            with patch("pvz_deeplearning.live.build_live_environment", return_value=bundle) as factory:
                self.assertEqual(cli.command_evaluate(arguments(config="live.yaml", yes=True)), 0)
        factory.assert_called_once_with(None)
        self.assertTrue(bundle.closed)

    def test_live_tuning_message_is_not_a_harness_version_claim(self):
        experiment = SimpleNamespace(mode="live")
        with patch.object(cli, "_resolve_config", return_value=(experiment, None, None)):
            with self.assertRaisesRegex(SystemExit, "validated real training/evaluation pilot"):
                cli.command_tune(Namespace(config="live.yaml", storage="x", study="x", trials=1))

