import unittest
from types import SimpleNamespace

from pvz_deeplearning.config import LevelProfile
from pvz_deeplearning.live import (
    LiveEnvironmentError, RecoveryAction, RecoveryPolicy, build_live_environment,
    validate_level_profile,
)


def profile(**changes):
    values = dict(id="level", adventure_level=5, active_rows=(True,) * 5 + (False,),
        step_interval_seconds=.25, max_episode_steps=10, auto_collect_pickups=True,
        reset_strategy="managed_current_level", expected_seed_types=("Peashooter", "Sunflower"),
        reward_profile="harness_reward_v1", evaluation_metric="waves")
    values.update(changes)
    return LevelProfile(**values)


def game_state(level=5, seeds=(0, 1)):
    return SimpleNamespace(adventure_level=level, paused=False,
        seeds=[SimpleNamespace(type_id=x) for x in seeds])


class FakeRuntime:
    def __init__(self, state):
        self.state = state
        self.health = SimpleNamespace(can_observe=True)
        self.closed = False

    def attach(self): return None
    def observe(self): return self.state
    def reader_adapter(self): return SimpleNamespace(read=lambda: self.state)
    def controller_adapter(self): return SimpleNamespace(plant=lambda *args: None)
    def close(self): self.closed = True


class LiveFactoryTests(unittest.TestCase):
    def test_release_gate_prevents_live_construction(self):
        with self.assertRaisesRegex(LiveEnvironmentError, "immutable harness"):
            build_live_environment(profile(), runtime_factory=lambda: self.fail("must not attach"))

    def test_level_and_seed_mismatch_refuse(self):
        with self.assertRaisesRegex(LiveEnvironmentError, "configured level"):
            validate_level_profile(game_state(level=6), profile())
        with self.assertRaisesRegex(LiveEnvironmentError, "configured seeds"):
            validate_level_profile(game_state(seeds=(0,)), profile())

    def test_factory_composes_public_harness_seams_under_test_bypass(self):
        runtime = FakeRuntime(game_state())
        bundle = build_live_environment(profile(), runtime_factory=lambda: runtime,
            allow_unreleased_for_tests=True)
        self.assertIs(bundle.runtime, runtime)
        self.assertTrue(bundle.episode_support.pickups.enabled)
        bundle.close()
        self.assertTrue(runtime.closed)

    def test_recovery_is_bounded_and_resets_after_success(self):
        policy = RecoveryPolicy(max_reattach_attempts=2)
        self.assertEqual(policy.on_technical_interruption(), RecoveryAction.REATTACH_AND_RESET)
        self.assertEqual(policy.on_technical_interruption(), RecoveryAction.REATTACH_AND_RESET)
        self.assertEqual(policy.on_technical_interruption(), RecoveryAction.STOP_CLEANLY)
        policy.recovered()
        self.assertEqual(policy.attempts, 0)
