import unittest
from types import SimpleNamespace

from pvz_deeplearning.config import LevelProfile
from pvz_deeplearning.live import (
    LiveEnvironmentError, RecoveryAction, RecoveryPolicy, build_live_environment,
    validate_level_profile,
)


def profile(**changes):
    values = dict(id="level", adventure_level=7, active_rows=(True,) * 5 + (False,),
        step_interval_seconds=.25, max_episode_steps=10, auto_collect_pickups=True,
        reset_strategy="managed_current_level", expected_seed_types=("Peashooter", "Sunflower"),
        reward_profile="harness_reward_v1", evaluation_metric="waves")
    values.update(changes)
    return LevelProfile(**values)


def game_state(level=7, seeds=(0, 1)):
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


class FakeTrainingEpisodeSupport:
    def __init__(self, runtime, **kwargs):
        self.runtime = runtime
        self.pickups = SimpleNamespace(enabled=kwargs["auto_collect_pickups"],
            collect_once=lambda: None, shutdown=lambda: None)

    def shutdown(self): self.pickups.shutdown()


class FakeResetExpectation:
    def __init__(self, adventure_level, seed_types):
        self.adventure_level = adventure_level
        self.seed_types = seed_types


FAKE_TRAINING_API = SimpleNamespace(
    PvZRuntime=FakeRuntime,
    ResetExpectation=FakeResetExpectation,
    ResetStatus=SimpleNamespace(NOT_ATTACHED="not_attached", UNHEALTHY="unhealthy"),
    TrainingEpisodeSupport=FakeTrainingEpisodeSupport,
)


class LiveFactoryTests(unittest.TestCase):
    def test_level_and_seed_mismatch_refuse(self):
        with self.assertRaisesRegex(LiveEnvironmentError, "configured level"):
            validate_level_profile(game_state(level=6), profile())
        with self.assertRaisesRegex(LiveEnvironmentError, "configured seeds"):
            validate_level_profile(game_state(seeds=(0,)), profile())

    def test_factory_composes_public_harness_seams(self):
        runtime = FakeRuntime(game_state())
        bundle = build_live_environment(profile(), runtime_factory=lambda: runtime,
            training_api=FAKE_TRAINING_API)
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
