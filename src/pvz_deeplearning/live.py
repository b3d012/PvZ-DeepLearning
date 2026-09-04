"""Release-gated construction of the real harness Environment v1 backend."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable
import uuid

from pvz_env import EpisodeConfig, EpisodeMetadata, PvZEnvironment
from pvz_env.rewards import OutcomeReason
from pvz_reader.game_state import PLANT_NAMES

from pvz_deeplearning.adapters import PvZGymEnv
from pvz_deeplearning.harness import HARNESS_RELEASE, assert_supported_harness_contract


TRAINING_SUPPORT_RELEASE = "v0.2.0"


class LiveEnvironmentError(RuntimeError):
    pass


class RecoveryAction(str, Enum):
    REATTACH_AND_RESET = "reattach_and_reset"
    STOP_CLEANLY = "stop_cleanly"


@dataclass
class RecoveryPolicy:
    max_reattach_attempts: int = 3
    attempts: int = 0

    def on_technical_interruption(self) -> RecoveryAction:
        self.attempts += 1
        return (RecoveryAction.REATTACH_AND_RESET
                if self.attempts <= self.max_reattach_attempts else RecoveryAction.STOP_CLEANLY)

    def recovered(self) -> None:
        self.attempts = 0


class RuntimeOutcomeDetector:
    def __init__(self, support: Any) -> None:
        self.support = support

    def reset(self, episode_config: Any, initial_state: Any) -> None:
        return None

    def detect(self, before: Any, after: Any) -> OutcomeReason | None:
        from pvz_runtime import GameOutcome
        outcome = self.support.outcome().outcome
        if outcome is GameOutcome.WON:
            return OutcomeReason.WIN
        if outcome is GameOutcome.LOST:
            return OutcomeReason.LOSS
        return None


@dataclass
class LiveEnvironmentBundle:
    gym_environment: PvZGymEnv
    environment: PvZEnvironment
    runtime: Any
    episode_support: Any
    recovery: RecoveryPolicy

    def close(self) -> None:
        self.episode_support.shutdown()
        self.runtime.close()


def _seed_ids(names: tuple[str, ...]) -> tuple[int, ...]:
    try:
        return tuple(PLANT_NAMES.index(name) for name in names)
    except ValueError as error:
        raise LiveEnvironmentError(f"unknown expected seed type: {error}") from error


def validate_level_profile(state: Any, level: Any) -> None:
    if int(state.adventure_level) != int(level.adventure_level):
        raise LiveEnvironmentError(
            f"configured level {level.adventure_level} != observed level {state.adventure_level}"
        )
    expected = _seed_ids(tuple(level.expected_seed_types))
    observed = tuple(int(seed.type_id) for seed in state.seeds)
    if expected and observed != expected:
        raise LiveEnvironmentError(f"configured seeds {expected} != observed seeds {observed}")
    if bool(state.paused):
        raise LiveEnvironmentError("prepared level is paused")


def build_live_environment(
    level: Any,
    *,
    runtime_factory: Callable[[], Any] | None = None,
    restart_driver: Any | None = None,
    allow_unreleased_for_tests: bool = False,
    training_api: Any | None = None,
) -> LiveEnvironmentBundle:
    """Construct the real backend; no input occurs until Gym reset/step.

    The test-only bypass is intentionally not exposed by the CLI. Durable live
    runs require the immutable training-support release pin.
    """
    if HARNESS_RELEASE != TRAINING_SUPPORT_RELEASE and not allow_unreleased_for_tests:
        raise LiveEnvironmentError(
            f"live training requires immutable harness {TRAINING_SUPPORT_RELEASE}; "
            f"Phase 4 is pinned to {HARNESS_RELEASE}"
        )
    if level.reset_strategy != "managed_current_level" or not level.auto_collect_pickups:
        raise LiveEnvironmentError("live profile must enable managed_current_level reset and pickups")
    assert_supported_harness_contract()
    if training_api is None:
        from pvz_runtime import PvZRuntime, ResetExpectation, ResetStatus, TrainingEpisodeSupport
    else:
        PvZRuntime = training_api.PvZRuntime
        ResetExpectation = training_api.ResetExpectation
        ResetStatus = training_api.ResetStatus
        TrainingEpisodeSupport = training_api.TrainingEpisodeSupport

    runtime = runtime_factory() if runtime_factory else PvZRuntime()
    runtime.attach()
    state = runtime.observe()
    if state is None or not runtime.health.can_observe:
        runtime.close()
        raise LiveEnvironmentError("runtime cannot observe a prepared Board")
    validate_level_profile(state, level)
    support = TrainingEpisodeSupport(runtime, restart_driver=restart_driver, auto_collect_pickups=True)
    detector = RuntimeOutcomeDetector(support)
    expectation = ResetExpectation(level.adventure_level, _seed_ids(level.expected_seed_types))

    def episode_factory() -> EpisodeConfig:
        return EpisodeConfig(
            episode_id=f"{level.id}-{uuid.uuid4().hex[:12]}",
            active_rows=level.active_rows,
            step_interval_seconds=level.step_interval_seconds,
            max_steps=level.max_episode_steps,
            max_consecutive_state_unavailable=1,
            terminal_detector=detector,
            metadata=EpisodeMetadata(label=level.id, adventure_level=level.adventure_level),
        )

    recovery = RecoveryPolicy()

    def prepare_reset(_config: EpisodeConfig) -> None:
        while True:
            result = support.reset_current_level(expectation)
            if result.success:
                break
            if result.status not in (ResetStatus.NOT_ATTACHED, ResetStatus.UNHEALTHY):
                raise LiveEnvironmentError(f"verified reset failed: {result.status.value}:{result.reason}")
            if recovery.on_technical_interruption() is RecoveryAction.STOP_CLEANLY:
                raise LiveEnvironmentError("bounded process recovery exhausted")
            runtime.reattach()
        fresh = runtime.observe()
        validate_level_profile(fresh, level)
        recovery.recovered()

    environment = PvZEnvironment(runtime.reader_adapter(), runtime.controller_adapter())
    gym_environment = PvZGymEnv(
        environment, episode_factory, prepare_reset=prepare_reset,
        before_step=support.pickups.collect_once,
    )
    return LiveEnvironmentBundle(gym_environment, environment, runtime, support, recovery)
