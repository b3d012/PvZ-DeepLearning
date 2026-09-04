"""Bounded MaskablePPO training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import time
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


class SafetyStopCallback(BaseCallback):
    """Enforce wall/episode limits and append lightweight local metrics."""

    def __init__(self, metrics_path: Path, *, max_episodes: int | None, max_wall_seconds: float | None) -> None:
        super().__init__()
        self.metrics_path = metrics_path
        self.max_episodes = max_episodes
        self.max_wall_seconds = max_wall_seconds
        self.started = time.monotonic()
        self.episodes = 0
        self.stop_requested = False

    def _on_training_start(self) -> None:
        try:
            signal.signal(signal.SIGINT, lambda *_: setattr(self, "stop_requested", True))
        except ValueError:
            # Python only permits signal handlers on the main thread. A GUI
            # worker can still request a stop by setting ``stop_requested``.
            pass

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", ())
        self.episodes += sum(bool(x) for x in dones)
        elapsed = time.monotonic() - self.started
        if self.num_timesteps % 16 == 0 or any(dones):
            payload = {"step": self.num_timesteps, "episodes": self.episodes, "wall_seconds": elapsed,
                       "steps_per_hour": self.num_timesteps / elapsed * 3600 if elapsed else 0.0}
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        return not (self.stop_requested or (self.max_episodes is not None and self.episodes >= self.max_episodes)
                    or (self.max_wall_seconds is not None and elapsed >= self.max_wall_seconds))


def train_model(backend: Any, model: Any, *, total_timesteps: int, run_path: Path,
                checkpoint_interval: int, max_episodes: int | None = None,
                max_wall_seconds: float | None = None, reset_num_timesteps: bool = True) -> Path:
    safety = SafetyStopCallback(run_path / "metrics" / "training.jsonl", max_episodes=max_episodes,
                                max_wall_seconds=max_wall_seconds)
    checkpoints = CheckpointCallback(save_freq=checkpoint_interval,
        save_path=str(run_path / "checkpoints"), name_prefix="step")
    try:
        backend.learn(model, total_timesteps, callback=[safety, checkpoints], reset_num_timesteps=reset_num_timesteps)
    finally:
        backend.save(model, run_path / "checkpoints" / "latest")
    return run_path / "checkpoints" / "latest.zip"
