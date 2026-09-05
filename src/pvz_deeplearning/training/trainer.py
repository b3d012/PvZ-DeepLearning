"""Bounded MaskablePPO training orchestration."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import time
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


SB3_METRIC_KEYS = {
    "train/policy_gradient_loss": "policy_gradient_loss",
    "train/value_loss": "value_loss",
    "train/entropy_loss": "entropy_loss",
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/explained_variance": "explained_variance",
    "train/learning_rate": "learning_rate",
}


def extract_sb3_metrics(logger: Any) -> dict[str, float]:
    values = getattr(logger, "name_to_value", {}) or {}
    result: dict[str, float] = {}
    for source, target in SB3_METRIC_KEYS.items():
        value = values.get(source)
        if value is not None:
            try:
                result[target] = float(value)
            except (TypeError, ValueError):
                continue
    return result


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
        self.completion_reason = "total_timesteps"
        self.technical_truncations = 0
        self.wins = 0
        self.losses = 0
        self.episode_returns: list[float] = []

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
        for info in self.locals.get("infos", ()):
            self.technical_truncations += int(bool(info.get("technical_truncation")))
            self.wins += int(info.get("outcome_reason") == "win")
            self.losses += int(info.get("outcome_reason") == "loss")
            if "episode" in info and "r" in info["episode"]:
                self.episode_returns.append(float(info["episode"]["r"]))
        elapsed = time.monotonic() - self.started
        if self.num_timesteps % 16 == 0 or any(dones):
            payload = {"step": self.num_timesteps, "episodes": self.episodes,
                       "wins": self.wins, "losses": self.losses,
                       "technical_truncations": self.technical_truncations,
                       "wall_seconds": elapsed,
                       "steps_per_hour": self.num_timesteps / elapsed * 3600 if elapsed else 0.0,
                       "latest_episode_return": self.episode_returns[-1] if self.episode_returns else None,
                       "rolling_mean_return": (sum(self.episode_returns[-20:]) / len(self.episode_returns[-20:])
                                               if self.episode_returns else None)}
            payload.update(extract_sb3_metrics(self.logger))
            with self.metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, sort_keys=True) + "\n")
        if self.stop_requested:
            self.completion_reason = "operator_stop"
            return False
        if self.max_episodes is not None and self.episodes >= self.max_episodes:
            self.completion_reason = "max_episodes"
            return False
        if self.max_wall_seconds is not None and elapsed >= self.max_wall_seconds:
            self.completion_reason = "max_wall_time"
            return False
        return True


def train_model(backend: Any, model: Any, *, total_timesteps: int, run_path: Path,
                checkpoint_interval: int, max_episodes: int | None = None,
                max_wall_seconds: float | None = None, reset_num_timesteps: bool = True) -> tuple[Path, str]:
    safety = SafetyStopCallback(run_path / "metrics" / "training.jsonl", max_episodes=max_episodes,
                                max_wall_seconds=max_wall_seconds)
    checkpoints = CheckpointCallback(save_freq=checkpoint_interval,
        save_path=str(run_path / "checkpoints"), name_prefix="step")
    try:
        backend.learn(model, total_timesteps, callback=[safety, checkpoints], reset_num_timesteps=reset_num_timesteps)
    finally:
        backend.save(model, run_path / "checkpoints" / "latest")
    return run_path / "checkpoints" / "latest.zip", safety.completion_reason
