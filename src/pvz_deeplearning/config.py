"""Strict tracked configuration loading for Phase 4 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LevelProfile:
    id: str
    adventure_level: int
    active_rows: tuple[bool, ...]
    step_interval_seconds: float
    max_episode_steps: int
    auto_collect_pickups: bool
    reset_strategy: str
    expected_seed_types: tuple[str, ...]
    reward_profile: str
    evaluation_metric: str
    training_speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or self.adventure_level <= 0:
            raise ValueError("level id must be non-empty and adventure_level positive")
        if len(self.active_rows) != 6 or not all(isinstance(x, bool) for x in self.active_rows):
            raise ValueError("active_rows must contain exactly six booleans")
        if self.step_interval_seconds <= 0 or self.max_episode_steps <= 0:
            raise ValueError("step interval and episode limit must be positive")
        if self.training_speed != 1.0:
            raise ValueError("the validated Adventure 1-7 profile requires training_speed 1.0")
        if self.reset_strategy not in {"operator_prepared", "managed_current_level"}:
            raise ValueError("reset_strategy must be operator_prepared or managed_current_level")


@dataclass(frozen=True)
class ModelConfig:
    id: str
    net_arch: tuple[int, ...]
    activation_fn: str = "tanh"

    def __post_init__(self) -> None:
        if not self.id or not self.net_arch or any(x <= 0 for x in self.net_arch):
            raise ValueError("model id and positive net_arch widths are required")
        if self.activation_fn not in {"tanh", "relu"}:
            raise ValueError("activation_fn must be tanh or relu")


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    mode: str
    algorithm: str
    level: str
    model: str
    seed: int
    total_timesteps: int
    checkpoint_interval: int
    device: str = "auto"
    max_episodes: int | None = None
    max_wall_time_seconds: float | None = None
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"mock", "live"}:
            raise ValueError("mode must be mock or live")
        if self.algorithm != "maskable_ppo":
            raise ValueError("unsupported algorithm")
        if self.total_timesteps <= 0 or self.checkpoint_interval <= 0:
            raise ValueError("training budgets must be positive")
        if self.max_episodes is None and self.max_wall_time_seconds is None and self.total_timesteps <= 0:
            raise ValueError("at least one stop condition is required")


def _load(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def load_level(path: str | Path) -> LevelProfile:
    data = _load(path)
    data["active_rows"] = tuple(data["active_rows"])
    data["expected_seed_types"] = tuple(data.get("expected_seed_types", ()))
    return LevelProfile(**data)


def load_model(path: str | Path) -> ModelConfig:
    data = _load(path)
    data["net_arch"] = tuple(data["net_arch"])
    return ModelConfig(**data)


def load_experiment(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig(**_load(path))


def resolved_dict(config: Any) -> dict[str, Any]:
    return asdict(config)
