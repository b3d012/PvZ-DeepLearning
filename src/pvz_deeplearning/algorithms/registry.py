"""Extensible algorithm registry with a MaskablePPO first baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np


class AlgorithmBackend(Protocol):
    id: str
    def build(self, environment: Any, *, model_config: Any, seed: int, device: str, hyperparameters: dict[str, Any]) -> Any: ...
    def learn(self, model: Any, total_timesteps: int, **kwargs: Any) -> Any: ...
    def predict(self, model: Any, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool = True) -> int: ...
    def save(self, model: Any, path: str | Path) -> Path: ...
    def load(self, path: str | Path, *, environment: Any = None, device: str = "auto") -> Any: ...


class MaskablePPOBackend:
    id = "maskable_ppo"

    def build(self, environment: Any, *, model_config: Any, seed: int, device: str, hyperparameters: dict[str, Any]) -> Any:
        import torch.nn as nn
        from sb3_contrib import MaskablePPO

        activation = {"tanh": nn.Tanh, "relu": nn.ReLU}[model_config.activation_fn]
        options = dict(hyperparameters)
        options.setdefault("policy_kwargs", {"net_arch": list(model_config.net_arch), "activation_fn": activation})
        return MaskablePPO("MlpPolicy", environment, seed=seed, device=device, **options)

    def learn(self, model: Any, total_timesteps: int, **kwargs: Any) -> Any:
        return model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, model: Any, observation: np.ndarray, action_mask: np.ndarray, *, deterministic: bool = True) -> int:
        action, _ = model.predict(observation, action_masks=action_mask, deterministic=deterministic)
        return int(np.asarray(action).item())

    def save(self, model: Any, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        model.save(target)
        return target.with_suffix(".zip") if target.suffix != ".zip" else target

    def load(self, path: str | Path, *, environment: Any = None, device: str = "auto") -> Any:
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, env=environment, device=device)


_BACKENDS: dict[str, AlgorithmBackend] = {"maskable_ppo": MaskablePPOBackend()}


def get_backend(name: str) -> AlgorithmBackend:
    try:
        return _BACKENDS[name]
    except KeyError as error:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(_BACKENDS)}") from error


def registered_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_BACKENDS))
