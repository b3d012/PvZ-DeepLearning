"""Sequential local Optuna tuning with held-out masked evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def suggest_maskable_ppo(trial: Any) -> dict[str, Any]:
    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128])
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    if n_steps % batch_size:
        raise ValueError("n_steps must be divisible by batch_size for one environment")
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.999),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.90, 0.99),
        "n_steps": n_steps, "batch_size": batch_size,
        "ent_coef": trial.suggest_float("ent_coef", 1e-5, 0.02, log=True),
    }


def run_study(storage_path: str | Path, study_name: str, trials: int,
              objective: Callable[[Any], float]) -> Any:
    import optuna
    if trials <= 0:
        raise ValueError("trials must be positive")
    target = Path(storage_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=study_name, direction="maximize",
        storage=f"sqlite:///{target.resolve().as_posix()}", load_if_exists=True)
    study.optimize(objective, n_trials=trials, n_jobs=1)
    return study
