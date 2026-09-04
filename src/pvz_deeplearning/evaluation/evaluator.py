"""Independent masked-policy evaluation and uncertainty summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import statistics
import time
from typing import Any, Callable


@dataclass(frozen=True)
class EvaluationEpisode:
    episode: int
    episode_return: float
    length: int
    terminated: bool
    truncated: bool
    technical_truncation: bool
    waves_reached: int
    wall_seconds: float


def evaluate_masked(env: Any, select_action: Callable[[Any, Any], int], episodes: int, seed: int = 0) -> tuple[list[EvaluationEpisode], dict[str, Any]]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    records: list[EvaluationEpisode] = []
    for episode in range(episodes):
        started = time.monotonic()
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        length = wave = 0
        while True:
            action = select_action(obs, env.action_masks())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            length += 1
            wave = max(wave, int(info.get("wave") or 0))
            if terminated or truncated:
                records.append(EvaluationEpisode(episode, total, length, terminated, truncated,
                    bool(info.get("technical_truncation")), wave, time.monotonic() - started))
                break
    returns = [x.episode_return for x in records]
    natural = [x for x in records if not x.technical_truncation]
    mean = statistics.fmean(returns)
    std = statistics.stdev(returns) if len(returns) > 1 else 0.0
    ci = 1.96 * std / math.sqrt(len(returns))
    summary = {
        "episodes": len(records), "mean_return": mean, "median_return": statistics.median(returns),
        "std_return": std, "min_return": min(returns), "max_return": max(returns),
        "return_95ci": [mean - ci, mean + ci], "mean_waves": statistics.fmean(x.waves_reached for x in records),
        "technical_truncations": sum(x.technical_truncation for x in records),
        "technical_truncation_rate": sum(x.technical_truncation for x in records) / len(records),
        "termination_rate": sum(x.terminated for x in natural) / len(natural) if natural else None,
        "mock": bool(getattr(env, "__class__", type(env)).__name__.startswith("Mock")),
    }
    return records, summary


def serializable_evaluation(records: list[EvaluationEpisode], summary: dict[str, Any]) -> dict[str, Any]:
    return {"summary": summary, "episodes": [asdict(x) for x in records]}
