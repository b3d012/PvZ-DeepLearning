"""Immutable run manifests, directories, lineage, and checkpoint identity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

from pvz_deeplearning import __version__
from pvz_deeplearning.harness import HARNESS_RELEASE, assert_supported_harness_contract


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    created_at: str
    phase4_version: str
    phase4_git_sha: str
    dirty_worktree: bool
    harness_release: str
    harness_resolved_sha: str
    game_version: str
    environment_contract: dict[str, Any]
    level_profile: dict[str, Any]
    algorithm: str
    algorithm_library: str
    algorithm_library_version: str
    model_architecture: dict[str, Any]
    hyperparameters: dict[str, Any]
    seeds: dict[str, int]
    game_rng_controlled: bool
    step_interval: float
    training_speed: float
    training_budget: dict[str, Any]
    starting_checkpoint: str | None
    parent_run_id: str | None
    python_version: str
    torch_version: str
    cuda_version: str | None
    device: str
    gpu_name: str | None
    evaluation_protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_state(path: str | Path) -> tuple[str, bool]:
    root = str(Path(path).resolve())
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip())
    return sha, dirty


def harness_sha() -> str:
    try:
        import pvz_env
        root = Path(pvz_env.__file__).resolve().parent.parent
        return git_state(root)[0]
    except Exception:
        return "installed-distribution-unknown"


def new_manifest(*, repository: str | Path, level: Any, experiment: Any, model: Any, parent_run_id: str | None = None, starting_checkpoint: str | None = None) -> RunManifest:
    import torch
    sha, dirty = git_state(repository)
    stamp = datetime.now(UTC)
    run_id = f"{stamp:%Y%m%dT%H%M%SZ}-{experiment.id}-{sha[:8]}"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    seeds = {name: experiment.seed for name in ("python", "numpy", "torch", "framework", "policy")}
    return RunManifest(
        schema_version=1, run_id=run_id, created_at=stamp.isoformat(), phase4_version=__version__,
        phase4_git_sha=sha, dirty_worktree=dirty, harness_release=HARNESS_RELEASE,
        harness_resolved_sha=harness_sha(), game_version="GOTY 1.2.0.1073",
        environment_contract=assert_supported_harness_contract(), level_profile=asdict(level),
        algorithm=experiment.algorithm, algorithm_library="sb3-contrib",
        algorithm_library_version=importlib.metadata.version("sb3-contrib"), model_architecture=asdict(model),
        hyperparameters=experiment.hyperparameters, seeds=seeds, game_rng_controlled=False,
        step_interval=level.step_interval_seconds, training_speed=level.training_speed,
        training_budget={"total_timesteps": experiment.total_timesteps, "max_episodes": experiment.max_episodes,
                         "max_wall_time_seconds": experiment.max_wall_time_seconds},
        starting_checkpoint=starting_checkpoint, parent_run_id=parent_run_id,
        python_version=platform.python_version(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=experiment.device, gpu_name=gpu,
        evaluation_protocol={"deterministic": True, "technical_truncations_excluded_from_win_rate": True},
    )


class RunDirectory:
    def __init__(self, root: str | Path, manifest: RunManifest) -> None:
        self.path = Path(root) / manifest.run_id
        self.manifest = manifest

    def create(self, resolved_config: dict[str, Any]) -> Path:
        self.path.mkdir(parents=True, exist_ok=False)
        for name in ("checkpoints", "metrics", "tensorboard", "evaluation", "transitions"):
            (self.path / name).mkdir()
        self.write_once("manifest.json", self.manifest.to_dict())
        import yaml
        (self.path / "config.resolved.yaml").write_text(yaml.safe_dump(resolved_config, sort_keys=True), encoding="utf-8")
        return self.path

    def write_once(self, relative: str, value: Any) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return target


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resume(manifest: dict[str, Any], *, algorithm: str, model_id: str) -> None:
    assert_supported_harness_contract()
    if manifest["algorithm"] != algorithm:
        raise ValueError("resume algorithm mismatch")
    if manifest["model_architecture"]["id"] != model_id:
        raise ValueError("resume model architecture mismatch")
    if manifest["environment_contract"] != assert_supported_harness_contract():
        raise ValueError("resume harness contract mismatch")
