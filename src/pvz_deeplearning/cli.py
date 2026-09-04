"""`pvz-dl` command line interface. Live mutation is always explicit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
from typing import Any

from pvz_deeplearning.algorithms import get_backend, registered_algorithms
from pvz_deeplearning.config import load_experiment, load_level, load_model
from pvz_deeplearning.evaluation import HarnessBaselineSelector, evaluate_masked, serializable_evaluation
from pvz_deeplearning.harness import HARNESS_RELEASE, assert_supported_harness_contract
from pvz_deeplearning.mock_env import MockPvZEnv
from pvz_deeplearning.reward_audit import audit_records, load_jsonl
from pvz_deeplearning.runs import RunDirectory, new_manifest, validate_resume
from pvz_deeplearning.training import train_model
from pvz_deeplearning.tuning import run_study, suggest_maskable_ppo


ROOT = Path(__file__).resolve().parents[2]


def _resolve_config(experiment_path: str | Path) -> tuple[Any, Any, Any]:
    experiment = load_experiment(experiment_path)
    level = load_level(ROOT / "configs" / "levels" / f"{experiment.level}.yaml")
    model = load_model(ROOT / "configs" / "models" / f"{experiment.model}.yaml")
    return experiment, level, model


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def command_doctor(_: argparse.Namespace) -> int:
    report: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable,
        "harness_release": HARNESS_RELEASE, "algorithms": registered_algorithms(), "gameplay_actions_sent": False}
    try:
        report["contract"] = assert_supported_harness_contract()
        report["contract_ok"] = True
    except Exception as error:
        report["contract_ok"] = False
        report["contract_error"] = str(error)
    try:
        import torch
        report.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except ImportError:
        report["torch"] = None
    try:
        from pvz_runtime import PvZRuntime, RuntimeConfig
        runtime = PvZRuntime(config=RuntimeConfig(observer_only=True))
        report["runtime"] = runtime.refresh().to_dict()
        runtime.detach()
    except Exception as error:
        report["runtime"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
    report["live_training_blockers"] = ["authoritative_terminal_detection", "automatic_current_level_reset",
        "environment_managed_pickup_collection"]
    _print(report)
    return 0 if report.get("contract_ok") else 1


def command_train(args: argparse.Namespace) -> int:
    experiment, level, model_config = _resolve_config(args.config)
    if experiment.mode == "live":
        if not args.yes:
            raise SystemExit("live control requires --yes")
        raise SystemExit("live multi-episode training is blocked: harness v0.1.0 has no authoritative reset/terminal/pickup service")
    env = MockPvZEnv()
    backend = get_backend(experiment.algorithm)
    parent_run_id = None
    if args.resume:
        source_run = Path(args.resume).resolve().parent.parent
        parent = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
        validate_resume(parent, algorithm=experiment.algorithm, model_id=model_config.id)
        parent_run_id = parent["run_id"]
    manifest = new_manifest(repository=ROOT, level=level, experiment=experiment, model=model_config,
        parent_run_id=parent_run_id, starting_checkpoint=str(Path(args.resume).resolve()) if args.resume else None)
    resolved = {"experiment": asdict(experiment), "level": asdict(level), "model": asdict(model_config), "mock": True}
    run = RunDirectory(args.output, manifest)
    run_path = run.create(resolved)
    _print({"run_id": manifest.run_id, "mode": experiment.mode, "level": level.id,
            "algorithm": experiment.algorithm, "model": model_config.id, "harness": HARNESS_RELEASE,
            "reward": level.reward_profile, "step_interval": level.step_interval_seconds,
            "device": experiment.device, "budget": manifest.training_budget, "output": str(run_path)})
    model = (backend.load(args.resume, environment=env, device=experiment.device) if args.resume else
        backend.build(env, model_config=model_config, seed=experiment.seed,
                      device=experiment.device, hyperparameters=experiment.hyperparameters))
    checkpoint = train_model(backend, model, total_timesteps=experiment.total_timesteps, run_path=run_path,
        checkpoint_interval=experiment.checkpoint_interval, max_episodes=experiment.max_episodes,
        max_wall_seconds=experiment.max_wall_time_seconds, reset_num_timesteps=not bool(args.resume))
    _print({"run_id": manifest.run_id, "checkpoint": str(checkpoint), "result_class": "MOCK"})
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    env = MockPvZEnv()
    if args.policy in {"random-valid", "scripted-heuristic"}:
        select = HarnessBaselineSelector(args.policy, env, args.seed)
    else:
        backend = get_backend("maskable_ppo")
        model = backend.load(args.checkpoint, environment=env, device=args.device)
        select = lambda obs, mask: backend.predict(model, obs, mask, deterministic=True)
    records, summary = evaluate_masked(env, select, args.episodes, args.seed)
    payload = serializable_evaluation(records, summary)
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _print(payload)
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    root = Path(args.run)
    payload = {"manifest": json.loads((root / "manifest.json").read_text(encoding="utf-8"))}
    metrics = root / "metrics" / "training.jsonl"
    payload["metrics"] = load_jsonl(metrics) if metrics.exists() else []
    payload["checkpoints"] = [str(x) for x in sorted((root / "checkpoints").glob("*.zip"))]
    _print(payload)
    return 0


def command_reproduce(args: argparse.Namespace) -> int:
    manifest = json.loads((Path(args.run) / "manifest.json").read_text(encoding="utf-8"))
    _print({"run_id": manifest["run_id"], "phase4_git_sha": manifest["phase4_git_sha"],
        "harness_release": manifest["harness_release"], "harness_sha": manifest["harness_resolved_sha"],
        "level_profile": manifest["level_profile"], "algorithm": manifest["algorithm"],
        "model": manifest["model_architecture"], "hyperparameters": manifest["hyperparameters"],
        "evaluation_protocol": manifest["evaluation_protocol"], "game_rng_controlled": False})
    return 0


def command_audit(args: argparse.Namespace) -> int:
    _print(audit_records(load_jsonl(args.transitions)))
    return 0


def command_dashboard(args: argparse.Namespace) -> int:
    from pvz_deeplearning.dashboard import launch_dashboard
    launch_dashboard(args.run)
    return 0


def command_tune(args: argparse.Namespace) -> int:
    experiment, _level, model_config = _resolve_config(args.config)
    if experiment.mode != "mock":
        raise SystemExit("live tuning is blocked with harness v0.1.0")
    backend = get_backend(experiment.algorithm)
    def objective(trial: Any) -> float:
        env = MockPvZEnv(8)
        params = dict(experiment.hyperparameters)
        params.update(suggest_maskable_ppo(trial))
        params["n_epochs"] = min(int(params.get("n_epochs", 2)), 2)
        params["verbose"] = 0
        model = backend.build(env, model_config=model_config, seed=experiment.seed + trial.number,
                              device=experiment.device, hyperparameters=params)
        backend.learn(model, min(experiment.total_timesteps, 128))
        _, summary = evaluate_masked(env,
            lambda obs, mask: backend.predict(model, obs, mask, deterministic=True), 3,
            seed=10_000 + trial.number * 10)
        trial.set_user_attr("evaluation_summary", summary)
        return float(summary["mean_waves"] * 1000 + summary["mean_return"])
    study = run_study(args.storage, args.study, args.trials, objective)
    _print({"study": study.study_name, "trials": len(study.trials), "best_value": study.best_value,
            "best_params": study.best_params, "result_class": "MOCK"})
    return 0


def command_tune_report(args: argparse.Namespace) -> int:
    import optuna
    study = optuna.load_study(study_name=args.study,
        storage=f"sqlite:///{Path(args.storage).resolve().as_posix()}")
    _print({"study": study.study_name, "direction": study.direction.name,
            "trials": [{"number": t.number, "value": t.value, "params": t.params,
                        "state": t.state.name} for t in study.trials],
            "best_trial": study.best_trial.number, "best_value": study.best_value,
            "best_params": study.best_params})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pvz-dl")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="read-only environment and capability checks")
    doctor.set_defaults(func=command_doctor)
    train = sub.add_parser("train", help="run bounded training from tracked configuration")
    train.add_argument("--config", required=True); train.add_argument("--output", default="artifacts/runs")
    train.add_argument("--resume", help="explicit checkpoint to resume into a new lineage run")
    train.add_argument("--yes", action="store_true", help="explicitly authorize live gameplay control")
    train.set_defaults(func=command_train)
    evaluate = sub.add_parser("evaluate", help="independent masked evaluation (mock until live blockers resolve)")
    evaluate.add_argument("--policy", choices=("random-valid", "scripted-heuristic", "checkpoint"), default="checkpoint")
    evaluate.add_argument("--checkpoint"); evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--seed", type=int, default=0); evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--output"); evaluate.set_defaults(func=command_evaluate)
    inspect = sub.add_parser("inspect"); inspect.add_argument("run"); inspect.set_defaults(func=command_inspect)
    reproduce = sub.add_parser("reproduce"); reproduce.add_argument("run"); reproduce.set_defaults(func=command_reproduce)
    audit = sub.add_parser("audit-reward"); audit.add_argument("transitions"); audit.set_defaults(func=command_audit)
    dashboard = sub.add_parser("dashboard"); dashboard.add_argument("--run"); dashboard.set_defaults(func=command_dashboard)
    tune = sub.add_parser("tune", help="sequential local tuning")
    tune.add_argument("--config", required=True); tune.add_argument("--trials", type=int, default=3)
    tune.add_argument("--study", default="maskable_ppo_mock"); tune.add_argument("--storage", default="artifacts/tuning/study.db")
    tune.set_defaults(func=command_tune)
    report = sub.add_parser("tune-report"); report.add_argument("--study", required=True); report.add_argument("--storage", required=True)
    report.set_defaults(func=command_tune_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
