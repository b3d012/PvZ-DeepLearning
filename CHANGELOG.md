# Changelog

## Unreleased — 0.1.0a1

- Selected Gymnasium + Stable-Baselines3/sb3-contrib MaskablePPO after a Phase 4.1 design study.
- Added the exact-contract masked Gym adapter and deterministic mock environment.
- Added configurable MLP models, algorithm registry, bounded training, checkpoints, immutable manifests, lineage validation, and local metrics.
- Added independent evaluation statistics, reward component auditing, sequential Optuna plumbing, doctor/inspect/reproduce commands, and a read-only-first tabbed dashboard.
- Added tracked algorithm, model, level, mock-smoke, and blocked live-pilot configurations.
- Documented that harness v0.1.0 prevents truthful autonomous live multi-episode training; no real results or release are claimed.
- Added a release-gated live environment factory, profile/seed checks, terminal
  adapter, serialized pickup hook, and bounded recovery policy seams.
- Added immutable `run_completion.json` checkpoint identity and stable SB3
  logger metric extraction.
- Added structured runtime/board/agent/evaluation dashboard models.
- Replaced the proposed 1-5 live condition with regular daytime Adventure 1-4;
  1-5 is Wall-nut Bowling. No new live performance result is claimed.
