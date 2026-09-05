# Phase 4 research plan

## Goal

Phase 4 begins the learned-agent portion of the project. The objective is to train and evaluate deep reinforcement-learning policies against the real Plants vs. Zombies GOTY game while treating `PvZ-AI-Harness` v0.1.0 as a frozen, versioned environment boundary.

The research question is not merely whether a network can produce clicks. The project should test whether a learned policy can develop useful strategic behavior from the structured state/action interface and outperform transparent non-learned baselines under a reproducible protocol.

## Fixed infrastructure

The following are inherited from the harness and should be treated as controlled infrastructure rather than research variables unless a later experiment explicitly versions them:

- read-only GameState v1;
- Observation v1, shape `(5534,)`;
- Action v1, 541 WAIT/PLANT actions with deterministic invalid-action masking;
- Controller v1;
- Environment v1 lifecycle/step contract;
- Reward v1 and Transition schema v2;
- PID-bound runtime, focus/pause safety, reattachment, and diagnostics;
- frozen random and scripted engineering baselines.

The initial Phase 4 dependency is pinned to harness release `v0.1.0`.

## Research questions

1. Which deep-RL family best fits the fixed discrete masked action space and expensive real-time environment interaction?
2. How much learning signal is available from Reward v1 without introducing reward-hacking incentives?
3. What decision interval provides a useful trade-off between responsiveness, sample efficiency, and real-time wall-clock cost?
4. How robustly does a learned policy generalize across seeds/episode conditions instead of memorizing one prepared board state?
5. How much do action masking and selected observation groups contribute to performance?
6. Can the learned policy outperform the harness random-valid and scripted heuristic baselines under the same evaluation protocol?

## Phase 4.1 — RL interface and protocol

Before selecting a model implementation, decide and document:

- whether to use a thin Gymnasium-compatible wrapper above Environment v1 or a framework-specific adapter;
- how masked actions are exposed to the selected library;
- step/decision timing for the first experiments;
- the manually prepared reset workflow and what constitutes a valid episode start;
- how technical interruptions are separated from environment termination/truncation;
- the initial target level(s) and active-row configuration;
- training/evaluation seed policy;
- baseline comparison protocol;
- metrics and stopping criteria;
- checkpoint/run-manifest schema;
- candidate algorithms and the rationale for the first baseline.

Do not modify the harness to satisfy a framework API if an adapter in this repository can solve the problem cleanly.

## Candidate algorithm criteria

The first algorithm should be judged against:

- support for discrete action spaces with invalid-action masks;
- stability with relatively low-throughput real-world interaction;
- sample efficiency;
- ease of reproducible seeding/checkpointing;
- Windows compatibility;
- mature implementation and inspectable training metrics;
- ability to use the fixed Observation v1 without architecture-specific harness changes.

A maskable PPO-family implementation is an obvious candidate. Alternatives should be considered explicitly before committing the project to it.

## Evaluation protocol principles

At minimum, durable comparisons should:

- use multiple random seeds where the environment/policy permits;
- report mean plus spread/uncertainty rather than one cherry-picked run;
- preserve identical environment/harness contracts across compared policies;
- compare against frozen random-valid and scripted baselines;
- separate training conditions from held-out evaluation conditions where practical;
- report technical truncations separately from game outcomes;
- retain exact run metadata for every published checkpoint/result.

## Metrics

Candidate headline metrics include:

- episode return under the fixed reward specification;
- waves survived/progressed;
- terminal win rate once an authoritative terminal detector exists;
- episode length;
- illegal/rejected action rate (expected near zero with correct masking);
- sun/economy utilization metrics if derivable without altering the reward;
- wall-clock training time and environment steps;
- variance across seeds;
- performance relative to scripted/random baselines.

Do not introduce a metric as a reward term merely because it is convenient to plot.

## Reproducibility manifest

Every durable run should eventually persist:

```text
run_id
created_at
phase4_git_sha
harness_release
harness_resolved_sha
environment_contract
algorithm
algorithm_library_version
model_architecture
hyperparameters
random_seeds
episode_config
active_rows
reward_spec
step_timing
training_budget
hardware_device
python_version
dependency_versions
checkpoint_path_or_id
evaluation_protocol
results
```

A run manifest is immutable historical evidence, not a dashboard state file.

## Artifact policy

Large generated artifacts stay out of Git. Checkpoints, datasets, trajectories, logs, tuning databases, and dashboard runs should live in ignored directories or external artifact storage. Only small curated summaries/configs/manifests that materially support reproducibility should be promoted into version control.

## Phase boundary rule

If Phase 4 discovers a missing low-level capability, first decide whether the problem belongs to the learning layer or the harness. Harness fixes happen in `PvZ-AI-Harness`, receive their own tests/release/version, and are then deliberately adopted here. Phase 4 must not silently fork the harness inside this repository.

## Decisions made in Phase 4.1

- **Framework:** Gymnasium 1.2.2 with Stable-Baselines3/sb3-contrib 2.9.0. A thin adapter preserves the harness contract and Gymnasium's terminated/truncated distinction.
- **First algorithm:** MaskablePPO. Native changing-action masks and mature local checkpoint/metric support outweigh its on-policy sample-efficiency limitation for the first baseline. DQN variants remain the most important future off-policy comparison.
- **Model:** flat Observation v1 MLP `[128,128]`, with tracked medium/large variants. A structured extractor is deferred until observation slices are public versioned metadata.
- **Initial level:** Adventure 1-4, a regular five-lane daytime economy/defense
  level. The earlier 1-5 proposal was rejected because it is Wall-nut Bowling,
  a conveyor/bowling condition that does not exercise the intended Action v1
  placement problem.
- **Reward:** unchanged harness Reward v1. Additional shaping requires a new named/versioned profile and reward-component tests.
- **Tuning objective:** held-out normalized wave progress, then evaluation return while wins remain sparse; win rate becomes primary only after natural outcomes are reliable and frequent.
- **Reset/pickup/speed:** operator-prepared reset, no managed pickup service, and 1x only under harness v0.1.0. These are explicit blockers, not model-layer workarounds.

Alternatives and the full rationale are recorded in `docs/PHASE_4_1_DESIGN.md`.

## Immediate experiment sequence

1. Release backward-compatible authoritative terminal, verified same-level reset, and serialized pickup support in the harness after offline and live validation.
2. Pin that immutable release and update contract metadata here.
3. Run a bounded Adventure 1-4 attach/reset/action/reward/checkpoint/resume pilot.
4. Evaluate random-valid, scripted, and learned policies with the identical protocol.
5. Run sequential Optuna trials, then fresh multi-seed confirmation on held-out episodes.

Open research questions remain data efficiency versus off-policy methods, decision-interval sensitivity, mask ablation, reward robustness, architecture scaling, and specialist-versus-curriculum transfer.
