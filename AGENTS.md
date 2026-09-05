# AGENTS.md

Repository instructions for coding agents working on **PvZ-DeepLearning** (Phase 4+ research).

Read this before substantial changes.

## Project boundary

- Repository: `b3d012/PvZ-DeepLearning`
- Upstream harness: `b3d012/PvZ-AI-Harness`
- Harness pin: **v0.2.0** (`1fb399f2873b6c36f26269385a751389a4ab95bf`)
- Current milestone: **Phase 4 live integration enabled; bounded pilots remain pending**
- Target game/harness platform: **Plants vs. Zombies GOTY 1.2.0.1073 on Windows**

The Phase 1–3.5 game-integration stack is not implemented here. It is frozen and versioned in `PvZ-AI-Harness`.

```text
PvZ GOTY
   ↕
PvZ-AI-Harness v0.2.0
   ↓
Environment v1
   ↓
PvZ-DeepLearning
   ├─ models
   ├─ training
   ├─ tuning
   └─ evaluation
```

## Non-negotiable harness rule

Do **not** copy or reimplement harness internals in this repository merely for convenience. In particular, do not duplicate:

- memory addresses, pointer chains, or reader structures;
- Win32 process/window/focus logic;
- raw controller coordinate logic;
- pause/resume mechanics;
- deterministic placement legality;
- Environment v1 action masking;
- runtime reconnect/watchdog behavior.

Use the public harness APIs.

If Phase 4 reveals a genuine harness deficiency:

1. prove that the limitation is below the learning boundary;
2. propose the smallest compatible change in `PvZ-AI-Harness`;
3. implement/test/release it there;
4. pin the new harness release here deliberately;
5. record the upgrade in experiment metadata and documentation.

Do not silently depend on harness `main`. Durable experiments use an explicit release/tag or exact commit.

## Initial frozen contract

`src/pvz_deeplearning/harness.py` is the compatibility gate. Initial expectations are:

- harness release `v0.2.0`;
- Observation schema v1, shape `(5534,)`;
- Action schema v1, 541 actions;
- Environment schema v1;
- Reward schema v1;
- Transition schema v2.

A contract mismatch is a reproducibility failure and should stop execution rather than be coerced.

## Phase 4 architecture rules

### Model code

- Model code consumes encoded observations/action masks and produces semantic Action v1 indexes or an explicitly versioned future learning interface.
- A model must not know PvZ memory offsets, HWNDs, screen coordinates, or keyboard/mouse primitives.
- Do not make the neural network relearn deterministic placement legality already encoded by the harness.
- Keep architecture definitions separate from training loops and evaluation.

### Training

- Training code owns algorithms, optimizers, rollout/replay logic, checkpoints, callbacks, and run metadata.
- Hyperparameters belong in tracked experiment/config files, not in `pvz_runtime` or the harness monitor.
- Training tools may display harness/runtime health but must not create a second focus/pause/session implementation.
- A failure in runtime health is a technical interruption, not automatically a game loss.

### Evaluation

- Keep evaluation code separate from training code.
- Compare learned policies against the frozen harness random/scripted baselines where meaningful.
- Report multiple seeds and uncertainty rather than only a best run.
- Avoid evaluating on the exact same episode conditions used for hyperparameter selection when a held-out protocol is practical.

### Reproducibility

Every durable run/checkpoint should eventually capture:

- algorithm name/version;
- full hyperparameters;
- model architecture;
- random seed(s);
- Phase 4 Git commit;
- harness release and resolved commit;
- environment contract metadata;
- episode/level configuration and active rows;
- reward specification;
- timing/training budget;
- Python/ML-library/device versions;
- checkpoint ID/path;
- evaluation metrics.

Never overwrite a run's configuration after training to make it match what was actually used. Preserve provenance.

## Algorithm selection

Do not lock the project to PPO/DQN/etc. merely because one is popular. Phase 4.1 should compare candidate approaches against the actual fixed observation/action structure, invalid-action mask, real-time environment cost, reset constraints, and data efficiency requirements.

A maskable PPO-family method is a reasonable first candidate, not a pre-approved final architecture.

## Dependency rules

The base project should remain small. Add ML/RL libraries only in the subphase that uses them.

When adding a dependency:

1. justify why it is needed;
2. pin or constrain it reproducibly;
3. update `pyproject.toml` and/or environment documentation;
4. keep CPU/Windows installation practical where possible;
5. update CI if the dependency belongs in offline tests.

Do not add optional experiment platforms (W&B, Ray, Optuna, etc.) until they provide concrete value.

## Artifact policy

Do not commit generated heavy artifacts:

- model checkpoints (`*.pt`, `*.pth`, `*.ckpt`, etc.);
- replay buffers/datasets;
- TensorBoard or W&B run folders;
- Optuna databases;
- raw trajectories/logs;
- generated plots/result dumps;
- local game files;
- secrets or `.env` files.

Tracked result summaries belong under `results/`; generated payloads remain ignored unless explicitly promoted as a small durable research artifact.

## Safety and live interaction

- Offline tests and CI must never require PvZ or send desktop input.
- Live training/evaluation commands must be explicit and visibly documented.
- Do not bypass harness focus/window/watchdog checks to increase throughput.
- Never reinterpret runtime failure as permission to click blindly.

## Git workflow

For substantial work:

1. start from current `main`;
2. create a focused branch;
3. implement one bounded Phase 4 subtask;
4. add/update offline tests;
5. run validation;
6. document experimental implications;
7. commit coherently;
8. open a PR;
9. do not start a later subphase merely because there is remaining context/time.

Suggested branch names:

```text
research/phase-4.1-rl-interface
feat/phase-4.2-first-baseline
feat/training-run-manifest
experiment/maskable-ppo-v1
fix/evaluation-seed-accounting
```

## Validation

At minimum for infrastructure changes:

```powershell
python -m compileall -q src tests scripts
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/check_harness.py
```

Training-model changes should also include deterministic smoke tests that do not launch the real game whenever possible.

Do not claim a live training/evaluation result unless it was actually run against the game and the exact configuration is recorded.

## Documentation responsibilities

- `README.md`: public/portfolio overview and current Phase 4 status.
- `docs/RESEARCH_PLAN.md`: experimental questions, protocol, phase sequencing, and design decisions.
- `docs/technical-report.tex`: formal Phase 4 ML/research report; do not duplicate the harness engineering report.
- `configs/`: tracked algorithm/experiment configuration once those formats are introduced.
- `results/`: concise durable result summaries, not giant generated artifacts.

The Phase 1–3.5 technical history remains in the harness repository and should be linked, not copied wholesale.

## Current handoff state

- `PvZ-AI-Harness` v0.2.0 is the immutable upstream foundation.
- This repository has been separated specifically for Phase 4+ learning research.
- Gymnasium 1.2.2, Stable-Baselines3/sb3-contrib 2.9.0, and MaskablePPO are selected.
- `adapters/` owns Gym semantics; `algorithms/` owns backend integrations; `models` are tracked YAML architecture templates; `training/` owns bounded learning/checkpoint callbacks; `evaluation/` never shares training decisions; `tuning.py` owns sequential studies.
- Generated run truth lives under ignored `artifacts/runs/<run-id>/`; tracked conclusions live under `results/levels/` and `results/RESULTS.md`.
- Never edit a completed manifest, silently resume a checkpoint, change reward semantics without a new profile/version, mix tuning and held-out evaluation episodes, or describe mock output as live.
- The dashboard is a frontend to public harness/run APIs. It must not own Windows input, runtime safety, action threads, or mutable canonical configuration.
- Harness v0.2.0 supplies validated outcome/reset/pickup APIs. `mLevelAwardSpawned`
  is authoritative for reward-pending live Board wins; BoardResult alone is not.
  Unknown paused modals fail closed and reset postconditions remain authoritative.
- Adventure 1-7 is the normal daytime controlled condition. Earlier forced-level
  selection was unstable on the target installation; do not return to 1-4/1-5.
