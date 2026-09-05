# PvZ Deep Learning

An auditable deep-reinforcement-learning research stack for the real Plants vs. Zombies GOTY 1.2.0.1073 client, built strictly above [PvZ AI Harness v0.2.1](https://github.com/b3d012/PvZ-AI-Harness/releases/tag/v0.2.1).

> **Status:** Phase 4 offline implementation is complete and its live factory
> is release-gated. The harness training-lifecycle candidate is pushed but not
> merged or released; real WON/LOST, automatic reset, and managed pickups still
> require live validation. No real training result is claimed.

```mermaid
flowchart TD
    G[PvZ GOTY] <--> H[PvZ-AI-Harness v0.2.1]
    H --> E[Environment v1: Observation 5534 / Action 541 / Reward v1]
    E --> A[Gymnasium adapter + masks]
    A --> P[SB3-Contrib MaskablePPO]
    A --> B[Random / scripted baselines]
    P --> T[Bounded training + checkpoints + manifests]
    T --> V[Independent evaluation / Optuna / reward audit]
    V --> D[Tk dashboard and curated per-level results]
```

## Why MaskablePPO

The environment has a large discrete action space whose legal subset changes every step, and interaction with one real game is slow. sb3-contrib provides native masks during both learning and prediction, mature checkpointing, custom policy architectures, CPU/CUDA support, and local metrics. The full comparison and limitations are in [Phase 4.1 design](docs/PHASE_4_1_DESIGN.md).

Available model IDs are `maskable_ppo_mlp_small` (`128,128`), `maskable_ppo_mlp_medium` (`256,256`), and `maskable_ppo_mlp_large` (`512,256,128`). Size is an experimental variable, not a quality ranking. New algorithms implement the backend protocol and register in `algorithms/registry.py`; no environment or result rewrite is needed.

## Setup

```powershell
conda env create -f environment.yml
conda activate pvz-rl
python -m pip install -e .[tuning,tensorboard]
pvz-dl doctor
```

The harness dependency is pinned to immutable tag `v0.2.1` at resolved commit `7f0b71049362b0efe4171b937f47a8acc1d6d1ef`. Local editable harness development is allowed for development only; durable runs record both release and resolved commit.

## Commands

```powershell
pvz-dl doctor
pvz-dl train --config configs/experiments/mock_smoke.yaml
pvz-dl train --config configs/experiments/mock_smoke.yaml --resume artifacts/runs/PARENT/checkpoints/latest.zip
pvz-dl evaluate --policy checkpoint --checkpoint artifacts/runs/RUN/checkpoints/latest.zip --episodes 10
pvz-dl evaluate --policy random-valid --episodes 10
pvz-dl audit-reward artifacts/runs/RUN/transitions/transitions.jsonl
pvz-dl inspect artifacts/runs/RUN
pvz-dl reproduce artifacts/runs/RUN
pvz-dl tune --config configs/experiments/mock_smoke.yaml --trials 3
pvz-dl tune-report --study maskable_ppo_mock --storage artifacts/tuning/study.db
pvz-dl dashboard --run artifacts/runs/RUN
tensorboard --logdir artifacts/runs/RUN/tensorboard
```

`live_pilot.yaml` is deliberately fail-closed even with `--yes` until the upstream blockers are released and validated. No ordinary/offline command sends desktop input.

## Reproducibility and results

Each ignored `artifacts/runs/<run-id>/` contains an immutable manifest, resolved YAML, checkpoints, JSONL metrics, evaluation, TensorBoard, and transitions. Manifests record Git/harness IDs, all schemas, level, architecture, hyperparameters, five software seed roles, uncontrolled game RNG, versions, device, timing, budget, and lineage. A separate immutable `run_completion.json` records final checkpoint path/SHA256, step, completion reason, finish time, and optional evaluation reference. Raw checkpoints never enter Git. Curated real/pilot/mock classifications live in [results/RESULTS.md](results/RESULTS.md); there are currently no real learned-policy results.

The first live candidate is Adventure 1-7: a normal daytime five-lane lawn, fixed six-packet seed bank, 1x speed, and 250 ms strategic decisions. It is prepared normally and verified through memory because forced earlier levels were unstable on the target installation. Evaluation separates natural win/loss from horizon and technical truncations and reports distributions, not a best episode. See [research plan](docs/RESEARCH_PLAN.md) and [technical report](docs/technical-report.tex).

## Limitations and roadmap

- A read-only check observed a real paused Adventure level 7 Board and the candidate outcome evidence mapped it to `RUNNING`; no WON/LOST, reset, pickup, model-action, reward, resume, evaluation, or throughput result is claimed.
- Harness v0.2.0 is the durable immutable pin. Its lifecycle support is live
  validated: terminal outcomes, active/pause-menu/loss/win same-level reset,
  managed pickup collection, and runtime serialization.
- PvZ RNG is not controlled by recorded software seeds.
- Action v1 has no shovel action; it remains intentionally deferred.
- Structured neural encoders await versioned observation slice metadata; the MLP uses the frozen flat vector without mystery indexes.

Next: run bounded random-valid, scripted, and MaskablePPO Adventure 1-7 pilots
after the live level preflight succeeds; only then consider checkpoint resume,
evaluation, throughput measurement, and tiny tuning plumbing.
