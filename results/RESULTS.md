# Global results

Real-game results below are bounded PILOT evidence, not statistically conclusive comparisons.

## Phase 4 live pilot — Adventure 1-7 — 13 September 2026

Common provenance: Phase4 merge `9ccdb2583746a6308491afbb9d12bceab03491d4`,
PvZ-AI-Harness `v0.2.3` resolved at
`43fd924808b6d0b0bd4bf9956e7c525a48054b8b`, Observation v1 `(5534,)`, Action
v1 (541), 250 ms decisions, Reward v1, six seeds, active rows 0–4, CPU.
Game RNG was uncontrolled.

| Policy | Run/evaluation | Episodes | Returns | Waves | Technical truncations | Wall time / throughput |
|---|---|---:|---|---|---:|---|
| random-valid | prior live pilot | 3 | 0.07, 0.07, 0.05 | 8, 7, 6 | 0 | ~10,000 steps/hour |
| scripted-heuristic | `scripted-heuristic-live.json` | 3 | 0.04, 0.0475, 0.08 | 9, 9, 9 | 0 | 540.2 s / ~10,230 steps/hour |
| MaskablePPO `mlp_small` | run `20260913T204411Z-live_pilot-9ccdb258` | 1 completed episode, 512 steps (configured max 3) | pilot completed; SB3 rollout mean 0.07 | horizon-bounded | 0 | 195 s / ~9,450 steps/hour |
| resumed checkpoint evaluation | `checkpoint-live.json` | 3 | 0.065, 0.05, 0.09 | 9, 7, 9 | 0 | 561.8 s / ~9,850 steps/hour |

The PPO result is a systems/learning-signal pilot only; no convergence or
strategy-superiority claim is justified. All evaluated episodes completed by
the configured horizon, not by natural win/loss.

Checkpoint: parent run step-512/latest SHA-256 `bb16bc6dbc95993f31c73c0fe7c43adb0082cf3cbc1006e7cb0c35d018a619f`.
Resume run: `20260913T204758Z-live_pilot-9ccdb258`, loaded the parent checkpoint
and advanced to 1024 model timesteps in a new lineage. A real 512-transition
checkpoint audit found total reward `0.03`, equal to component sum: wave
progress `0.03`; terminal, rejected-action, controller-failure,
plant-not-observed, and state-unavailable all `0.0`. The auditor warned that
wave progress was 100% of absolute shaped reward because the bounded episode
had no terminal event; this is expected pilot evidence, not a reward change.

A read-only runtime check on 4 September 2026 observed a real paused Adventure
level 7 Board and candidate terminal evidence `scene=3`, `board_result=0`, and
`level_complete=false`, mapping to `RUNNING`. This is lifecycle evidence only,
not an episode, baseline, or policy result.

## v0.2.0 live-preflight record â€” 5 September 2026

The released harness v0.2.0 contract was verified locally: Observation v1
`(5534,)`, Action v1 `541`, Environment v1, Reward v1, and transition v2 all
matched. Read-only doctor attached to the supported client and observed a
level-7, six-seed, wave-0 Board with `GameOutcome.RUNNING`. The Board was
paused without provenance for a normal PvZ Menu, so the production live factory
refused before any reset, action, model construction, artifact creation, or
desktop input. This is a successful safety preflight, not a baseline or
learning result.

| Class | Model | Level | Training steps | Eval episodes | Finding |
|---|---|---|---:|---:|---|
| MOCK | MaskablePPO MLP small | synthetic exact-shape environment | 128 | not promoted | Pipeline validation only; no PvZ performance claim |
