# Phase 4.1 — RL interface and experimental protocol

Status: implemented and offline-validated. A release-gated live factory and
upstream lifecycle candidate exist; autonomous live episodes remain blocked by
live validation, an automatic reset driver, and the absent immutable release.

## Boundary audit

Environment v1 supplies a deterministic `(5534,)` float32 observation, 541-index `WAIT`/`PLANT` action space, boolean legality mask, configured decision interval, post-action reconciliation, Reward v1, and explicit lifecycle. It safely adopts one operator-prepared board. It does not authoritatively detect natural win/loss, restart the same level, or collect pickups between strategic decisions. Runtime failures are observable but are not game losses.

The bundled pvztoolkit reference contains game-specific patches for auto-collection, speed, and forced completion, but those are memory-writing cheats—not evidence of a safe read-only terminal field or verified current-level reset. They are therefore not imported. Shovel remains deferred because the first controlled daytime baseline can be studied with Action v1 and changing its schema would invalidate checkpoints.

## Algorithm study

| Candidate | Native mask | Data/realtime fit | Maturity and checkpoints | Decision |
|---|---:|---|---|---|
| sb3-contrib MaskablePPO | Yes | On-policy, stable but not maximally sample-efficient | Mature SB3 tooling; Windows/Python 3.12 | First baseline |
| PPO without masking | No | Wastes probability mass over up to 540 illegal actions | Mature | Rejected as primary; future ablation only |
| A2C | No native mask in SB3 | Lower overhead, often less stable | Mature | Deferred |
| DQN / distributional DQN | Custom masking required | Replay may improve expensive-sample reuse | Mature variants exist, integration risk higher | High-priority future comparison |
| Custom masked actor-critic | Can be exact | Tunable | High implementation/validation burden | Deferred |
| Imitation learning | Depends on learner | Could bootstrap from scripted/expert traces | Requires a curated dataset | Future initialization study |

Stable-Baselines3 2.9.0, sb3-contrib 2.9.0, Gymnasium 1.2.2, and PyTorch 2.x are selected. MaskablePPO consumes `action_masks()` during learning; inference passes the mask explicitly. The initial policy is an understandable two-layer MLP (`[128,128]`); medium and large tracked alternatives are available without changing adapter, evaluation, or manifests.

## Gymnasium mapping

`PvZGymEnv` is a thin wrapper; the harness is not changed to mimic a framework. Harness observations and action indexes pass through unchanged. `action_masks()` returns a defensive bool copy. Natural `WIN`/`LOSS` becomes `terminated`; horizon becomes `truncated`; unavailable state, process/focus/runtime failures become technical truncations. This distinction preserves correct bootstrapping semantics and prevents the policy from learning that losing Windows focus means losing PvZ.

Reset preparation is injected. Under v0.1.0, only operator-prepared adoption is truthful. The adapter is ready for a future released harness reset service, but the live CLI refuses autonomous training until that exists.

## Initial condition and protocol

Adventure 1-4 is the proposed first target: it is a regular daytime level with
all five lawn lanes and the early Sunflower/Peashooter economy. The previous
1-5 proposal was invalidated because that level is Wall-nut Bowling. The
profile must still match the live `adventure_level`, exact seed types, normal
speed, and a 250 ms decision interval. Active rows remain an explicit profile
assertion because GameState v1 cannot prove temporary row activation.

Training uses random initialization, explicit step/episode/wall-clock bounds, periodic and final checkpoints, exact config copies, immutable manifests, separate Python/NumPy/PyTorch/framework/policy seed fields, and `game_rng_controlled: false`. A run can fine-tune from an explicitly named parent checkpoint; silent latest-checkpoint selection is prohibited.

Evaluation is independent and deterministic where applicable. Every policy—including random-valid and scripted baselines—must use the same level, timing, reward, harness, reset, and episode limit. Report all episodes, mean/median/std/min/max/95% CI, wave progress, natural outcomes, and technical truncations. Tuning maximizes held-out normalized wave progress, then evaluation return until wins are sufficiently frequent; it never selects on training return alone. Trials are sequential because one real client is available.

## Reward validity

The first baseline preserves `harness_reward_v1`: terminal ±1, wave delta shaping, and small technical diagnostics as defined upstream. No planting bonus exists. Transition logs retain component decomposition. The audit tool aggregates components, episode totals, extremes, action frequencies, and warns about domination by one shaping source. External wave, win, technical-failure, action, and timing metrics are compared with return to detect reward hacking.

## Blockers and release decision

The harness feature branch now has typed raw outcome evidence, reset
postcondition verification, and synchronous serialized pickup collection. Its
199-test offline suite passes. A read-only real Board produced `RUNNING`, but
WON/LOST and pickup behavior are not validated, and reset has only an
operator-assisted callback: no automatic driver is claimed. Phase 4 therefore
keeps the v0.1.0 pin and the live factory refuses construction until v0.2.0 is
published. Game speed remains 1x.
