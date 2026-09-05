# Global results

No real or pilot learned-policy evaluation has been run. Synthetic smoke runs validate software plumbing only and are intentionally excluded from scientific comparison tables.

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
