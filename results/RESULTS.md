# Global results

No real or pilot learned-policy evaluation has been run. Synthetic smoke runs validate software plumbing only and are intentionally excluded from scientific comparison tables.

A read-only runtime check on 4 September 2026 observed a real paused Adventure
level 7 Board and candidate terminal evidence `scene=3`, `board_result=0`, and
`level_complete=false`, mapping to `RUNNING`. This is lifecycle evidence only,
not an episode, baseline, or policy result.

| Class | Model | Level | Training steps | Eval episodes | Finding |
|---|---|---|---:|---:|---|
| MOCK | MaskablePPO MLP small | synthetic exact-shape environment | 128 | not promoted | Pipeline validation only; no PvZ performance claim |
