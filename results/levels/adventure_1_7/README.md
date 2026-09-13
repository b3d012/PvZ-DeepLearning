# Adventure 1-7

Controlled Phase 4 condition: normal daytime five-lane lawn, fixed six-packet
seed bank, 1x game speed, and a 250 ms initial strategic decision interval.
The level is prepared normally and verified through harness memory because
forcing earlier Adventure levels was unstable on the target installation.

The v0.2.3 lifecycle foundation and bounded live pilots are now recorded in
`results/RESULTS.md`. Random-valid (3), scripted-heuristic (3), and a
MaskablePPO `mlp_small` 512-step pilot all ran against the same production
condition without technical truncation. A resumed checkpoint and live
checkpoint evaluation also completed. These are PILOT results only; the game
RNG was uncontrolled and no learned-strategy claim is made.
