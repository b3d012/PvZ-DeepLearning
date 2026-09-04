# Experiment configuration

Tracked model/training/evaluation configuration belongs here once Phase 4.1 chooses the configuration format and first learned baseline.

Keep algorithm-specific settings separate. A future structure may look like:

```text
configs/
  algorithms/
    maskable_ppo.yaml
    ...
  experiments/
    level_1_1_baseline.yaml
    ...
```

Do not place harness runtime/focus/memory configuration here unless it is genuinely an experiment-level parameter exposed by the harness public API.

Every durable config should be immutable after the run begins or copied into that run's manifest so historical experiments remain reproducible.
