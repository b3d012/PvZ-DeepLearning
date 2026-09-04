# Tracked configuration

- `algorithms/`: library identity and invariant mask behavior.
- `models/`: neural architecture templates.
- `levels/`: controlled live conditions and capability constraints.
- `experiments/`: budgets, seeds, backend, device, and hyperparameters.

Templates are versioned inputs. Every run writes a resolved immutable copy; never edit a completed run manifest to match remembered settings.

`adventure_1_4.yaml` is the unreleased live candidate and intentionally requests
managed reset/pickups; the release gate prevents it from running with v0.1.0.
`adventure_1_5.yaml` is retained for provenance but is not the first RL target:
1-5 is Wall-nut Bowling rather than a regular placement/economy level.
