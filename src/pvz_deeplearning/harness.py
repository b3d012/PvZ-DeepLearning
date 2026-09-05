"""Compatibility gate for the frozen PvZ AI Harness dependency."""

from __future__ import annotations

from typing import Any

from pvz_env import environment_contract


HARNESS_RELEASE = "v0.2.0"

EXPECTED_HARNESS_CONTRACT: dict[str, Any] = {
    "observation_schema_version": 1,
    "observation_shape": [5534],
    "action_schema_version": 1,
    "action_count": 541,
    "environment_schema_version": 1,
    "reward_schema_version": 1,
    "transition_schema_version": 2,
}


def harness_contract() -> dict[str, Any]:
    """Return the installed harness Environment contract as JSON-friendly data."""
    return environment_contract().to_dict()


def assert_supported_harness_contract() -> dict[str, Any]:
    """Fail loudly if the installed harness no longer matches the Phase 4 pin."""
    actual = harness_contract()
    if actual != EXPECTED_HARNESS_CONTRACT:
        raise RuntimeError(
            "Unsupported PvZ AI Harness contract. "
            f"Expected {EXPECTED_HARNESS_CONTRACT!r}, got {actual!r}. "
            "Do not continue an experiment until the harness upgrade is reviewed "
            "and the experiment contract is versioned deliberately."
        )
    return actual
