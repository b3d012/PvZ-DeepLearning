"""Print and verify the installed PvZ AI Harness contract."""

from __future__ import annotations

import json

from pvz_deeplearning.harness import HARNESS_RELEASE, assert_supported_harness_contract


def main() -> int:
    contract = assert_supported_harness_contract()
    payload = {"harness_release": HARNESS_RELEASE, "environment_contract": contract}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
