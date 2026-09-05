"""Offline contract tests for the pinned PvZ AI Harness."""

import unittest

from pvz_deeplearning.harness import (
    EXPECTED_HARNESS_CONTRACT,
    HARNESS_RELEASE,
    assert_supported_harness_contract,
    harness_contract,
)


class HarnessContractTests(unittest.TestCase):
    def test_release_pin_is_explicit(self):
        self.assertEqual(HARNESS_RELEASE, "v0.2.0")

    def test_installed_harness_matches_expected_contract(self):
        self.assertEqual(harness_contract(), EXPECTED_HARNESS_CONTRACT)

    def test_assertion_returns_verified_contract(self):
        self.assertEqual(assert_supported_harness_contract(), EXPECTED_HARNESS_CONTRACT)


if __name__ == "__main__":
    unittest.main()
