import json
import tempfile
import unittest
from pathlib import Path

from pvz_deeplearning.dashboard import DashboardState, TABS


class DashboardTests(unittest.TestCase):
    def test_tabs_and_run_loading(self):
        self.assertEqual(TABS[:5], ("Runtime", "Board", "Agent", "Training", "Evaluation"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "metrics").mkdir(); (root / "evaluation").mkdir()
            (root / "manifest.json").write_text('{"run_id":"x"}', encoding="utf-8")
            (root / "metrics/training.jsonl").write_text('{"step":1}\n', encoding="utf-8")
            state = DashboardState(); state.load_run(root)
            self.assertEqual(state.runs["run_id"], "x"); self.assertEqual(state.training["step"], 1)
