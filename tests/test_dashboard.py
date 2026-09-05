import json
import tempfile
import unittest
from pathlib import Path

from types import SimpleNamespace

from pvz_deeplearning.dashboard import (
    DashboardState, TABS, agent_action_model, board_state_model,
    evaluation_comparison, runtime_snapshot_model,
)


class DashboardTests(unittest.TestCase):
    def test_tabs_and_run_loading(self):
        self.assertEqual(TABS[:5], ("Runtime", "Board", "Agent", "Training", "Evaluation"))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); (root / "metrics").mkdir(); (root / "evaluation").mkdir()
            (root / "manifest.json").write_text('{"run_id":"x"}', encoding="utf-8")
            (root / "metrics/training.jsonl").write_text('{"step":1}\n', encoding="utf-8")
            state = DashboardState(); state.load_run(root)
            self.assertEqual(state.runs["run_id"], "x"); self.assertEqual(state.training["step"], 1)
            self.assertEqual(len(state.training_history), 1)

    def test_runtime_board_agent_and_evaluation_models(self):
        snapshot = {"session": {"process": {"name": "PvZ", "process_id": 4},
            "window": {"hwnd": 8, "title": "Plants"}, "focused": True},
            "health": {"focus_mode": "manual", "reader_valid": True,
            "controller_ready": True, "board_valid": True, "state_age_ms": 2},
            "game_state": {"adventure_level": 5, "wave": 1},
            "outcome": {"outcome": "running"}, "phase": "playing"}
        self.assertEqual(runtime_snapshot_model(snapshot)["outcome"], "running")
        state = SimpleNamespace(
            plants=[SimpleNamespace(row=1, col=2, name="Sunflower")],
            zombies=[SimpleNamespace(row=1, x=500, name="Zombie")],
            mowers=[SimpleNamespace(row=1, x=20, available=True)], grid_items=[])
        self.assertEqual(board_state_model(state)["plants"][0]["col"], 2)
        self.assertEqual(agent_action_model(0, [True] + [False] * 540)["semantic_action"]["action_type"], "wait")
        rows = evaluation_comparison({"random": {"episodes": 2, "win_rate": 0.0}})
        self.assertEqual(rows[0]["policy"], "random")
