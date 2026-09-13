"""Read-only-first Tk dashboard over public harness/runtime and run data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any

from pvz_env.actions import decode_action


TABS = ("Runtime", "Board", "Agent", "Training", "Evaluation", "Tuning", "Runs")


@dataclass
class DashboardState:
    runtime: dict[str, Any] = field(default_factory=dict)
    board: dict[str, Any] = field(default_factory=dict)
    agent: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)
    tuning: dict[str, Any] = field(default_factory=dict)
    runs: dict[str, Any] = field(default_factory=dict)
    training_history: list[dict[str, Any]] = field(default_factory=list)

    def load_run(self, path: str | Path) -> None:
        root = Path(path)
        self.runs = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        metrics = root / "metrics" / "training.jsonl"
        if metrics.exists():
            lines = metrics.read_text(encoding="utf-8").splitlines()
            self.training_history = [json.loads(line) for line in lines if line.strip()]
            self.training = self.training_history[-1] if self.training_history else {}
        evaluation = root / "evaluation" / "summary.json"
        if evaluation.exists():
            self.evaluation = json.loads(evaluation.read_text(encoding="utf-8"))


def runtime_snapshot_model(snapshot: Any) -> dict[str, Any]:
    data = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
    session, health = data.get("session") or {}, data.get("health") or {}
    game, outcome = data.get("game_state") or {}, data.get("outcome") or {}
    process, window = session.get("process") or {}, session.get("window") or {}
    return {
        "process": process.get("name"), "pid": process.get("process_id"),
        "hwnd": window.get("hwnd"), "window_title": window.get("title"),
        "focused": session.get("focused"), "focus_mode": health.get("focus_mode"),
        "reader": health.get("reader_valid"), "controller": health.get("controller_ready"),
        "board": health.get("board_valid"), "phase": data.get("phase"),
        "outcome": outcome.get("outcome"), "level": game.get("adventure_level"),
        "wave": game.get("wave"), "paused": game.get("paused"), "sun": game.get("sun"),
        "plants": game.get("plants"), "zombies": game.get("zombies"),
        "pickups": game.get("pickups"), "state_age_ms": health.get("state_age_ms"),
        "last_action": data.get("last_action"), "last_error": data.get("last_error"),
    }


def board_state_model(state: Any) -> dict[str, Any]:
    return {
        "plants": [{"row": int(p.row), "col": int(p.col), "type": p.name} for p in state.plants],
        "zombies": [{"row": int(z.row), "x": float(z.x), "type": z.name} for z in state.zombies],
        "mowers": [{"row": int(m.row), "x": float(m.x), "available": bool(m.available)} for m in state.mowers],
        "grid_items": [{"row": int(g.row), "col": int(g.col), "type": g.name} for g in state.grid_items],
        "shape": {"rows": 6, "columns": 9},
    }


def agent_action_model(action_index: int, mask: Any) -> dict[str, Any]:
    action = decode_action(int(action_index))
    valid = sum(bool(value) for value in mask)
    return {
        "action_index": int(action_index), "semantic_action": {
            **asdict(action), "action_type": action.action_type.value,
        },
        "selected_action_valid": bool(mask[action_index]),
        "valid_actions": valid, "invalid_actions": len(mask) - valid,
    }


def evaluation_comparison(rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("episodes", "win_rate", "mean_waves", "mean_return", "std_return", "technical_truncations")
    return [{"policy": policy, **{key: summary.get(key) for key in keys}} for policy, summary in rows.items()]


def launch_dashboard(run: str | Path | None = None) -> None:
    state = DashboardState()
    if run:
        state.load_run(run)
    try:
        from pvz_runtime import PvZRuntime, RuntimeConfig
        runtime = PvZRuntime(config=RuntimeConfig(observer_only=True))
        state.runtime = runtime_snapshot_model(runtime.refresh())
        live_state = runtime.observe()
        if live_state is not None:
            state.board = board_state_model(live_state)
        runtime.close()
    except Exception as error:
        state.runtime = {"available": False, "error": f"{type(error).__name__}: {error}"}
    root = tk.Tk()
    root.title("PvZ Deep Learning — Phase 4")
    root.geometry("1000x680")
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)
    for name in TABS:
        frame = ttk.Frame(notebook, padding=12)
        notebook.add(frame, text=name)
        text = tk.Text(frame, wrap="word")
        text.pack(fill="both", expand=True)
        value = getattr(state, name.lower())
        text.insert("1.0", json.dumps(value, indent=2, default=str) if value else f"{name}: no data loaded")
        text.configure(state="disabled")
    root.mainloop()
