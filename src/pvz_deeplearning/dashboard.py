"""Read-only-first Tk dashboard over public harness/runtime and run data."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Any


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

    def load_run(self, path: str | Path) -> None:
        root = Path(path)
        self.runs = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        metrics = root / "metrics" / "training.jsonl"
        if metrics.exists():
            lines = metrics.read_text(encoding="utf-8").splitlines()
            self.training = json.loads(lines[-1]) if lines else {}
        evaluation = root / "evaluation" / "summary.json"
        if evaluation.exists():
            self.evaluation = json.loads(evaluation.read_text(encoding="utf-8"))


def launch_dashboard(run: str | Path | None = None) -> None:
    state = DashboardState()
    if run:
        state.load_run(run)
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
