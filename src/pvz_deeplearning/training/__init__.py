"""Training infrastructure for Phase 4 learning experiments."""
from .trainer import SafetyStopCallback, train_model

__all__ = ["SafetyStopCallback", "train_model"]
