"""Optional training telemetry for visualizers (batch/epoch events)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

TelemetryCallback = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class TrainingTelemetryOptions:
    """When callback is None, training behaves as before (no extra work)."""

    callback: TelemetryCallback | None = None
    """Invoked as callback(event_type, payload) for supported events."""

    sample_every_n_batches: int = 1
    """Emit at most every N-th batch (1 = every batch). Only applies to batch events."""

    defer_fit_done_callback: bool = False
    """If True, ``fit`` does not emit ``done``; ``run_training`` emits it after saving artifacts."""

    def should_emit_batch(self, batch_index_zero_based: int) -> bool:
        if self.callback is None:
            return False
        n = max(1, int(self.sample_every_n_batches))
        return batch_index_zero_based % n == 0
