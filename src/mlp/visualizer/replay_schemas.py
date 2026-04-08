"""Pydantic models for lesson replay: replay_manifest.json + replay_steps.jsonl."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

REPLAY_SCHEMA_VERSION = 1

ReplayPhase = Literal[
    "init",
    "forward_input",
    "forward_layer",
    "activation",
    "loss",
    "backward_layer",
    "optimizer",
    "batch_end",
    "epoch_end",
]

VizMode = Literal["toy2d", "two_features", "pca2", "tabular"]


class TocEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    toc_id: str
    label: str
    step_index: int


class Toy2DPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x0: float
    x1: float
    y: int


class ReplayManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = REPLAY_SCHEMA_VERSION
    run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    input_dim: int
    layer_sizes: list[int] = Field(
        description="Hidden layer widths only; output is n_classes logits",
    )
    n_classes: int = 2
    activation: Literal["relu", "sigmoid"]
    loss: Literal["softmax_cross_entropy"] = "softmax_cross_entropy"
    optimizer: Literal["sgd", "rmsprop"]
    learning_rate: float
    rmsprop_decay: float | None = None
    rmsprop_eps: float | None = None
    viz_mode: VizMode = "toy2d"
    viz_note: str | None = None
    toy_points: list[Toy2DPoint] | None = None
    n_epochs: int
    batches_per_epoch: list[int] = Field(
        default_factory=list,
        description="Batch count per epoch; len == n_epochs",
    )
    total_micro_steps: int
    toc_entries: list[TocEntry] = Field(default_factory=list)
    steps_file: str = "replay_steps.jsonl"
    lesson_anchor_train_index: int | None = Field(
        default=None,
        description="Fixed training-row index (original order) traced in forward/loss/backward when in batch.",
    )


class ChainLocalTerm(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    to: str
    name: str
    value: float


class OptimizerLayerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer: int
    grad_W: list[list[float]] | None = None
    grad_b: list[float] | None = None
    rms_W: list[list[float]] | None = None
    rms_b: list[float] | None = None
    delta_W: list[list[float]] | None = None
    delta_b: list[float] | None = None
    effective_scale_W: list[list[float]] | None = None
    effective_scale_b: list[float] | None = None


class ReplayStep(BaseModel):
    """One scrub frame. Tensor fields are optional and depend on phase."""

    model_config = ConfigDict(extra="forbid")

    step_index: int
    phase: ReplayPhase
    epoch: int
    batch: int
    sample_in_batch: int = 0
    layer: int | None = None
    toc_id: str
    explanation: str = ""
    math: str | None = None

    a_in: list[float] | None = None
    W: list[list[float]] | None = None
    b: list[float] | None = None
    z: list[float] | None = None
    a_out: list[float] | None = None
    edge_contributions: list[list[float]] | None = None

    logits: list[float] | None = None
    probs: list[float] | None = None
    label: int | None = None
    loss_contribution: float | None = None
    pred_class: int | None = None
    correct: bool | None = None

    dL_dz: list[float] | None = None
    dL_da_in: list[float] | None = None
    dL_dW: list[list[float]] | None = None
    dL_db: list[float] | None = None
    chain_local: list[ChainLocalTerm] | None = None

    optimizer_layers: list[OptimizerLayerState] | None = None
    learning_rate: float | None = None
    optimizer_name: Literal["sgd", "rmsprop"] | None = None

    loss_batch_mean: float | None = None

    lesson_anchor_train_index: int | None = None
    lesson_trace_this_batch: bool | None = None


def replay_step_to_jsonable(step: ReplayStep) -> dict[str, Any]:
    """Dump for JSONL (ChainLocal uses alias 'from')."""
    d = step.model_dump(mode="json", by_alias=True)
    return d
