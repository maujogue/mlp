"""Pydantic models for visualizer HTTP responses (aligned with docs/visualizer_api.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mlp.model.schemas import TrainingHistory, TrainingRunConfig


class RunListItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Path relative to runs root (POSIX)")
    relative_path: str
    has_history: bool = True
    has_run_config: bool = False
    epochs_ran: int | None = None
    elapsed_seconds: float | None = None
    final_train_loss: float | None = None
    final_val_loss: float | None = None
    history_mtime_ms: int | None = Field(
        default=None,
        description="Unix ms from history.json mtime for sort/filter by recency",
    )
    config_train_path: str | None = Field(
        default=None,
        description="From run_config.json when present (absolute or as saved)",
    )
    config_layers_str: str | None = Field(
        default=None,
        description="Hyphen-joined layer widths, e.g. 24-24",
    )
    config_epochs: int | None = None
    config_learning_rate: float | None = None
    config_seed: int | None = None
    config_batch_size: int | None = None
    config_optimizer: str | None = None
    config_patience: int | None = None


class RunListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runs_root: str
    runs: list[RunListItem]


class RunDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_path: str
    history: TrainingHistory
    run_config: TrainingRunConfig | None = None


class LiveTrainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: Path
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    layers: list[int] = Field(default_factory=lambda: [24, 24], min_length=2)
    epochs: int = Field(default=70, gt=0)
    learning_rate: float = Field(default=0.01, gt=0.0)
    seed: int = 42
    batch_size: int = Field(default=0, ge=0)
    optimizer: Literal["sgd", "rmsprop"] = "rmsprop"
    patience: int = Field(default=0, ge=0)
    parent_dir: Path = Field(default=Path("temp"))
    telemetry_sample_every_n_batches: int = Field(default=1, ge=1)
    eval_test_path: Path | None = Field(
        default=None,
        description="If set, persist artifacts and emit test_eval SSE (binary CE on this CSV)",
    )
    lesson_mode: bool = Field(
        default=False,
        description="Record micro-step replay during training; opens fullscreen viewer when done.",
    )
    lesson_max_micro_steps: int = Field(
        default=25_000,
        ge=500,
        le=500_000,
        description="Reject live train if estimated micro-steps exceed this (protects browser/SSE).",
    )


class LiveTrainResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


class LiveBestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: Path
    val_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    epochs: int = Field(default=70, gt=0)
    seed: int = 42
    parent_dir: Path = Field(default=Path("temp"))
    test_paths: list[Path] = Field(default_factory=list)
    grid_mode: Literal["quick", "full"] = "quick"


class LiveBestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str


class EvaluateTestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Field(
        default=Path("temp"), description="Runs root (same as /api/runs)"
    )
    test_path: Path
    run_paths: list[str] = Field(
        min_length=1,
        description="Run folder paths relative to root (RunListItem.id)",
    )


class EvaluateTestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test_path: str
    results: dict[str, Any]
    """Map run id -> metric fields or {\"error\": \"...\"}."""


class DatasetListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    datasets_root: str
    files: list[str]


class PrepareDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Path
    output: Path | None = Field(
        default=None,
        description="If omitted, writes {stem}_prepared.csv next to source (CLI behavior).",
    )


class PrepareDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    output: str


class SplitDatasetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepared_path: Path
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)


class SplitDatasetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str
    test_path: str
    folder: str


def training_run_config_from_live(req: LiveTrainRequest) -> TrainingRunConfig:
    """Build TrainingRunConfig from a live training request."""
    parent = req.parent_dir
    if not parent.is_absolute():
        parent = Path.cwd() / parent
    return TrainingRunConfig(
        train_path=req.train_path.resolve(),
        val_ratio=req.val_ratio,
        layers=req.layers,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        seed=req.seed,
        batch_size=req.batch_size,
        optimizer=req.optimizer,
        patience=req.patience,
        parent_dir=parent.resolve(),
    )


class LessonLossSliceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path = Field(default=Path("lesson_replays"))
    replay_path: str = Field(
        description="Folder name under `root`, or an absolute run directory containing `lesson_replay/`.",
    )
    step_index: int = Field(ge=0)
    param_i: int = Field(ge=0)
    param_j: int = Field(ge=0)
    grid_half_extent: float = Field(default=0.35, gt=0.0)
    grid_n: int = Field(default=17, ge=5, le=41)


def training_run_config_for_best_search(req: LiveBestRequest) -> TrainingRunConfig:
    """Base config for hyperparameter grid; layers/lr/batch/optim/patience come from each grid row."""
    parent = req.parent_dir
    if not parent.is_absolute():
        parent = Path.cwd() / parent
    tests = [p.resolve() for p in req.test_paths]
    return TrainingRunConfig(
        train_path=req.train_path.resolve(),
        val_ratio=req.val_ratio,
        layers=[24, 24],
        epochs=req.epochs,
        learning_rate=0.01,
        seed=req.seed,
        batch_size=0,
        optimizer="rmsprop",
        patience=0,
        test_paths=tests,
        parent_dir=parent.resolve(),
    )
