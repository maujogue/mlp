from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainingMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    loss: float = Field(default=0.0, ge=0.0)
    accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    precision: float = Field(default=0.0, ge=0.0, le=1.0)
    recall: float = Field(default=0.0, ge=0.0, le=1.0)
    f1: float = Field(default=0.0, ge=0.0, le=1.0)


class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(default="", min_length=1)
    val_ratio: float = Field(ge=0.0, lt=1.0)
    layers: list[int] = Field(default=[], min_length=2)
    epochs: int = Field(default=0, gt=0)
    learning_rate: float = Field(default=0.0, gt=0.0)
    seed: int = Field(default=42)
    batch_size: int = Field(ge=0)
    optimizer: Literal["sgd", "rmsprop"] = "sgd"
    patience: int = Field(ge=0)
    test_paths: list[str] = Field(default=[])


class TrainingHistory(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    train_loss: list[float] = Field(
        default_factory=list,
        alias="history_train_loss",
    )
    val_loss: list[float] = Field(
        default_factory=list,
        alias="history_val_loss",
    )
    train_accuracy: list[float] = Field(
        default_factory=list,
        alias="history_train_acc",
    )
    val_accuracy: list[float] = Field(
        default_factory=list,
        alias="history_val_acc",
    )
    train_precision: list[float] = Field(
        default_factory=list,
        alias="history_train_precision",
    )
    val_precision: list[float] = Field(
        default_factory=list,
        alias="history_val_precision",
    )
    train_recall: list[float] = Field(
        default_factory=list,
        alias="history_train_recall",
    )
    val_recall: list[float] = Field(
        default_factory=list,
        alias="history_val_recall",
    )
    train_f1: list[float] = Field(
        default_factory=list,
        alias="history_train_f1",
    )
    val_f1: list[float] = Field(
        default_factory=list,
        alias="history_val_f1",
    )

    test_loss: list[float] = Field(
        default_factory=list,
        alias="history_test_loss",
    )
    test_accuracy: list[float] = Field(
        default_factory=list,
        alias="history_test_accuracy",
    )
    test_precision: list[float] = Field(
        default_factory=list,
        alias="history_test_precision",
    )
    test_recall: list[float] = Field(
        default_factory=list,
        alias="history_test_recall",
    )
    test_f1: list[float] = Field(
        default_factory=list,
        alias="history_test_f1",
    )
