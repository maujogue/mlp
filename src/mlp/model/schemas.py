from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_path: str = Field(min_length=1)
    val_ratio: float = Field(ge=0.0, lt=1.0)
    layers: list[int] = Field(min_length=2)
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    seed: int
    batch_size: int = Field(ge=0)
    optimizer: Literal["sgd", "rmsprop"] = "sgd"
    patience: int = Field(ge=0)


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

    @model_validator(mode="after")
    def _validate_lengths(self) -> "TrainingHistory":
        train_len = len(self.train_loss)
        if train_len == 0:
            raise ValueError("history_train_loss/train_loss must be non-empty.")

        train_series = (
            self.train_accuracy,
            self.train_precision,
            self.train_recall,
            self.train_f1,
        )
        if any(len(values) != train_len for values in train_series):
            raise ValueError("All train history series must have the same length.")

        val_lengths = (
            len(self.val_loss),
            len(self.val_accuracy),
            len(self.val_precision),
            len(self.val_recall),
            len(self.val_f1),
        )
        non_zero_val_lengths = [length for length in val_lengths if length > 0]
        if non_zero_val_lengths and any(
            length != non_zero_val_lengths[0] for length in val_lengths
        ):
            raise ValueError("All validation history series must have the same length.")
        if non_zero_val_lengths and non_zero_val_lengths[0] != train_len:
            raise ValueError(
                "Validation history length must match training history length when provided."
            )
        return self

    def to_fit_history(self) -> dict[str, list[float]]:
        return self.model_dump(by_alias=False)
