import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ..data.data_engineering import (
    fit_scaler_on_train_and_transform_train_val,
    fix_dataset,
    split_features_labels,
    split_train_validation,
)
from ..utils.constants import FEATURE_COLUMNS
from ..utils.loader import build_run_dir, load_dataset
from .mlp_classifier import MLPClassifier
from .plots import save_learning_curves
from .schemas import TrainingHistory, TrainingRunConfig
from .telemetry import TrainingTelemetryOptions
from .serialization import (
    save_model,
    save_run_config,
    save_training_history,
)


def _load_and_prepare_train_val_arrays(
    train_path: Path,
    val_ratio: float,
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train_path, fix and split into train/val, scale, and return (X_train, y_train, X_val, y_val).
    """
    df = load_dataset(train_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    train_df, val_df = split_train_validation(df, val_ratio)
    scaler_path: Path = run_dir / Path("scaler.pkl")
    train_df, val_df = fit_scaler_on_train_and_transform_train_val(
        train_df, val_df, scaler_path
    )
    X_train, y_train = split_features_labels(train_df)
    X_val, y_val = split_features_labels(val_df)
    X_train = X_train.to_numpy(dtype=np.float64)
    y_train = y_train.astype(np.int64).values
    X_val = X_val.to_numpy(dtype=np.float64)
    y_val = y_val.astype(np.int64).values
    return X_train, y_train, X_val, y_val


def train_cmd(
    run_config: TrainingRunConfig,
) -> Path:
    # batch_size=0 means full dataset per step (full-batch gradient descent)
    # patience=0 means early stopping disabled
    # val_ratio: fraction of train_path to use as validation (train is split into train/val)
    run_dir: Path = build_run_dir(run_config)
    curves_dir: Path = run_dir / "figures"
    run_training(
        run_dir,
        run_config,
        telemetry=None,
        save_artifacts=True,
    )
    print(f"Training figures saved to {curves_dir}")
    return run_dir


def run_training(
    run_dir: Path,
    run_config: TrainingRunConfig,
    *,
    telemetry: TrainingTelemetryOptions | None = None,
    save_artifacts: bool = True,
    after_save: Callable[[], None] | None = None,
) -> TrainingHistory:
    """Load data, train model, optionally persist artifacts into ``run_dir``."""
    curves_dir: Path = run_dir / "figures"
    model_path: Path = run_dir / "model.pkl"

    X_train, y_train, X_val, y_val = _load_and_prepare_train_val_arrays(
        train_path=run_config.train_path,
        val_ratio=run_config.val_ratio,
        run_dir=run_dir,
    )

    model = MLPClassifier(
        n_features=len(FEATURE_COLUMNS),
        hidden_layers=run_config.layers,
        output_size=2,
        seed=run_config.seed,
    )

    start_time = time.perf_counter()
    history: TrainingHistory = model.fit(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        run_config=run_config,
        telemetry=telemetry,
    )

    elapsed_seconds = time.perf_counter() - start_time
    if save_artifacts:
        save_model(model, model_path)
        save_learning_curves(history, curves_dir)
        save_training_history(run_dir, history, elapsed_seconds)
        save_run_config(run_dir, run_config)
    if after_save is not None:
        after_save()
    if (
        telemetry is not None
        and telemetry.callback is not None
        and telemetry.defer_fit_done_callback
    ):
        telemetry.callback(
            "done",
            {
                "elapsed_seconds": model.last_fit_seconds,
                "epochs_ran": len(history.train_loss),
                "history": history.model_dump(by_alias=True),
            },
        )
    return history
