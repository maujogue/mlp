import time
from pathlib import Path

import numpy as np

from ..data.data_engineering import (
    fit_scaler_on_train_and_transform_train_val,
    fix_dataset,
    split_features_labels,
    split_train_validation,
)
from ..utils.constants import DEFAULT_RUN_DIR, FEATURE_COLUMNS
from ..utils.loader import build_run_dir, load_dataset
from .mlp_classifier import MLPClassifier
from .plots import save_learning_curves
from .schemas import TrainingHistory, TrainingRunConfig
from .serialization import (
    save_model,
    save_run_config,
    save_training_history,
)


def _load_and_prepare_train_val_arrays(
    train_path: str,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train_path, fix and split into train/val, scale, and return (X_train, y_train, X_val, y_val).
    """
    df = load_dataset(train_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    train_df, val_df = split_train_validation(df, val_ratio)
    scaler_path = str(DEFAULT_RUN_DIR + "scaler.pkl")
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
    train_path: str = "datasets/train.csv",
    val_ratio: float = 0.2,
    layers: list[int] | None = None,
    epochs: int = 70,
    learning_rate: float = 0.01,
    seed: int = 42,
    run_dir: str | None = None,
    batch_size: int = 0,
    optimizer: str = "sgd",
    patience: int = 0,
) -> None:
    # batch_size=0 means full dataset per step (full-batch gradient descent)
    # patience=0 means early stopping disabled
    # val_ratio: fraction of train_path to use as validation (train is split into train/val)
    run_config = TrainingRunConfig.model_validate(
        {
            "train_path": train_path,
            "val_ratio": val_ratio,
            "layers": list(layers or [24, 24]),
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "patience": patience,
        }
    )

    if run_dir is None:
        run_dir = build_run_dir(
            root=DEFAULT_RUN_DIR,
            train_path=run_config.train_path,
            layers=run_config.layers,
            epochs=run_config.epochs,
            learning_rate=run_config.learning_rate,
            seed=run_config.seed,
            batch_size=run_config.batch_size,
            optimizer=run_config.optimizer,
            patience=run_config.patience,
        )
    curves_dir = str(Path(run_dir) / "figures")
    model_path = str(Path(run_dir) / "model.pkl")

    X_train, y_train, X_val, y_val = _load_and_prepare_train_val_arrays(
        run_config.train_path, run_config.val_ratio
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
        epochs=run_config.epochs,
        learning_rate=run_config.learning_rate,
        batch_size=run_config.batch_size,
        optimizer=run_config.optimizer,
        patience=run_config.patience,
        seed=run_config.seed,
        verbose=True,
    )
    history = TrainingHistory.model_validate(history)

    elapsed_seconds = time.perf_counter() - start_time
    save_model(model, model_path)
    save_learning_curves(history, curves_dir)
    save_training_history(run_dir, history, elapsed_seconds)
    save_run_config(run_dir, run_config)
    print(f"Training figures saved to {curves_dir}")
