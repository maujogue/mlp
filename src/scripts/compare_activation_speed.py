#!/usr/bin/env python3
"""Compare training speed: ReLU vs Sigmoid activation (same epochs, report % difference)."""

import numpy as np
from pathlib import Path

from mlp.data.data_engineering import (
    fit_scaler_on_train_and_transform_train_val,
    fix_dataset,
    split_features_labels,
    split_train_validation,
)
from mlp.model.mlp_classifier import MLPClassifier
from mlp.model.schemas import TrainingRunConfig
from mlp.utils.constants import FEATURE_COLUMNS
from mlp.utils.loader import load_dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ReLU vs Sigmoid training speed."
    )
    parser.add_argument(
        "data_path",
        type=Path,
        help="CSV with 'label' and feature columns (default: 42_evaluation/data_prepared.csv)",
    )
    parser.add_argument(
        "--epochs", type=int, default=30, help="Epochs per run (default: 30)"
    )
    parser.add_argument(
        "--batch", type=int, default=0, help="Batch size (0 = full batch)"
    )
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--optimizer", default="rmsprop", choices=["sgd", "rmsprop"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=[32, 16], help="Hidden layer sizes"
    )
    parser.add_argument("--patience", type=int, default=0, help="Patience")
    args = parser.parse_args()

    # Load and prepare data (same pipeline as train_cmd)
    path: Path = args.data_path
    if not path.exists():
        print(f"Data file not found: {path}. Using synthetic data.")
        n_samples = 500
        n_features = len(FEATURE_COLUMNS)
        X_train = np.random.standard_normal((n_samples, n_features)).astype(np.float64)
        y_train = np.random.randint(0, 2, size=n_samples, dtype=np.int64)
    else:
        df = load_dataset(path)
        if "label" not in df.columns or not all(
            c in df.columns for c in FEATURE_COLUMNS
        ):
            df = fix_dataset(df)
        train_df, val_df = split_train_validation(df, 0.2)
        scaler_path: Path = Path("/tmp/compare_activation_scaler.pkl")
        train_df, _ = fit_scaler_on_train_and_transform_train_val(
            train_df, val_df, scaler_path
        )
        X_df, y_mapped = split_features_labels(train_df)
        X_train = X_df.to_numpy(dtype=np.float64)
        y_train = y_mapped.astype(np.int64).values

    n_features = X_train.shape[1]
    hidden = args.layers

    run_config = TrainingRunConfig(
        train_path=args.data_path,
        val_ratio=0.2,
        layers=hidden,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        patience=args.patience,
    )
    # ReLU
    model_relu = MLPClassifier(
        n_features=n_features,
        hidden_layers=hidden,
        output_size=2,
        activation="relu",
    )
    model_relu.fit(
        X_train,
        y_train,
        run_config=run_config,
    )
    t_relu = model_relu.last_fit_seconds or 0.0

    # Sigmoid
    model_sigmoid = MLPClassifier(
        n_features=n_features,
        hidden_layers=hidden,
        output_size=2,
        seed=args.seed,
        activation="sigmoid",
    )
    run_config = TrainingRunConfig(
        train_path=args.data_path,
        val_ratio=0.2,
        layers=hidden,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        seed=args.seed,
        patience=args.patience,
    )
    model_sigmoid.fit(X_train, y_train, run_config=run_config)
    t_sigmoid = model_sigmoid.last_fit_seconds or 0.0

    # Report
    print(f"Data: {X_train.shape[0]} samples, {n_features} features")
    print(
        f"Epochs: {args.epochs}  Batch: {args.batch or 'full'}  Optimizer: {args.optimizer}"
    )
    print(f"ReLU:    {t_relu:.3f} s")
    print(f"Sigmoid: {t_sigmoid:.3f} s")
    if t_relu > 0:
        pct = (t_sigmoid - t_relu) / t_relu * 100
        if pct >= 0:
            print(f"Sigmoid is {pct:.1f}% slower than ReLU.")
        else:
            print(f"Sigmoid is {-pct:.1f}% faster than ReLU.")


if __name__ == "__main__":
    main()
