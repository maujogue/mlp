#!/usr/bin/env python3
"""Compare training speed: ReLU vs Sigmoid activation (same epochs, report % difference)."""

import time

import numpy as np


from mlp.model.model import MLPClassifier
from mlp.model.training import (
    _margin_loss_grad,
    fit_scaler_on_train_and_transform_train_val,
    fix_dataset,
    split_features_labels,
    split_train_validation,
)
from mlp.utils.constants import FEATURE_COLUMNS
from mlp.utils.loader import load_dataset


def run_training_epochs(
    model: MLPClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer: str,
    seed: int,
) -> float:
    """Run a fixed number of epochs; return elapsed time in seconds."""
    n_train = len(X_train)
    effective_batch = batch_size if batch_size > 0 else n_train
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    for epoch in range(epochs):
        indices = rng.permutation(n_train)
        for start_idx in range(0, n_train, effective_batch):
            batch_idx = indices[start_idx : start_idx + effective_batch]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            model.zero_grad()
            logits = model.forward(X_batch)
            d_logits = _margin_loss_grad(logits, y_batch)
            model.backward(d_logits)
            model.step(learning_rate=learning_rate, optimizer=optimizer)
    return time.perf_counter() - start


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Compare ReLU vs Sigmoid training speed.")
    parser.add_argument(
        "--data",
        default="42_evaluation/data_prepared.csv",
        help="CSV with 'label' and feature columns (default: 42_evaluation/data_prepared.csv)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per run (default: 30)")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0 = full batch)")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "rmsprop"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layers", type=int, nargs="+", default=[32, 16], help="Hidden layer sizes")
    args = parser.parse_args()

    # Load and prepare data (same pipeline as train_cmd)
    from pathlib import Path
    path = Path(args.data)
    if not path.exists():
        print(f"Data file not found: {path}. Using synthetic data.")
        n_samples = 500
        n_features = len(FEATURE_COLUMNS)
        X_train = np.random.standard_normal((n_samples, n_features)).astype(np.float64)
        y_train = np.random.randint(0, 2, size=n_samples, dtype=np.int64)
    else:
        df = load_dataset(str(path))
        if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
            df = fix_dataset(df)
        train_df, val_df = split_train_validation(df, split=0.2)
        scaler_path = "/tmp/compare_activation_scaler.pkl"
        train_df, _ = fit_scaler_on_train_and_transform_train_val(
            train_df, val_df, scaler_path
        )
        X_df, y_mapped = split_features_labels(train_df)
        X_train = X_df.to_numpy(dtype=np.float64)
        y_train = y_mapped.astype(np.int64).values

    n_features = X_train.shape[1]
    hidden = args.layers

    # ReLU
    model_relu = MLPClassifier(
        n_features=n_features,
        hidden_layers=hidden,
        output_size=2,
        seed=args.seed,
        activation="relu",
    )
    t_relu = run_training_epochs(
        model_relu, X_train, y_train,
        epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, optimizer=args.optimizer, seed=args.seed,
    )

    # Sigmoid
    model_sigmoid = MLPClassifier(
        n_features=n_features,
        hidden_layers=hidden,
        output_size=2,
        seed=args.seed,
        activation="sigmoid",
    )
    t_sigmoid = run_training_epochs(
        model_sigmoid, X_train, y_train,
        epochs=args.epochs, batch_size=args.batch,
        learning_rate=args.lr, optimizer=args.optimizer, seed=args.seed,
    )

    # Report
    print(f"Data: {X_train.shape[0]} samples, {n_features} features")
    print(f"Epochs: {args.epochs}  Batch: {args.batch or 'full'}  Optimizer: {args.optimizer}")
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
