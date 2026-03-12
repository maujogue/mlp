from pathlib import Path
from typing import Protocol, Union

import numpy as np

from ..data.data_engineering import fix_dataset, scale_features, split_features_labels
from ..utils.constants import FEATURE_COLUMNS
from ..utils.loader import load_dataset
from .serialization import load_model


class _ProbabilisticModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


def binary_cross_entropy_from_probabilities(
    y_true: Union[list[int], np.ndarray],
    p_positive: Union[list[float], np.ndarray],
) -> float:
    """Vectorized binary cross-entropy from positive-class probabilities."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_positive, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def precision_recall_f1(
    y_true: Union[list[int], np.ndarray],
    pred_positive: Union[list[float], np.ndarray],
) -> tuple[float, float, float]:
    """Binary precision, recall, F1 (positive class = 1), vectorized."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = (np.asarray(pred_positive, dtype=np.float64) >= 0.5).astype(np.float64)
    tp = float(np.sum(pred * y))
    fp = float(np.sum(pred * (1.0 - y)))
    fn = float(np.sum((1.0 - pred) * y))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def evaluate(
    model: _ProbabilisticModel,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Evaluate a model on one dataset with vectorized metrics."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    y_arr = np.asarray(y, dtype=np.float64)
    p = model.predict_proba(X_arr)[:, 1]
    loss = binary_cross_entropy_from_probabilities(y_arr, p)
    accuracy = float(np.mean((p >= 0.5) == y_arr))
    precision, recall, f1 = precision_recall_f1(y_arr, p)
    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _load_and_preprocess_for_predict(
    dataset_path: str,
    scaler_path: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_dataset(dataset_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    df = scale_features(df, training=False, scaler_path=scaler_path)
    X_df, y_mapped = split_features_labels(df)
    return X_df.to_numpy(dtype=np.float64), y_mapped.astype(np.int64).values


def evaluate_model_on_dataset(
    model_path_or_dir: str,
    dataset_path: str,
) -> dict[str, float]:
    model_dir = (
        Path(model_path_or_dir)
        if Path(model_path_or_dir).is_dir()
        else Path(model_path_or_dir).parent
    )
    model, _ = load_model(str(model_dir))
    scaler_path = str(model_dir / "scaler.pkl")
    X, y = _load_and_preprocess_for_predict(dataset_path, scaler_path)
    metrics = evaluate(model, X, y)
    return {
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
    }


def evaluate_model_on_datasets(
    model_path_or_dir: str,
    dataset_paths: list[str],
) -> dict[str, float]:
    if not dataset_paths:
        raise ValueError("dataset_paths must be non-empty")
    keys = ["test_loss", "test_accuracy", "test_precision", "test_recall", "test_f1"]
    sums: dict[str, float] = {k: 0.0 for k in keys}
    for path in dataset_paths:
        m = evaluate_model_on_dataset(model_path_or_dir, path)
        for k in keys:
            sums[k] += m[k]
    n = len(dataset_paths)
    return {k: sums[k] / n for k in keys}
