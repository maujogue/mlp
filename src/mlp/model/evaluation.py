from pathlib import Path
from typing import Protocol, Union

import numpy as np

from ..data.data_engineering import fix_dataset, scale_features, split_features_labels
from ..utils.constants import FEATURE_COLUMNS
from ..utils.loader import load_dataset
from .schemas import TrainingMetrics
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


def multiclass_precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> tuple[float, float, float]:
    """Macro precision, recall, and F1 for integer class labels."""
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for class_idx in range(n_classes):
        pred_class = y_pred == class_idx
        true_class = y_true == class_idx
        tp = float(np.sum(pred_class & true_class))
        fp = float(np.sum(pred_class & ~true_class))
        fn = float(np.sum(~pred_class & true_class))
        precision = tp / (tp + fp) if (tp + fp) > 0.0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0.0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0.0
            else 0.0
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1
    return precision_sum / n_classes, recall_sum / n_classes, f1_sum / n_classes


def softmax_cross_entropy_from_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """Mean softmax cross-entropy from class probabilities."""
    y_idx = np.asarray(y_true, dtype=np.intp)
    p = np.asarray(probabilities, dtype=np.float64)
    eps = 1e-12
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], eps, 1.0))))


def evaluate(
    model: _ProbabilisticModel,
    X: np.ndarray,
    y: np.ndarray,
) -> TrainingMetrics:
    """Evaluate a model on one dataset with vectorized metrics."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    y_arr = np.asarray(y, dtype=np.intp)
    probabilities = model.predict_proba(X_arr)
    n_classes = probabilities.shape[1]
    if np.any(y_arr < 0) or np.any(y_arr >= n_classes):
        raise ValueError("y contains class labels outside the model output range.")
    predictions = np.argmax(probabilities, axis=1)
    loss = softmax_cross_entropy_from_probabilities(y_arr, probabilities)
    accuracy = float(np.mean(predictions == y_arr))
    if n_classes == 2:
        precision, recall, f1 = precision_recall_f1(y_arr, probabilities[:, 1])
    else:
        precision, recall, f1 = multiclass_precision_recall_f1(
            y_arr,
            predictions,
            n_classes,
        )
    return TrainingMetrics(
        loss=loss,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _load_and_preprocess_for_predict(
    dataset_path: Path,
    scaler_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    df = load_dataset(dataset_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    df = scale_features(df, training=False, scaler_path=scaler_path)
    X_df, y_mapped = split_features_labels(df)
    return X_df.to_numpy(dtype=np.float64), y_mapped.astype(np.int64).values


def evaluate_model_on_dataset(
    model_path_or_dir: Path,
    dataset_path: Path,
) -> TrainingMetrics:
    model_dir = (
        Path(model_path_or_dir)
        if Path(model_path_or_dir).is_dir()
        else Path(model_path_or_dir).parent
    )
    model = load_model(model_dir)
    scaler_path = model_dir / "scaler.pkl"
    X, y = _load_and_preprocess_for_predict(dataset_path, scaler_path)
    metrics: TrainingMetrics = evaluate(model, X, y)
    return metrics


def evaluate_model_on_datasets(
    model_path_or_dir: Path,
    dataset_paths: list[Path],
) -> TrainingMetrics:
    if not dataset_paths:
        raise ValueError("dataset_paths must be non-empty")
    metrics: TrainingMetrics = TrainingMetrics()
    for path in dataset_paths:
        m: TrainingMetrics = evaluate_model_on_dataset(model_path_or_dir, path)
        metrics.loss += m.loss
        metrics.accuracy += m.accuracy
        metrics.precision += m.precision
        metrics.recall += m.recall
        metrics.f1 += m.f1
    metrics.loss /= len(dataset_paths)
    metrics.accuracy /= len(dataset_paths)
    metrics.precision /= len(dataset_paths)
    metrics.recall /= len(dataset_paths)
    metrics.f1 /= len(dataset_paths)
    return metrics
