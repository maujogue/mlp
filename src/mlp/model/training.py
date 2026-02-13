from typing import Union

import numpy as np

from ..data.data_engineering import split_features_labels
from ..utils.constants import FEATURE_COLUMNS, LABELS
from ..utils.loader import load_dataset
from .model import MLPClassifier
from .plots import save_learning_curves
from .serialization import load_model, save_model

INDEX_TO_LABEL = {v: k for k, v in LABELS.items()}


def _load_split_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load CSV once and convert to NumPy arrays (no pandas in training loop)."""
    df = load_dataset(path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        raise ValueError(
            f"CSV must contain 'label' and feature columns. Got: {list(df.columns)[:5]}..."
        )
    X_df, y_mapped = split_features_labels(df)
    return X_df.to_numpy(dtype=np.float64), y_mapped.astype(np.int64).values


def _binary_cross_entropy_from_probabilities(
    y_true: Union[list[int], np.ndarray],
    p_positive: Union[list[float], np.ndarray],
) -> float:
    """Vectorized BCE: single NumPy expression, no Python loop over samples."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p_positive, dtype=np.float64)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _precision_recall_f1(
    y_true: np.ndarray,
    pred_positive: np.ndarray,
) -> tuple[float, float, float]:
    """Binary precision, recall, F1 (positive class = 1). Vectorized."""
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


def _evaluate_dataset(
    model: MLPClassifier,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Eval once per epoch: single batch forward, vectorized metrics."""
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(1, -1)
    p = model.predict_proba(X_arr)[:, 1]  # p(positive)
    y_arr = np.asarray(y, dtype=np.float64)
    bce = _binary_cross_entropy_from_probabilities(y_arr, p)
    accuracy = float(np.mean((p >= 0.5) == y_arr))
    precision, recall, f1 = _precision_recall_f1(y_arr, p)
    return bce, accuracy, precision, recall, f1


def _margin_loss(logits: np.ndarray, y: np.ndarray) -> float:
    """Multiclass hinge: mean over batch of relu(1 + logit_wrong - logit_correct)."""
    B = logits.shape[0]
    y = np.asarray(y, dtype=np.intp)
    correct = logits[np.arange(B), y]
    wrong = logits[np.arange(B), 1 - y]
    margins = np.maximum(0.0, 1.0 + wrong - correct)
    return float(np.mean(margins))


def _margin_loss_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of margin loss w.r.t. logits (B, 2)."""
    B = logits.shape[0]
    y = np.asarray(y, dtype=np.intp)
    correct = logits[np.arange(B), y]
    wrong = logits[np.arange(B), 1 - y]
    margins = np.maximum(0.0, 1.0 + wrong - correct)
    mask = (margins > 0).astype(np.float64)
    d_logits = np.zeros_like(logits)
    d_logits[np.arange(B), y] = -mask / B
    d_logits[np.arange(B), 1 - y] = mask / B
    return d_logits


def train_cmd(
    train_path: str = "datasets/train.csv",
    val_path: str = "datasets/val.csv",
    layers: list[int] | None = None,
    epochs: int = 70,
    learning_rate: float = 0.01,
    seed: int = 42,
    model_path: str = "weights/model",
    curves_dir: str = "figures/training",
    batch_size: int = 0,
    optimizer: str = "sgd",
    patience: int = 0,
    min_delta: float = 0.0,
) -> None:
    # batch_size=0 means full dataset per step (full-batch gradient descent)
    # patience=0 means early stopping disabled
    X_train, y_train = _load_split_csv(train_path)
    X_val, y_val = _load_split_csv(val_path)

    hidden = layers or [24, 24]
    model = MLPClassifier(
        n_features=len(FEATURE_COLUMNS),
        hidden_layers=hidden,
        output_size=2,
        seed=seed,
    )

    history_train_loss: list[float] = []
    history_val_loss: list[float] = []
    history_train_acc: list[float] = []
    history_val_acc: list[float] = []
    history_train_precision: list[float] = []
    history_val_precision: list[float] = []
    history_train_recall: list[float] = []
    history_val_recall: list[float] = []
    history_train_f1: list[float] = []
    history_val_f1: list[float] = []

    best_val_loss: float = np.inf
    epochs_no_improve: int = 0
    best_weights: list[tuple[np.ndarray, np.ndarray]] | None = None

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")

    n_train = len(X_train)
    effective_batch_size = batch_size if batch_size > 0 else n_train
    for epoch in range(1, epochs + 1):
        indices = np.random.default_rng(seed + epoch).permutation(n_train)

        for start in range(0, n_train, effective_batch_size):
            batch_idx = indices[start : start + effective_batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]
            model.zero_grad()
            logits = model.forward(X_batch)
            d_logits = _margin_loss_grad(logits, y_batch)
            model.backward(d_logits)
            model.step(
                learning_rate=learning_rate,
                optimizer=optimizer,
            )

        train_loss, train_acc, train_prec, train_rec, train_f1 = _evaluate_dataset(
            model, X_train, y_train
        )
        val_loss, val_acc, val_prec, val_rec, val_f1 = _evaluate_dataset(
            model, X_val, y_val
        )
        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_train_acc.append(train_acc)
        history_val_acc.append(val_acc)
        history_train_precision.append(train_prec)
        history_val_precision.append(val_prec)
        history_train_recall.append(train_rec)
        history_val_recall.append(val_rec)
        history_train_f1.append(train_f1)
        history_val_f1.append(val_f1)

        # Early stopping: track best validation loss and save weights
        if patience > 0:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_weights = [(W.copy(), b.copy()) for W, b in model.parameters()]
            else:
                epochs_no_improve += 1

        print(
            f"epoch {epoch:02d}/{epochs} - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"prec: {train_prec:.4f} - rec: {train_rec:.4f} - f1: {train_f1:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - "
            f"val_prec: {val_prec:.4f} - val_rec: {val_rec:.4f} - val_f1: {val_f1:.4f}"
        )

        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    if best_weights is not None:
        for i in range(len(model._layers)):
            np.copyto(model._layers[i][0], best_weights[i][0])
            np.copyto(model._layers[i][1], best_weights[i][1])

    save_model(model, model_path)
    save_learning_curves(
        history_train_loss,
        history_val_loss,
        history_train_acc,
        history_val_acc,
        history_train_precision,
        history_val_precision,
        history_train_recall,
        history_val_recall,
        history_train_f1,
        history_val_f1,
        curves_dir,
    )
    print(f"Training figures saved to {curves_dir}")


def predict_cmd(
    dataset_path: str = "datasets/val.csv",
    model_path: str = "weights/model",
    output_path: str | None = None,
) -> None:
    model, _ = load_model(model_path)
    X, y = _load_split_csv(dataset_path)

    p_arr = model.predict_proba(X)[:, 1]
    p_positive = p_arr.tolist()

    if output_path:
        df = load_dataset(dataset_path)
        df = df.copy()
        df["predicted_label"] = [
            INDEX_TO_LABEL[1 if p >= 0.5 else 0] for p in p_positive
        ]
        df["proba_m"] = p_positive
        df["proba_b"] = [1.0 - p for p in p_positive]
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    bce = _binary_cross_entropy_from_probabilities(y, p_arr)
    accuracy = float(np.mean((p_arr >= 0.5) == np.asarray(y)))
    print(f"binary_cross_entropy: {bce:.6f}")
    print(f"accuracy: {accuracy:.6f}")
