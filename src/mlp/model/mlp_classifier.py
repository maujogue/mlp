"""Classifier wrapper around the generic NumPy MLP network."""

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from mlp_core.mlp_classifier import (
    MLPClassifier as _CoreMLPClassifier,
    softmax_cross_entropy_grad,
    softmax_cross_entropy_loss,
)
from .schemas import TrainingHistory, TrainingMetrics, TrainingRunConfig
from .telemetry import TrainingTelemetryOptions

# Training-matrix row (0-based, original order before each epoch's shuffle) used as the
# single exemplar for lesson-replay forward/loss/backward traces. When that row is not in
# the current minibatch, those micro-steps are omitted for that batch (optimizer still runs).
LESSON_REPLAY_ANCHOR_TRAIN_INDEX = 0



def _softmax_row(logits_row: np.ndarray) -> np.ndarray:
    """Stable softmax for a single logit vector (C,)."""
    shift = float(np.max(logits_row))
    exp = np.exp(logits_row - shift)
    return exp / np.sum(exp)


class MLPClassifier(_CoreMLPClassifier):
    """Classification MLP using softmax cross-entropy training."""

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        run_config: TrainingRunConfig,
        telemetry: TrainingTelemetryOptions | None = None,
        lesson_hook: Callable[[dict[str, Any]], None] | None = None,
        on_lesson_batch_end: Callable[[], None] | None = None,
        on_lesson_epoch_end: Callable[[], None] | None = None,
    ) -> TrainingHistory:
        """Train the model and return epoch-wise metrics history."""
        from .evaluation import evaluate

        fit_start_time = time.perf_counter()
        X_train_arr = np.asarray(X_train, dtype=np.float64)
        y_train_arr = np.asarray(y_train, dtype=np.int64)
        if X_train_arr.ndim != 2:
            raise ValueError("X_train must be a 2D array.")
        if len(X_train_arr) != len(y_train_arr):
            raise ValueError("X_train and y_train must have the same length.")
        if len(X_train_arr) == 0:
            raise ValueError("X_train must be non-empty.")
        if X_train_arr.shape[1] != self.input_size:
            raise ValueError("X_train feature count must match input_size.")
        if np.any(y_train_arr < 0) or np.any(y_train_arr >= self.output_size):
            raise ValueError(
                "y_train contains class labels outside the model output range."
            )

        has_val = X_val is not None and y_val is not None
        X_val_arr: np.ndarray | None = None
        y_val_arr: np.ndarray | None = None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            if X_val_arr.ndim != 2:
                raise ValueError("X_val must be a 2D array.")
            if len(X_val_arr) != len(y_val_arr):
                raise ValueError("X_val and y_val must have the same length.")
            if X_val_arr.shape[1] != self.input_size:
                raise ValueError("X_val feature count must match input_size.")
            if np.any(y_val_arr < 0) or np.any(y_val_arr >= self.output_size):
                raise ValueError(
                    "y_val contains class labels outside the model output range."
                )

        history: TrainingHistory = TrainingHistory()

        n_train = len(X_train_arr)
        effective_batch_size = (
            run_config.batch_size if run_config.batch_size > 0 else n_train
        )
        rng_seed = self.seed if run_config.seed is None else run_config.seed
        rng = np.random.default_rng(rng_seed)

        best_loss = np.inf
        epochs_no_improve = 0
        best_weights: list[tuple[np.ndarray, np.ndarray]] | None = None

        n_batches = (n_train + effective_batch_size - 1) // effective_batch_size

        self._lesson_hook = lesson_hook
        try:
            for epoch in range(1, run_config.epochs + 1):
                indices = rng.permutation(n_train)
                batch_num = 0
                for start in range(0, n_train, effective_batch_size):
                    batch_idx = indices[start : start + effective_batch_size]
                    X_batch = X_train_arr[batch_idx]
                    y_batch = y_train_arr[batch_idx]
                    anchor_idx = int(
                        np.clip(
                            LESSON_REPLAY_ANCHOR_TRAIN_INDEX, 0, max(0, n_train - 1)
                        ),
                    )
                    hits = np.flatnonzero(batch_idx == anchor_idx)
                    trace_this_batch = hits.size > 0
                    exemplar_slot = int(hits[0]) if trace_this_batch else 0
                    self._lesson_meta = {
                        "epoch": epoch,
                        "batch": batch_num,
                        "exemplar": exemplar_slot,
                        "learning_rate": float(run_config.learning_rate),
                        "lesson_anchor_train_index": anchor_idx,
                        "lesson_trace_this_batch": trace_this_batch,
                    }
                    self.zero_grad()
                    if lesson_hook is not None and not trace_this_batch:
                        self._lesson_hook = None
                    try:
                        logits = self.forward(X_batch)
                    finally:
                        self._lesson_hook = lesson_hook
                    d_logits = softmax_cross_entropy_grad(logits, y_batch)
                    if lesson_hook is not None and trace_this_batch:
                        ex = exemplar_slot
                        probs = _softmax_row(logits[ex])
                        y_ex = int(y_batch[ex])
                        ce = float(-np.log(probs[y_ex] + 1e-15))
                        pred = int(np.argmax(probs))
                        lesson_hook(
                            {
                                **self._lesson_meta,
                                "phase": "loss",
                                "toc_id": "loss",
                                "sample_in_batch": ex,
                                "logits": logits[ex].tolist(),
                                "probs": probs.tolist(),
                                "label": y_ex,
                                "loss_contribution": ce,
                                "pred_class": pred,
                                "correct": pred == y_ex,
                                "loss_batch_mean": float(
                                    softmax_cross_entropy_loss(logits, y_batch),
                                ),
                                "math": (
                                    r"For one example: $C = -\log p_y$ where "
                                    r"$p=\mathrm{softmax}(z)$. "
                                    r"Gradients on logits: $\partial C/\partial z_k = p_k - \mathbb{1}_{k=y}$ "
                                    r"(averaged over the minibatch in code)."
                                ),
                            },
                        )
                    if lesson_hook is not None and not trace_this_batch:
                        self._lesson_hook = None
                    try:
                        self.backward(d_logits)
                    finally:
                        self._lesson_hook = lesson_hook

                    emit_batch = (
                        telemetry is not None
                        and telemetry.callback is not None
                        and telemetry.should_emit_batch(batch_num)
                    )
                    if emit_batch:
                        loss_batch = softmax_cross_entropy_loss(logits, y_batch)
                        grad_norm_per_layer = [
                            float(
                                np.sqrt(
                                    np.sum(self._grad_W[i] ** 2)
                                    + np.sum(self._grad_b[i] ** 2)
                                )
                            )
                            for i in range(len(self._layers))
                        ]
                        W_snap = [W.copy() for W, _ in self._layers]
                        b_snap = [b.copy() for _, b in self._layers]
                        self.step(
                            learning_rate=run_config.learning_rate,
                            optimizer=run_config.optimizer,
                        )
                        weight_delta_norm_per_layer = []
                        for i in range(len(self._layers)):
                            d_w = self._layers[i][0] - W_snap[i]
                            d_b = self._layers[i][1] - b_snap[i]
                            weight_delta_norm_per_layer.append(
                                float(np.sqrt(np.sum(d_w * d_w) + np.sum(d_b * d_b)))
                            )
                        assert telemetry is not None and telemetry.callback is not None
                        telemetry.callback(
                            "batch",
                            {
                                "epoch": epoch,
                                "batch_index": batch_num,
                                "n_batches": n_batches,
                                "loss": loss_batch,
                                "grad_norm_per_layer": grad_norm_per_layer,
                                "weight_delta_norm_per_layer": weight_delta_norm_per_layer,
                            },
                        )
                    else:
                        self.step(
                            learning_rate=run_config.learning_rate,
                            optimizer=run_config.optimizer,
                        )
                    if lesson_hook is not None:
                        lesson_hook(
                            {
                                **self._lesson_meta,
                                "phase": "batch_end",
                                "toc_id": "batch_end",
                                "sample_in_batch": 0,
                            },
                        )
                    if on_lesson_batch_end is not None:
                        on_lesson_batch_end()
                    batch_num += 1

                if on_lesson_epoch_end is not None:
                    on_lesson_epoch_end()
                if lesson_hook is not None:
                    lesson_hook(
                        {
                            "phase": "epoch_end",
                            "toc_id": "epoch_end",
                            "epoch": epoch,
                            "batch": max(0, batch_num - 1),
                            "sample_in_batch": 0,
                        },
                    )

                train_metrics: TrainingMetrics = evaluate(
                    self, X_train_arr, y_train_arr
                )
                history.train_loss.append(train_metrics.loss)
                history.train_accuracy.append(train_metrics.accuracy)
                history.train_precision.append(train_metrics.precision)
                history.train_recall.append(train_metrics.recall)
                history.train_f1.append(train_metrics.f1)

                if has_val and X_val_arr is not None and y_val_arr is not None:
                    val_metrics: TrainingMetrics = evaluate(self, X_val_arr, y_val_arr)
                    history.val_loss.append(val_metrics.loss)
                    history.val_accuracy.append(val_metrics.accuracy)
                    history.val_precision.append(val_metrics.precision)
                    history.val_recall.append(val_metrics.recall)
                    history.val_f1.append(val_metrics.f1)
                    monitor_loss = val_metrics.loss
                else:
                    monitor_loss = train_metrics.loss

                if telemetry is not None and telemetry.callback is not None:
                    val_payload: dict | None = None
                    if has_val and history.val_loss:
                        val_payload = {
                            "loss": history.val_loss[-1],
                            "accuracy": history.val_accuracy[-1],
                            "precision": history.val_precision[-1],
                            "recall": history.val_recall[-1],
                            "f1": history.val_f1[-1],
                        }
                    telemetry.callback(
                        "epoch",
                        {
                            "epoch": epoch,
                            "train": {
                                "loss": train_metrics.loss,
                                "accuracy": train_metrics.accuracy,
                                "precision": train_metrics.precision,
                                "recall": train_metrics.recall,
                                "f1": train_metrics.f1,
                            },
                            "val": val_payload,
                        },
                    )

                if has_val and X_val_arr is not None and y_val_arr is not None:
                    print(
                        f"epoch {epoch:02d}/{run_config.epochs} - "
                        f"loss: {train_metrics.loss:.4f} - acc: {train_metrics.accuracy:.4f} - "
                        f"prec: {train_metrics.precision:.4f} - rec: {train_metrics.recall:.4f} - "
                        f"f1: {train_metrics.f1:.4f} - val_loss: {history.val_loss[-1]:.4f} - "
                        f"val_acc: {history.val_accuracy[-1]:.4f} - val_prec: {history.val_precision[-1]:.4f} - "
                        f"val_rec: {history.val_recall[-1]:.4f} - val_f1: {history.val_f1[-1]:.4f}"
                    )
                else:
                    print(
                        f"epoch {epoch:02d}/{run_config.epochs} - "
                        f"loss: {train_metrics.loss:.4f} - acc: {train_metrics.accuracy:.4f} - "
                        f"prec: {train_metrics.precision:.4f} - rec: {train_metrics.recall:.4f} - "
                        f"f1: {train_metrics.f1:.4f}"
                    )

                if run_config.patience > 0:
                    if monitor_loss < best_loss:
                        best_loss = monitor_loss
                        epochs_no_improve = 0
                        best_weights = [
                            (W.copy(), b.copy()) for W, b in self.parameters()
                        ]
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= run_config.patience:
                            print(
                                f"Early stopping at epoch {epoch} (no improvement for {run_config.patience} epochs)."
                            )
                            break
        finally:
            self._lesson_hook = None
            self._lesson_meta = None

        if best_weights is not None:
            for i in range(len(self._layers)):
                np.copyto(self._layers[i][0], best_weights[i][0])
                np.copyto(self._layers[i][1], best_weights[i][1])

        self.last_fit_seconds = time.perf_counter() - fit_start_time

        if (
            telemetry is not None
            and telemetry.callback is not None
            and not telemetry.defer_fit_done_callback
        ):
            telemetry.callback(
                "done",
                {
                    "elapsed_seconds": self.last_fit_seconds,
                    "epochs_ran": len(history.train_loss),
                    "history": history.model_dump(by_alias=True),
                },
            )

        return TrainingHistory.model_validate(history)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Batch class probabilities (B, output_size). Softmax over last axis."""
        logits = self.logits(X)
        shift = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - shift)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch class prediction (B,), using the largest output probability."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return np.argmax(self.predict_proba(X_arr), axis=1).astype(np.int64)
