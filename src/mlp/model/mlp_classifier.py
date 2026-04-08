"""Vectorized NumPy MLP for binary classification (no micrograd)."""

import time
from collections.abc import Callable
from typing import Any, Literal, cast

import numpy as np

from .schemas import TrainingHistory, TrainingMetrics, TrainingRunConfig
from .telemetry import TrainingTelemetryOptions
from ..utils.constants import SEED

# Training-matrix row (0-based, original order before each epoch's shuffle) used as the
# single exemplar for lesson-replay forward/loss/backward traces. When that row is not in
# the current minibatch, those micro-steps are omitted for that batch (optimizer still runs).
LESSON_REPLAY_ANCHOR_TRAIN_INDEX = 0


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def softmax_cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean softmax cross-entropy loss for integer class labels (matches backward)."""
    batch_size = logits.shape[0]
    y_arr = np.asarray(y, dtype=np.intp)
    shift = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - shift)
    log_sum_exp = np.log(exp.sum(axis=1, keepdims=True)) + shift
    log_probs = logits - log_sum_exp
    nll = -log_probs[np.arange(batch_size), y_arr]
    return float(np.mean(nll))


def softmax_cross_entropy_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of softmax cross-entropy w.r.t. logits (B, 2)."""
    batch_size = logits.shape[0]
    y_arr = np.asarray(y, dtype=np.intp)
    shift = logits.max(axis=1, keepdims=True)
    exp = np.exp(logits - shift)
    proba = exp / exp.sum(axis=1, keepdims=True)
    d_logits = proba
    d_logits[np.arange(batch_size), y_arr] -= 1.0
    d_logits /= batch_size
    return d_logits


def _softmax_row(logits_row: np.ndarray) -> np.ndarray:
    """Stable softmax for a single logit vector (C,)."""
    shift = float(np.max(logits_row))
    exp = np.exp(logits_row - shift)
    return exp / np.sum(exp)


class MLPClassifier:
    """NumPy MLP: batch forward/backward, ReLU or sigmoid hidden, 2-class output."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 2,
        seed: int = SEED,
        activation: str = "relu",
    ) -> None:
        hidden_layers = list(hidden_layers or [24, 24])
        if len(hidden_layers) < 1:
            raise ValueError("At least one hidden layer is required.")
        if output_size != 2:
            raise ValueError("Only binary classification with 2 outputs is supported.")

        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.seed = seed
        act = activation.lower()
        if act not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'")
        self._activation: Literal["relu", "sigmoid"] = cast(
            Literal["relu", "sigmoid"], act
        )

        self._layers: list[tuple[np.ndarray, np.ndarray]] = []  # weights and biases
        self._init_weights_and_biases()

        # Forward cache for backward; gradients (set by zero_grad)
        self._cache: list[tuple[np.ndarray, ...]] = []
        self._grad_W: list[np.ndarray] = []
        self._grad_b: list[np.ndarray] = []
        # RMSprop state: running average of squared gradients (lazy init)
        self._rms_W: list[np.ndarray] = []
        self._rms_b: list[np.ndarray] = []
        # Total duration of the last fit() call in seconds.
        self.last_fit_seconds: float | None = None
        # Optional lesson replay hooks (set only during fit when requested).
        self._lesson_hook: Callable[[dict[str, Any]], None] | None = None
        self._lesson_meta: dict[str, Any] | None = None

    def _init_weights_and_biases(self) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [self.n_features, *self.hidden_layers, self.output_size]
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            if i < len(dims) - 2:
                # Match hidden-layer initialization to activation to stabilize gradients.
                if self._activation == "relu":
                    # Kaiming/He normal initialization.
                    scale = np.sqrt(2.0 / in_d)
                else:
                    # Xavier/Glorot normal initialization.
                    scale = np.sqrt(2.0 / (in_d + out_d))
            else:
                scale = 0.1
            W = rng.standard_normal((in_d, out_d)).astype(np.float64) * scale
            b = np.zeros(out_d, dtype=np.float64)
            self._layers.append((W, b))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Batch forward: X (B, n_features) -> logits (B, output_size)."""
        self._cache = []
        out = X
        hook = self._lesson_hook
        meta = self._lesson_meta
        ex = int(meta["exemplar"]) if meta is not None else 0
        if hook is not None and meta is not None:
            hook(
                {
                    **meta,
                    "phase": "forward_input",
                    "toc_id": "forward_input",
                    "sample_in_batch": ex,
                    "a_in": X[ex].tolist(),
                },
            )
        for i, (W, b) in enumerate(self._layers):
            Z = out @ W + b  # (B, out_d)
            if hook is not None and meta is not None:
                a_prev = out[ex]
                contrib = (a_prev.reshape(-1, 1) * W).tolist()
                hook(
                    {
                        **meta,
                        "phase": "forward_layer",
                        "toc_id": f"fwd_l{i}",
                        "layer": i,
                        "sample_in_batch": ex,
                        "a_in": a_prev.tolist(),
                        "W": W.tolist(),
                        "b": b.tolist(),
                        "z": Z[ex].tolist(),
                        "edge_contributions": contrib,
                    },
                )
            if i < len(self._layers) - 1:
                self._cache.append((out, Z))
                z_row = Z[ex].copy()
                out = np.maximum(0, Z) if self._activation == "relu" else _sigmoid(Z)
                if hook is not None and meta is not None:
                    hook(
                        {
                            **meta,
                            "phase": "activation",
                            "toc_id": f"act_l{i}",
                            "layer": i,
                            "sample_in_batch": ex,
                            "z": z_row.tolist(),
                            "a_out": out[ex].tolist(),
                        },
                    )
            else:
                self._cache.append((out, Z))
                out = Z
        return out

    def backward(self, d_logits: np.ndarray) -> None:
        """Backprop: d_logits (B, output_size). Accumulates gradients in-place."""
        d_out = d_logits
        hook = self._lesson_hook
        meta = self._lesson_meta
        ex = int(meta["exemplar"]) if meta is not None else 0
        for i in range(len(self._layers) - 1, -1, -1):
            X_in, Z = self._cache[i]
            W, b = self._layers[i]
            if i == len(self._layers) - 1:
                d_Z = d_out
            else:
                if self._activation == "relu":
                    d_Z = d_out * (Z > 0).astype(np.float64)
                else:
                    s = _sigmoid(Z)
                    d_Z = d_out * s * (1.0 - s)  # sigmoid derivative
            # d_out was d_L/d_(output of this layer). d_Z = d_L/d_Z.
            # Z = X_in @ W + b  =>  dW = X_in.T @ d_Z, db = sum(d_Z), d_X_in = d_Z @ W.T
            self._grad_W[i] = X_in.T @ d_Z
            self._grad_b[i] = d_Z.sum(axis=0)
            if hook is not None and meta is not None:
                hook(
                    {
                        **meta,
                        "phase": "backward_layer",
                        "toc_id": f"bp_l{i}",
                        "layer": i,
                        "sample_in_batch": ex,
                        "dL_dz": d_Z[ex].tolist(),
                        "dL_dW": self._grad_W[i].tolist(),
                        "dL_db": self._grad_b[i].tolist(),
                        "dL_da_in": (d_Z @ W.T)[ex].tolist(),
                    },
                )
            d_out = d_Z @ W.T
        return

    def zero_grad(self) -> None:
        """Reset parameter gradients to zero (and allocate if first call)."""
        if not self._grad_W:
            self._grad_W = [np.zeros_like(W) for W, _ in self._layers]
            self._grad_b = [np.zeros_like(b) for _, b in self._layers]
        else:
            for g in self._grad_W:
                g.fill(0)
            for g in self._grad_b:
                g.fill(0)

    def step(
        self,
        learning_rate: float,
        optimizer: str = "sgd",
        *,
        decay: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        """Update parameters using accumulated gradients.

        optimizer: "sgd" (fixed lr) or "rmsprop" (per-parameter adaptive lr).
        For RMSprop: decay is the smoothing constant (rho), eps stabilizes sqrt.
        """
        hook = self._lesson_hook
        meta = self._lesson_meta
        opt_layers: list[dict[str, Any]] = []

        if optimizer == "sgd":
            for i in range(len(self._layers)):
                W, b = self._layers[i]
                gW, gb = self._grad_W[i], self._grad_b[i]
                if hook is not None and meta is not None:
                    opt_layers.append(
                        {
                            "layer": i,
                            "grad_W": gW.tolist(),
                            "grad_b": gb.tolist(),
                            "delta_W": (-learning_rate * gW).tolist(),
                            "delta_b": (-learning_rate * gb).tolist(),
                        },
                    )
                W -= learning_rate * gW
                b -= learning_rate * gb
            if hook is not None and meta is not None:
                hook(
                    {
                        **meta,
                        "phase": "optimizer",
                        "toc_id": "optimizer",
                        "optimizer_name": "sgd",
                        "learning_rate": learning_rate,
                        "optimizer_layers": opt_layers,
                    },
                )
            return
        if optimizer == "rmsprop":
            if not self._rms_W:
                self._rms_W = [np.zeros_like(W) for W, _ in self._layers]
                self._rms_b = [np.zeros_like(b) for _, b in self._layers]
            for i in range(len(self._layers)):
                gW, gb = self._grad_W[i], self._grad_b[i]
                self._rms_W[i] = decay * self._rms_W[i] + (1.0 - decay) * (gW * gW)
                self._rms_b[i] = decay * self._rms_b[i] + (1.0 - decay) * (gb * gb)
                denom_W = np.sqrt(self._rms_W[i]) + eps
                denom_b = np.sqrt(self._rms_b[i]) + eps
                eff_W = learning_rate / denom_W
                eff_b = learning_rate / denom_b
                dW = -eff_W * gW
                db = -eff_b * gb
                if hook is not None and meta is not None:
                    opt_layers.append(
                        {
                            "layer": i,
                            "grad_W": gW.tolist(),
                            "grad_b": gb.tolist(),
                            "rms_W": self._rms_W[i].tolist(),
                            "rms_b": self._rms_b[i].tolist(),
                            "effective_scale_W": eff_W.tolist(),
                            "effective_scale_b": eff_b.tolist(),
                            "delta_W": dW.tolist(),
                            "delta_b": db.tolist(),
                        },
                    )
                W, b = self._layers[i]
                W += dW
                b += db
            if hook is not None and meta is not None:
                hook(
                    {
                        **meta,
                        "phase": "optimizer",
                        "toc_id": "optimizer",
                        "optimizer_name": "rmsprop",
                        "learning_rate": learning_rate,
                        "optimizer_layers": opt_layers,
                    },
                )
            return
        raise ValueError(f"Unknown optimizer: {optimizer!r}. Use 'sgd' or 'rmsprop'.")

    def parameters(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return list(self._layers)

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

        has_val = X_val is not None and y_val is not None
        X_val_arr: np.ndarray | None = None
        y_val_arr: np.ndarray | None = None
        if has_val:
            X_val_arr = np.asarray(X_val, dtype=np.float64)
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            if len(X_val_arr) != len(y_val_arr):
                raise ValueError("X_val and y_val must have the same length.")

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
                        np.clip(LESSON_REPLAY_ANCHOR_TRAIN_INDEX, 0, max(0, n_train - 1)),
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

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Return logits (B, 2); no cache (for inference)."""
        out = X
        for i, (W, b) in enumerate(self._layers):
            Z = out @ W + b
            if i < len(self._layers) - 1:
                out = np.maximum(0, Z) if self._activation == "relu" else _sigmoid(Z)
            else:
                out = Z
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Batch probabilities (B, 2). Softmax over last axis."""
        logits = self.logits(X)
        shift = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - shift)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batch class prediction (B,), thresholding positive class at 0.5."""
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)
        return (self.predict_proba(X_arr)[:, 1] >= 0.5).astype(np.int64)
