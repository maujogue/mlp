"""Vectorized NumPy MLP for binary classification (no micrograd)."""

import time

import numpy as np

from .schemas import TrainingHistory, TrainingMetrics
from ..utils.constants import SEED


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


def margin_loss_grad(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of margin loss w.r.t. logits (B, 2)."""
    batch_size = logits.shape[0]
    y_arr = np.asarray(y, dtype=np.intp)
    correct = logits[np.arange(batch_size), y_arr]
    wrong = logits[np.arange(batch_size), 1 - y_arr]
    margins = np.maximum(0.0, 1.0 + wrong - correct)
    mask = (margins > 0).astype(np.float64)
    d_logits = np.zeros_like(logits)
    d_logits[np.arange(batch_size), y_arr] = -mask / batch_size
    d_logits[np.arange(batch_size), 1 - y_arr] = mask / batch_size
    return d_logits


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
        if len(hidden_layers) < 2:
            raise ValueError("At least two hidden layers are required.")
        if output_size != 2:
            raise ValueError("Only binary classification with 2 outputs is supported.")

        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.seed = seed
        self._activation = activation.lower()
        if self._activation not in ("relu", "sigmoid"):
            raise ValueError("activation must be 'relu' or 'sigmoid'")

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

    def _init_weights_and_biases(self) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [self.n_features, *self.hidden_layers, self.output_size]
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            scale = np.sqrt(2.0 / in_d) if i < len(dims) - 2 else 0.1
            W = rng.standard_normal((in_d, out_d)).astype(np.float64) * scale
            b = np.zeros(out_d, dtype=np.float64)
            self._layers.append((W, b))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Batch forward: X (B, n_features) -> logits (B, output_size)."""
        self._cache = []
        out = X
        for i, (W, b) in enumerate(self._layers):
            Z = out @ W + b  # (B, out_d)
            if i < len(self._layers) - 1:
                self._cache.append((out, Z))  # pre-activation input, pre-activation Z
                out = np.maximum(0, Z) if self._activation == "relu" else _sigmoid(Z)
            else:
                self._cache.append((out, np.zeros_like(Z)))
                out = Z
        return out

    def backward(self, d_logits: np.ndarray) -> None:
        """Backprop: d_logits (B, output_size). Accumulates gradients in-place."""
        d_out = d_logits
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
        if optimizer == "sgd":
            for i in range(len(self._layers)):
                W, b = self._layers[i]
                W -= learning_rate * self._grad_W[i]
                b -= learning_rate * self._grad_b[i]
            return
        if optimizer == "rmsprop":
            if not self._rms_W:
                self._rms_W = [np.zeros_like(W) for W, _ in self._layers]
                self._rms_b = [np.zeros_like(b) for _, b in self._layers]
            for i in range(len(self._layers)):
                gW, gb = self._grad_W[i], self._grad_b[i]
                self._rms_W[i] = decay * self._rms_W[i] + (1.0 - decay) * (gW * gW)
                self._rms_b[i] = decay * self._rms_b[i] + (1.0 - decay) * (gb * gb)
                W, b = self._layers[i]
                W -= learning_rate * gW / (np.sqrt(self._rms_W[i]) + eps)
                b -= learning_rate * gb / (np.sqrt(self._rms_b[i]) + eps)
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
        epochs: int = 70,
        learning_rate: float,
        batch_size: int,
        optimizer: str,
        patience: int,
        seed: int | None = None,
        verbose: bool = False,
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
        effective_batch_size = batch_size if batch_size > 0 else n_train
        rng_seed = self.seed if seed is None else seed
        rng = np.random.default_rng(rng_seed)

        best_loss = np.inf
        epochs_no_improve = 0
        best_weights: list[tuple[np.ndarray, np.ndarray]] | None = None

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(n_train)
            for start in range(0, n_train, effective_batch_size):
                batch_idx = indices[start : start + effective_batch_size]
                X_batch = X_train_arr[batch_idx]
                y_batch = y_train_arr[batch_idx]
                self.zero_grad()
                logits = self.forward(X_batch)
                d_logits = margin_loss_grad(logits, y_batch)
                self.backward(d_logits)
                self.step(learning_rate=learning_rate, optimizer=optimizer)

            train_metrics: TrainingMetrics = evaluate(self, X_train_arr, y_train_arr)
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

            if verbose:
                if has_val and X_val_arr is not None and y_val_arr is not None:
                    print(
                        f"epoch {epoch:02d}/{epochs} - "
                        f"loss: {train_metrics.loss:.4f} - acc: {train_metrics.accuracy:.4f} - "
                        f"prec: {train_metrics.precision:.4f} - rec: {train_metrics.recall:.4f} - "
                        f"f1: {train_metrics.f1:.4f} - val_loss: {history.val_loss[-1]:.4f} - "
                        f"val_acc: {history.val_accuracy[-1]:.4f} - val_prec: {history.val_precision[-1]:.4f} - "
                        f"val_rec: {history.val_recall[-1]:.4f} - val_f1: {history.val_f1[-1]:.4f}"
                    )
                else:
                    print(
                        f"epoch {epoch:02d}/{epochs} - "
                        f"loss: {train_metrics.loss:.4f} - acc: {train_metrics.accuracy:.4f} - "
                        f"prec: {train_metrics.precision:.4f} - rec: {train_metrics.recall:.4f} - "
                        f"f1: {train_metrics.f1:.4f}"
                    )

            if patience > 0:
                if monitor_loss < best_loss:
                    best_loss = monitor_loss
                    epochs_no_improve = 0
                    best_weights = [(W.copy(), b.copy()) for W, b in self.parameters()]
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)."
                            )
                        break

        if best_weights is not None:
            for i in range(len(self._layers)):
                np.copyto(self._layers[i][0], best_weights[i][0])
                np.copyto(self._layers[i][1], best_weights[i][1])

        self.last_fit_seconds = time.perf_counter() - fit_start_time
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
