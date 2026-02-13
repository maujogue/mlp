"""Vectorized NumPy MLP for binary classification (no micrograd)."""

from typing import Any

import numpy as np


class MLPClassifier:
    """NumPy MLP: batch forward/backward, ReLU hidden, 2-class output."""

    def __init__(
        self,
        n_features: int,
        hidden_layers: list[int] | tuple[int, ...] | None = None,
        output_size: int = 2,
        seed: int = 42,
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

        rng = np.random.default_rng(seed)
        dims = [n_features, *hidden_layers, output_size]
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(dims) - 1):
            in_d, out_d = dims[i], dims[i + 1]
            # Kaiming-style init for ReLU (last layer is linear, use smaller scale)
            scale = np.sqrt(2.0 / in_d) if i < len(dims) - 2 else 0.1
            W = rng.standard_normal((in_d, out_d)).astype(np.float64) * scale
            b = np.zeros(out_d, dtype=np.float64)
            self._layers.append((W, b))

        # Forward cache for backward; gradients (set by zero_grad)
        self._cache: list[tuple[np.ndarray, ...]] = []
        self._grad_W: list[np.ndarray] = []
        self._grad_b: list[np.ndarray] = []
        # RMSprop state: running average of squared gradients (lazy init)
        self._rms_W: list[np.ndarray] = []
        self._rms_b: list[np.ndarray] = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Batch forward: X (B, n_features) -> logits (B, output_size)."""
        self._cache = []
        out = X
        for i, (W, b) in enumerate(self._layers):
            Z = out @ W + b  # (B, out_d)
            if i < len(self._layers) - 1:
                self._cache.append((out, Z))  # pre-activation input, pre-activation Z
                out = np.maximum(0, Z)  # ReLU
            else:
                self._cache.append((out, None))
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
                d_Z = d_out * (Z > 0).astype(np.float64)  # ReLU derivative
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

    def logits(self, X: np.ndarray) -> np.ndarray:
        """Return logits (B, 2); no cache (for inference)."""
        out = X
        for i, (W, b) in enumerate(self._layers):
            Z = out @ W + b
            if i < len(self._layers) - 1:
                out = np.maximum(0, Z)
            else:
                out = Z
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Batch probabilities (B, 2). Softmax over last axis."""
        logits = self.logits(X)
        shift = logits.max(axis=1, keepdims=True)
        exp = np.exp(logits - shift)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba_one(self, features: list[float] | np.ndarray) -> list[float]:
        """Single-sample probabilities [p0, p1] for API compatibility."""
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        p = self.predict_proba(x)[0]
        return [float(p[0]), float(p[1])]

    def export_state(self) -> dict[str, Any]:
        return {
            "n_features": self.n_features,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "seed": self.seed,
            "layers": [{"W": W.copy(), "b": b.copy()} for W, b in self._layers],
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> "MLPClassifier":
        obj = cls(
            n_features=int(state["n_features"]),
            hidden_layers=list(state["hidden_layers"]),
            output_size=int(state["output_size"]),
            seed=int(state.get("seed", 42)),
        )
        for i, layer_state in enumerate(state["layers"]):
            obj._layers[i] = (
                np.asarray(layer_state["W"], dtype=np.float64),
                np.asarray(layer_state["b"], dtype=np.float64),
            )
        return obj
