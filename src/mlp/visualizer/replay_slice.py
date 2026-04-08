"""Reconstruct weights at a replay step and sample a 2D loss slice (two scalar parameters)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from mlp.model.mlp_classifier import MLPClassifier, softmax_cross_entropy_loss
from mlp.visualizer.replay_schemas import ReplayManifest


def load_manifest(replay_dir: Path) -> ReplayManifest:
    p = replay_dir / "replay_manifest.json"
    with open(p, encoding="utf-8") as f:
        return ReplayManifest.model_validate_json(f.read())


def load_steps(
    replay_dir: Path, *, manifest: ReplayManifest | None = None
) -> list[dict[str, Any]]:
    m = manifest or load_manifest(replay_dir)
    path = replay_dir / m.steps_file
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def snapshot_weights_biases(
    steps: list[dict[str, Any]],
    step_index: int,
    *,
    n_layers: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Take latest W,b from forward_layer events up to step_index (inclusive)."""
    last: dict[int, tuple[list[list[float]], list[float]]] = {}
    for i, st in enumerate(steps):
        if i > step_index:
            break
        if st.get("phase") == "forward_layer" and st.get("W") is not None:
            L = int(st["layer"])
            last[L] = (st["W"], st["b"])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for L in range(n_layers):
        if L not in last:
            raise ValueError(
                f"No forward_layer snapshot for layer {L} at step {step_index}"
            )
        W, b = last[L]
        out.append((np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64)))
    return out


def flatten_params(layers: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for W, b in layers:
        parts.append(W.reshape(-1))
        parts.append(b.reshape(-1))
    return np.concatenate(parts)


def unflatten_like(
    flat: np.ndarray, template: list[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    off = 0
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for Wt, bt in template:
        nw = Wt.size
        nb = bt.size
        W = flat[off : off + nw].reshape(Wt.shape).copy()
        off += nw
        b = flat[off : off + nb].reshape(bt.shape).copy()
        off += nb
        out.append((W, b))
    return out


def loss_slice_grid(
    manifest: ReplayManifest,
    steps: list[dict[str, Any]],
    *,
    step_index: int,
    param_i: int,
    param_j: int,
    grid_half_extent: float = 0.35,
    grid_n: int = 17,
) -> dict[str, Any]:
    """Vary two entries of the flattened weight vector; hold others fixed at snapshot."""
    if manifest.toy_points is None or len(manifest.toy_points) == 0:
        raise ValueError("manifest.toy_points required for loss slice")
    X = np.array([[p.x0, p.x1] for p in manifest.toy_points], dtype=np.float64)
    y = np.array([p.y for p in manifest.toy_points], dtype=np.int64)

    n_layers = len(manifest.layer_sizes) + 1
    base = snapshot_weights_biases(steps, step_index, n_layers=n_layers)
    flat0 = flatten_params(base)
    n_params = flat0.size
    if param_i < 0 or param_i >= n_params or param_j < 0 or param_j >= n_params:
        raise ValueError(f"param indices out of range (0..{n_params - 1})")
    if param_i == param_j:
        raise ValueError("param_i and param_j must differ")

    model = MLPClassifier(
        n_features=manifest.input_dim,
        hidden_layers=tuple(manifest.layer_sizes),
        output_size=manifest.n_classes,
        seed=0,
        activation=manifest.activation,
    )

    def loss_at(flat: np.ndarray) -> float:
        layers = unflatten_like(flat, base)
        for idx, (W, b) in enumerate(layers):
            np.copyto(model._layers[idx][0], W)
            np.copyto(model._layers[idx][1], b)
        logits = model.forward(X)
        return float(softmax_cross_entropy_loss(logits, y))

    v0, vj = flat0[param_i], flat0[param_j]
    xs = np.linspace(v0 - grid_half_extent, v0 + grid_half_extent, grid_n)
    ys = np.linspace(vj - grid_half_extent, vj + grid_half_extent, grid_n)
    z = np.zeros((grid_n, grid_n), dtype=np.float64)
    for ii, a in enumerate(xs):
        for jj, b in enumerate(ys):
            f = flat0.copy()
            f[param_i] = a
            f[param_j] = b
            z[jj, ii] = loss_at(f)

    return {
        "param_i": param_i,
        "param_j": param_j,
        "center_i": float(v0),
        "center_j": float(vj),
        "x_axis": xs.tolist(),
        "y_axis": ys.tolist(),
        "z": z.tolist(),
        "n_params": n_params,
    }
