import os

import numpy as np

from .model import MLPClassifier


def _save_txt(model: MLPClassifier, dirpath: str) -> None:
    """Save model as human-readable .txt files in a directory."""
    os.makedirs(dirpath, exist_ok=True)
    state = model.export_state()
    with open(os.path.join(dirpath, "metadata.txt"), "w") as f:
        f.write(f"n_features={state['n_features']}\n")
        f.write(f"output_size={state['output_size']}\n")
        f.write(f"seed={state['seed']}\n")
        f.write(f"hidden_layers={','.join(map(str, state['hidden_layers']))}\n")
        f.write(f"num_layers={len(state['layers'])}\n")
    for i, layer in enumerate(state["layers"]):
        np.savetxt(os.path.join(dirpath, f"layer_{i}_W.txt"), layer["W"])
        np.savetxt(os.path.join(dirpath, f"layer_{i}_b.txt"), layer["b"])


def _load_txt(dirpath: str) -> dict:
    """Load model state from directory of .txt files."""
    metadata_path = os.path.join(dirpath, "metadata.txt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    state = {}
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key == "hidden_layers":
                state[key] = [int(x) for x in value.split(",")]
            elif key in ("n_features", "output_size", "seed", "num_layers"):
                state[key] = int(value)
    num_layers = state["num_layers"]
    state["layers"] = []
    for i in range(num_layers):
        W = np.loadtxt(os.path.join(dirpath, f"layer_{i}_W.txt"), dtype=np.float64)
        b = np.loadtxt(os.path.join(dirpath, f"layer_{i}_b.txt"), dtype=np.float64)
        if b.ndim == 0:
            b = np.atleast_1d(b)
        state["layers"].append({"W": W, "b": b})
    return state


def save_model(
    model: MLPClassifier,
    filepath: str = "weights/model",
) -> None:
    parent = os.path.dirname(filepath)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    if filepath.endswith(".npz"):
        state = model.export_state()
        kwargs = {
            "n_features": np.array(state["n_features"], dtype=np.int64),
            "output_size": np.array(state["output_size"], dtype=np.int64),
            "seed": np.array(state["seed"], dtype=np.int64),
            "hidden_layers": np.array(state["hidden_layers"], dtype=np.int64),
            "num_layers": np.array(len(state["layers"]), dtype=np.int64),
        }
        for i, layer in enumerate(state["layers"]):
            kwargs[f"layer_{i}_W"] = layer["W"]
            kwargs[f"layer_{i}_b"] = layer["b"]
        np.savez_compressed(filepath, **kwargs)
    else:
        _save_txt(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(
    filepath: str = "weights/model",
) -> tuple[MLPClassifier, None]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    if filepath.endswith(".npz"):
        data = np.load(filepath, allow_pickle=False)
        num_layers = int(data["num_layers"])
        state = {
            "n_features": int(data["n_features"]),
            "output_size": int(data["output_size"]),
            "seed": int(data["seed"]),
            "hidden_layers": list(data["hidden_layers"]),
            "layers": [
                {"W": data[f"layer_{i}_W"], "b": data[f"layer_{i}_b"]}
                for i in range(num_layers)
            ],
        }
    else:
        state = _load_txt(filepath)
    return MLPClassifier.from_state(state), None
