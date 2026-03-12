import json
import os
from pathlib import Path

import numpy as np

from .mlp_classifier import MLPClassifier
from .schemas import TrainingHistory, TrainingRunConfig


def save_model(
    model: MLPClassifier,
    filepath: str | Path,
) -> None:
    path = Path(filepath)
    if not path.suffix or path.suffix != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
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
    np.savez_compressed(str(path), allow_pickle=True, **kwargs)
    print(f"Model saved to {path}")


def save_training_history(
    run_dir: str,
    history: TrainingHistory | dict[str, list[float]],
    elapsed_seconds: float,
) -> None:
    """Write history.json and run_config.json into run_dir (used when using temp layout)."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    validated_history = (
        history
        if isinstance(history, TrainingHistory)
        else TrainingHistory.model_validate(history)
    )
    data = {
        **validated_history.model_dump(by_alias=True),
        "elapsed_seconds": elapsed_seconds,
        "epochs_ran": len(validated_history.train_loss),
    }
    with open(path / "history.json", "w") as f:
        json.dump(data, f, indent=2)


def save_run_config(
    run_dir: str,
    run_config: TrainingRunConfig | dict,
) -> None:
    """Write run_config.json with the options used for this run."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    validated_config = (
        run_config
        if isinstance(run_config, TrainingRunConfig)
        else TrainingRunConfig.model_validate(run_config)
    )
    data = validated_config.model_dump()
    with open(path / "run_config.json", "w") as f:
        json.dump(data, f, indent=2)


def load_training_history(run_folder: str) -> dict:
    """Load training history (and optional run_config) from a run folder."""
    folder = Path(run_folder)
    history_path = folder / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found in {run_folder}")
    with open(history_path) as f:
        history_data = json.load(f)
    validated_history = TrainingHistory.model_validate(history_data)
    data = validated_history.model_dump(by_alias=True)
    data["elapsed_seconds"] = history_data.get("elapsed_seconds")
    data["epochs_ran"] = history_data.get(
        "epochs_ran", len(validated_history.train_loss)
    )
    config_path = folder / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            run_config_data = json.load(f)
        data["run_config"] = TrainingRunConfig.model_validate(
            run_config_data
        ).model_dump()
    return data


def load_model(
    filepath: str,
) -> tuple[MLPClassifier, None]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Explicit .npz file
    if filepath.endswith(".npz"):
        path = filepath
    # Directory (e.g. run folder): look for model.npz inside
    elif os.path.isdir(filepath):
        npz_path = os.path.join(filepath, "model.npz")
        if os.path.exists(npz_path):
            path = npz_path
        else:
            raise FileNotFoundError(f"No model.npz found in {filepath}")
    else:
        raise FileNotFoundError(
            f"Expected path to model.npz or a directory containing model.npz: {filepath}"
        )

    data = np.load(path, allow_pickle=False)
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
    return MLPClassifier.from_state(state), None
