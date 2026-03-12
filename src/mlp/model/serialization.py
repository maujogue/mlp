import json
import pickle
from pathlib import Path

from .mlp_classifier import MLPClassifier
from .schemas import TrainingHistory, TrainingRunConfig


def save_model(
    model: MLPClassifier,
    filepath: str | Path,
) -> None:
    path = Path(filepath)
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {path}")


def load_model(
    filepath: str | Path,
) -> MLPClassifier:
    path_input = Path(filepath)
    if not path_input.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Explicit .pkl file
    if path_input.is_file() and path_input.suffix == ".pkl":
        path = path_input
    # Directory (e.g. run folder): look for model.pkl inside
    elif path_input.is_dir():
        pkl_path = path_input / "model.pkl"
        if pkl_path.exists():
            path = pkl_path
        else:
            raise FileNotFoundError(f"No model.pkl found in {filepath}")
    else:
        raise FileNotFoundError(
            f"Expected path to model.pkl or a directory containing model.pkl: {filepath}"
        )

    with open(path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, MLPClassifier):
        raise TypeError(f"Loaded object from {path} is not an MLPClassifier.")
    return model


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
