import json
import pickle
from pathlib import Path

from .mlp_classifier import MLPClassifier
from .schemas import TrainingHistory, TrainingRunConfig


def save_model(
    model: MLPClassifier,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to {path.absolute()}")


def load_model(
    path: Path,
) -> MLPClassifier:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Directory (e.g. run folder): look for model.pkl inside
    if path.is_dir():
        pkl_path = path / "model.pkl"
        if pkl_path.exists():
            path = pkl_path
        else:
            raise FileNotFoundError(f"No model.pkl found in {path}")
    else:
        if path.suffix != ".pkl":
            raise FileNotFoundError(
                f"Expected path to a .pkl model file or a directory containing model.pkl: {path}"
            )

    with open(path, "rb") as f:
        model = pickle.load(f)
    if not isinstance(model, MLPClassifier):
        raise TypeError(f"Loaded object from {path} is not an MLPClassifier.")
    return model


def save_training_history(
    run_dir: Path,
    history: TrainingHistory,
    elapsed_seconds: float,
) -> None:
    """Write history.json and run_config.json into run_dir (used when using temp layout)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = {
        **history.model_dump(by_alias=True),
        "elapsed_seconds": elapsed_seconds,
        "epochs_ran": len(history.train_loss),
    }
    with open(run_dir / "history.json", "w") as f:
        json.dump(data, f, indent=2)


def save_run_config(
    run_dir: Path,
    run_config: TrainingRunConfig,
) -> None:
    """Write run_config.json with the options used for this run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    data = run_config.model_dump(mode="json")
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(data, f, indent=2)


def load_training_history(run_folder: Path) -> TrainingHistory:
    """Load training history (and optional run_config) from a run folder."""
    history_path = run_folder / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"history.json not found in {run_folder}")
    with open(history_path, "r") as f:
        history_data = json.load(f)
    return TrainingHistory.model_validate(history_data)
