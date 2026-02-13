"""Compare training runs: expand folders, load histories, rank by metric, plot, summarize."""

import json
import time
from pathlib import Path

from .plots import plot_comparison
from .serialization import load_training_history
from .training import _load_split_csv, _sanitize, train_cmd

_VAL_RECALL = "history_val_recall"
_ELAPSED = "elapsed_seconds"


def _make_grid() -> list[dict]:
    """Build full hyperparameter grid. Epochs are computed per combo from batch_size in run_best_search."""
    layers = [[24, 24], [32, 16], [48, 24], [24, 24, 12], [32, 32]]
    lrs = [0.01, 0.03, 0.05, 0.08]
    batch_sizes = [0, 16, 32, 64, 128]
    optimizers = ["sgd", "rmsprop"]
    patience_vals = [0, 5, 10]
    grid = []
    for lay in layers:
        for lr in lrs:
            for batch in batch_sizes:
                for opt in optimizers:
                    for pat in patience_vals:
                        grid.append({
                            "layers": lay,
                            "learning_rate": lr,
                            "batch_size": batch,
                            "optimizer": opt,
                            "patience": pat,
                        })
    return grid


BEST_GRID = _make_grid()


def _epochs_for_combo(base_epochs: int, batch_size: int, n_train: int) -> int:
    """Epochs so that total gradient updates ≈ base_epochs (1 update/epoch when batch_size=0)."""
    if batch_size <= 0:
        return base_epochs
    steps_per_epoch = (n_train + batch_size - 1) // batch_size
    return max(1, round(base_epochs / steps_per_epoch))


def _combo_run_name(combo: dict) -> str:
    """Short directory name for a grid combo."""
    batch = "full" if combo["batch_size"] == 0 else combo["batch_size"]
    return (
        f"layers-{_sanitize(combo['layers'])}_lr-{_sanitize(combo['learning_rate'])}_"
        f"batch-{batch}_optim-{combo['optimizer']}_patience-{combo['patience']}"
    )


def expand_run_folders(run_folders: list[str]) -> list[str]:
    """Expand parent dirs: if a path has no history.json, use its subdirs that do."""
    out: list[str] = []
    for folder in run_folders:
        p = Path(folder)
        if not p.is_dir():
            out.append(folder)
            continue
        if (p / "history.json").exists():
            out.append(folder)
            continue
        for sub in sorted(p.iterdir()):
            if sub.is_dir() and (sub / "history.json").exists():
                out.append(str(sub))
    return out


def load_histories(folders: list[str]) -> list[tuple[str, dict]]:
    """Load history.json for each folder. Raises FileNotFoundError if any missing."""
    histories: list[tuple[str, dict]] = []
    for folder in folders:
        hist = load_training_history(folder)
        histories.append((folder, hist))
    return histories


def _recall_and_time(hist: dict) -> tuple[float, float]:
    """(max val recall, elapsed_seconds). Used for ranking."""
    recall_list = hist.get(_VAL_RECALL) or []
    recall = max(recall_list) if recall_list else 0.0
    elapsed = hist.get(_ELAPSED) or float("inf")
    return recall, elapsed


def rank_runs(histories: list[tuple[str, dict]]) -> list[tuple[str, float, float]]:
    """Sort by val recall desc, then elapsed asc. Returns [(folder, recall, elapsed), ...]."""
    ranked = [
        (folder, *_recall_and_time(hist))
        for folder, hist in histories
    ]
    ranked.sort(key=lambda x: (-x[1], x[2]))
    return ranked


def print_best_summary(ranked: list[tuple[str, float, float]]) -> None:
    """Print the best run (first in ranked list)."""
    if not ranked:
        return
    folder, recall, elapsed = ranked[0]
    print(f"Best model (by val recall, then fastest): {folder}")
    print(f"  val_recall (max):   {recall:.4f}")
    print(f"  training time (s): {elapsed:.2f}")


def write_best_summary(parent_dir: str | Path, ranked: list[tuple[str, float, float]]) -> None:
    """Write best_summary.json into parent_dir."""
    path = Path(parent_dir)
    path.mkdir(parents=True, exist_ok=True)
    summary = {
        "best_run_dir": ranked[0][0],
        "best_val_recall": ranked[0][1],
        "best_elapsed_seconds": ranked[0][2],
        "all_runs": [
            {"run_dir": d, "val_recall": r, "elapsed_seconds": t}
            for d, r, t in ranked
        ],
    }
    with open(path / "best_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary saved to   {path / 'best_summary.json'}")


def compare_cmd(
    run_folders: list[str],
    save_path: str | None = None,
    plot_accuracy: bool = True,
    plot_f1: bool = False,
    plot_recall: bool = False,
    plot_precision: bool = False,
    plot_val: bool = False,
    plot_train: bool = False,
    plot_val_train: bool = False,
) -> None:
    """Load run histories, plot comparison curves, print best run summary."""
    folders = expand_run_folders(run_folders)
    if not folders:
        raise FileNotFoundError(
            "No run folders found (each must contain history.json, or be a parent of such dirs)"
        )
    histories = load_histories(folders)
    plot_comparison(
        histories=histories,
        save_path=save_path,
        plot_accuracy=plot_accuracy,
        plot_f1=plot_f1,
        plot_recall=plot_recall,
        plot_precision=plot_precision,
        plot_val=plot_val,
        plot_train=plot_train,
        plot_val_train=plot_val_train,
    )
    print_best_summary(rank_runs(histories))


def run_best_search(
    train_path: str = "datasets/train.csv",
    val_path: str = "datasets/val.csv",
    epochs: int = 70,
    seed: int = 42,
    min_delta: float = 0.0,
) -> str:
    """Run hyperparameter grid, rank by val recall then time; return best run dir.
    Epochs are scaled per combo so total gradient updates ≈ base epochs (full-batch equivalent).
    """
    n_train = len(_load_split_csv(train_path)[0])
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    parent = Path("temp") / f"best_{timestamp}"
    parent.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(BEST_GRID):
        run_dir = parent / _combo_run_name(combo)
        run_dir = str(run_dir)
        combo_epochs = _epochs_for_combo(epochs, combo["batch_size"], n_train)
        print(f"[best] Run {i + 1}/{len(BEST_GRID)}: {Path(run_dir).name} (epochs={combo_epochs})")
        train_cmd(
            train_path=train_path,
            val_path=val_path,
            layers=combo["layers"],
            epochs=combo_epochs,
            learning_rate=combo["learning_rate"],
            seed=seed,
            model_path=None,
            curves_dir=None,
            run_dir=run_dir,
            batch_size=combo["batch_size"],
            optimizer=combo["optimizer"],
            patience=combo["patience"],
            min_delta=min_delta,
        )

    subdirs = [str(s) for s in sorted(parent.iterdir()) if s.is_dir() and (s / "history.json").exists()]
    if not subdirs:
        raise RuntimeError("No completed runs with history.json found")
    histories = load_histories(subdirs)
    ranked = rank_runs(histories)
    print()
    print_best_summary(ranked)
    write_best_summary(parent, ranked)
    return ranked[0][0]
