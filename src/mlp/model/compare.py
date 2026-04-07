"""Compare training runs: expand folders, load histories, rank by metric, plot, summarize."""

import json
import time
from pathlib import Path

from mlp.utils.constants import DEFAULT_RUN_DIR

from .plots import plot_comparison
from .serialization import load_training_history
from .training import (
    train_cmd,
)
from .evaluation import evaluate_model_on_dataset, evaluate_model_on_datasets
from .schemas import TrainingHistory, TrainingMetrics, TrainingRunConfig

TOP_N = 5  # Number of best runs to display per metric

_VAL_RECALL = "history_val_recall"
_VAL_PRECISION = "history_val_precision"
_VAL_F1 = "history_val_f1"
_VAL_LOSS = "history_val_loss"
_TEST_RECALL = "history_test_recall"
_TEST_PRECISION = "history_test_precision"
_TEST_F1 = "history_test_f1"
_TEST_LOSS = "history_test_loss"

RunHistory = tuple[Path, TrainingHistory]


def _history_series(hist: TrainingHistory, metric_key: str) -> list[float]:
    """Return a history series by alias key from the validated model."""
    values = hist.model_dump(by_alias=True).get(metric_key, [])
    return values if isinstance(values, list) else []


def _make_grid() -> list[dict]:
    """Build full hyperparameter grid. Epochs are computed per combo from batch_size in run_best_search."""
    layers = [[24, 24], [32, 16], [20, 10], [24, 24, 12], [16, 8]]
    lrs = [0.01, 0.03, 0.05, 0.08]
    batch_sizes = [0, 16, 32, 64, 128]
    optimizers = ["sgd", "rmsprop"]
    patience_vals = [0, 10, 20]
    grid = []
    for lay in layers:
        for lr in lrs:
            for batch in batch_sizes:
                for opt in optimizers:
                    for pat in patience_vals:
                        grid.append(
                            {
                                "layers": lay,
                                "learning_rate": lr,
                                "batch_size": batch,
                                "optimizer": opt,
                                "patience": pat,
                            }
                        )
    return grid


BEST_GRID = _make_grid()


def expand_run_folders(run_folders: list[Path]) -> list[Path]:
    """Expand parent dirs: if a path has no history.json, use its subdirs that do."""
    out: list[Path] = []
    for folder in run_folders:
        p = folder
        if not p.is_dir():
            out.append(folder)
            continue
        if (p / "history.json").exists():
            out.append(folder)
            continue
        for sub in sorted(p.iterdir()):
            if sub.is_dir() and (sub / "history.json").exists():
                out.append(sub)
    return out


def load_histories(folders: list[Path]) -> list[RunHistory]:
    """Load and validate history.json for each folder. Raises FileNotFoundError if any missing."""
    histories: list[RunHistory] = []
    for folder in folders:
        history = load_training_history(folder)
        histories.append((folder, history))
    return histories


def _metric_and_time(hist: TrainingHistory, metric_key: str) -> tuple[float, float]:
    """(max val/test metric, elapsed_seconds). Used for ranking (higher is better)."""
    values = _history_series(hist, metric_key)
    best = max(values) if values else 0.0
    return best, hist.elapsed_seconds


def _loss_and_time(
    hist: TrainingHistory, loss_key: str = _VAL_LOSS
) -> tuple[float, float]:
    """(min loss, elapsed_seconds). Used for ranking BCE (lower is better)."""
    values = _history_series(hist, loss_key)
    best = min(values) if values else float("inf")
    return best, hist.elapsed_seconds


def _use_test_metrics(histories: list[RunHistory]) -> bool:
    """True if we have test metrics (from --best TEST_CSV) to rank by."""
    return bool(histories and histories[0][1].test_recall)


def rank_runs_by_metric(
    histories: list[RunHistory], metric_key: str
) -> list[tuple[Path, float, float]]:
    """Sort by given metric desc, then elapsed asc. Returns [(folder, value, elapsed), ...]."""
    ranked = [
        (folder, *_metric_and_time(hist, metric_key)) for folder, hist in histories
    ]
    ranked.sort(key=lambda x: (-x[1], x[2]))
    return ranked


def rank_runs(histories: list[RunHistory]) -> list[tuple[Path, float, float]]:
    """Sort by recall (test if available else val) desc, then elapsed asc."""
    key = _TEST_RECALL if _use_test_metrics(histories) else _VAL_RECALL
    return rank_runs_by_metric(histories, key)


def rank_runs_by_val_loss(
    histories: list[RunHistory],
) -> list[tuple[Path, float, float]]:
    """Sort by loss (test if available else val) asc, then elapsed asc."""
    loss_key = _TEST_LOSS if _use_test_metrics(histories) else _VAL_LOSS
    ranked = [(folder, *_loss_and_time(hist, loss_key)) for folder, hist in histories]
    ranked.sort(key=lambda x: (x[1], x[2]))
    return ranked


def _best_per_metric(
    histories: list[RunHistory],
) -> dict[str, tuple[Path, float, float]]:
    """Return best (folder, value, elapsed) for each of recall, precision, F1, loss (BCE)."""
    use_test = _use_test_metrics(histories)
    rec_key = _TEST_RECALL if use_test else _VAL_RECALL
    prec_key = _TEST_PRECISION if use_test else _VAL_PRECISION
    f1_key = _TEST_F1 if use_test else _VAL_F1
    return {
        "recall": rank_runs_by_metric(histories, rec_key)[:1][0],
        "precision": rank_runs_by_metric(histories, prec_key)[:1][0],
        "f1": rank_runs_by_metric(histories, f1_key)[:1][0],
        "val_loss": rank_runs_by_val_loss(histories)[:1][0],
    }


# Display label and "max" vs "min" for each metric in print_best_summary
_BEST_LABELS_VAL: dict[str, tuple[str, str]] = {
    "recall": ("val recall", "max"),
    "precision": ("val precision", "max"),
    "f1": ("val F1", "max"),
    "val_loss": ("val loss (BCE)", "min"),
}
_BEST_LABELS_TEST: dict[str, tuple[str, str]] = {
    "recall": ("test recall", "max"),
    "precision": ("test precision", "max"),
    "f1": ("test F1", "max"),
    "val_loss": ("test loss (BCE)", "min"),
}


def _format_loss(value: float) -> str:
    """Format BCE for display; avoid showing 0.0000 when value is tiny but non-zero."""
    if value <= 0.0 or value >= 0.0001:
        return f"{value:.4f}"
    return f"{value:.6g}"


def print_best_summary(histories: list[RunHistory]) -> None:
    """Print the top TOP_N runs for each metric (recall, precision, F1, loss BCE)."""
    if not histories:
        return
    labels = _BEST_LABELS_TEST if _use_test_metrics(histories) else _BEST_LABELS_VAL
    use_test = _use_test_metrics(histories)
    rec_key = _TEST_RECALL if use_test else _VAL_RECALL
    prec_key = _TEST_PRECISION if use_test else _VAL_PRECISION
    f1_key = _TEST_F1 if use_test else _VAL_F1
    for name, key in [
        ("recall", rec_key),
        ("precision", prec_key),
        ("f1", f1_key),
    ]:
        label, extrema = labels[name]
        ranked = rank_runs_by_metric(histories, key)
        top_n = ranked[:TOP_N]
        print(f"Top {len(top_n)} by {label} ({extrema}, then fastest):")
        for i, (folder, value, elapsed) in enumerate(top_n, 1):
            print(f"  {i}. {value:.4f}  ({elapsed:.2f}s)  {folder}")
        print()
    label, extrema = labels["val_loss"]
    ranked = rank_runs_by_val_loss(histories)
    top_n = ranked[:TOP_N]
    print(f"Top {len(top_n)} by {label} ({extrema}, then fastest):")
    for i, (folder, value, elapsed) in enumerate(top_n, 1):
        print(f"  {i}. {_format_loss(value)}  ({elapsed:.2f}s)  {folder}")
    print()


def _all_runs_flat(histories: list[RunHistory]) -> list[dict[str, str | float]]:
    """Build list of run_dir + val (and test if present) metrics + elapsed for each run."""
    out: list[dict[str, str | float]] = []
    for folder, hist in histories:
        r, elapsed = _metric_and_time(hist, _VAL_RECALL)
        p, _ = _metric_and_time(hist, _VAL_PRECISION)
        f, _ = _metric_and_time(hist, _VAL_F1)
        loss, _ = _loss_and_time(hist, _VAL_LOSS)
        row = {
            "run_dir": str(folder),
            "val_recall": r,
            "val_precision": p,
            "val_f1": f,
            "val_loss": loss,
            "elapsed_seconds": elapsed,
        }
        if hist.test_recall:
            tr, _ = _metric_and_time(hist, _TEST_RECALL)
            tp, _ = _metric_and_time(hist, _TEST_PRECISION)
            tf, _ = _metric_and_time(hist, _TEST_F1)
            tloss, _ = _loss_and_time(hist, _TEST_LOSS)
            row["test_recall"] = tr
            row["test_precision"] = tp
            row["test_f1"] = tf
            row["test_loss"] = tloss
        out.append(row)
    return out


def _top_n_flat(
    ranked: list[tuple[Path, float, float]], value_key: str
) -> list[dict[str, str | float]]:
    """Build list of {run_dir, value_key, elapsed_seconds} for top TOP_N."""
    out = []
    for folder, value, elapsed in ranked[:TOP_N]:
        out.append(
            {"run_dir": str(folder), value_key: value, "elapsed_seconds": elapsed}
        )
    return out


def write_best_summary(parent_dir: Path, histories: list[RunHistory]) -> None:
    """Write best_summary.json with best run, top TOP_N per metric, and all_runs."""
    path = parent_dir
    path.mkdir(parents=True, exist_ok=True)
    best = _best_per_metric(histories)
    use_test = _use_test_metrics(histories)
    rec_key = "test_recall" if use_test else "val_recall"
    prec_key = "test_precision" if use_test else "val_precision"
    f1_key = "test_f1" if use_test else "val_f1"
    loss_key = "test_loss" if use_test else "val_loss"
    rec_key_rank = _TEST_RECALL if use_test else _VAL_RECALL
    prec_key_rank = _TEST_PRECISION if use_test else _VAL_PRECISION
    f1_key_rank = _TEST_F1 if use_test else _VAL_F1
    summary = {
        "best_by_recall": {
            "run_dir": str(best["recall"][0]),
            rec_key: best["recall"][1],
            "elapsed_seconds": best["recall"][2],
        },
        "top_5_by_recall": _top_n_flat(
            rank_runs_by_metric(histories, rec_key_rank), rec_key
        ),
        "best_by_precision": {
            "run_dir": str(best["precision"][0]),
            prec_key: best["precision"][1],
            "elapsed_seconds": best["precision"][2],
        },
        "top_5_by_precision": _top_n_flat(
            rank_runs_by_metric(histories, prec_key_rank), prec_key
        ),
        "best_by_f1": {
            "run_dir": str(best["f1"][0]),
            f1_key: best["f1"][1],
            "elapsed_seconds": best["f1"][2],
        },
        "top_5_by_f1": _top_n_flat(rank_runs_by_metric(histories, f1_key_rank), f1_key),
        "best_by_val_loss": {
            "run_dir": str(best["val_loss"][0]),
            loss_key: best["val_loss"][1],
            "elapsed_seconds": best["val_loss"][2],
        },
        "top_5_by_val_loss": _top_n_flat(rank_runs_by_val_loss(histories), loss_key),
        "all_runs": _all_runs_flat(histories),
    }
    with open(path / "best_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  summary saved to   {path / 'best_summary.json'}")


def _histories_to_plot_by_bce(
    histories: list[RunHistory],
) -> list[tuple[Path, TrainingHistory]]:
    """Select 5 best BCE, 1 middle, 1 worst for comparison plots (no duplicates)."""
    ranked = rank_runs_by_val_loss(histories)
    n = len(ranked)
    if n == 0:
        return []
    best_5_folders = [r[0] for r in ranked[:5]]
    mid_folder = ranked[n // 2][0]
    worst_folder = ranked[-1][0]
    folders_ordered = list(
        dict.fromkeys(best_5_folders + [mid_folder] + [worst_folder])
    )
    folder_to_hist: dict[Path, TrainingHistory] = {
        folder: history for folder, history in histories
    }
    return [(f, folder_to_hist[f]) for f in folders_ordered if f in folder_to_hist]


def _add_test_metrics_to_histories(
    histories: list[RunHistory],
    test_paths: list[Path],
) -> None:
    """Evaluate each run on test_paths (equal-weight average if multiple) and merge into histories (in-place)."""
    for folder, history in histories:
        if len(test_paths) == 1:
            metrics: TrainingMetrics = evaluate_model_on_dataset(folder, test_paths[0])
        else:
            metrics: TrainingMetrics = evaluate_model_on_datasets(folder, test_paths)
        history.test_accuracy.append(metrics.accuracy)
        history.test_loss.append(metrics.loss)
        history.test_precision.append(metrics.precision)
        history.test_recall.append(metrics.recall)
        history.test_f1.append(metrics.f1)


def compare_cmd(
    run_folders: list[Path],
    save_path: Path | None = None,
    plot_accuracy: bool = True,
    plot_f1: bool = False,
    plot_recall: bool = False,
    plot_precision: bool = False,
    plot_val: bool = False,
    plot_train: bool = False,
    plot_val_train: bool = False,
    test_paths: list[Path] | None = None,
) -> None:
    """Load run histories, optionally evaluate on test set(s), plot comparison curves, print best run summary.
    When test_paths is provided, each run's model is evaluated on those datasets; metrics are averaged (equal weight)
    and ranking uses test metrics. Displays top TOP_N runs per metric.
    """
    folders = expand_run_folders(run_folders)
    if not folders:
        raise FileNotFoundError(
            "No run folders found (each must contain history.json, or be a parent of such dirs)"
        )
    histories = load_histories(folders)
    if test_paths:
        n = len(test_paths)
        print(
            f"Evaluating {len(histories)} run(s) on {n} test set(s) (equal-weight average): {', '.join(str(p) for p in test_paths)}"
        )
        _add_test_metrics_to_histories(histories, test_paths)
        print("Ranking by test metrics.")
    histories_to_plot = _histories_to_plot_by_bce(histories)
    plot_comparison(
        histories=histories_to_plot,
        save_path=save_path,
        plot_accuracy=plot_accuracy,
        plot_f1=plot_f1,
        plot_recall=plot_recall,
        plot_precision=plot_precision,
        plot_val=plot_val,
        plot_train=plot_train,
        plot_val_train=plot_val_train,
    )
    print_best_summary(histories)


def run_best_search(
    run_config: TrainingRunConfig,
) -> Path:
    """Run hyperparameter grid, rank by recall (test if test_paths else val) then time; return best run dir.
    When test_paths is provided, each trained model is evaluated on those datasets; metrics are averaged
    (equal weight) and best models are chosen by test metrics. Displays top TOP_N runs per metric.
    """
    from ..data.data_engineering import fix_dataset
    from ..utils.loader import load_dataset

    df = load_dataset(run_config.train_path)
    if "label" not in df.columns:
        df = fix_dataset(df)
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    parent_dir = Path(DEFAULT_RUN_DIR) / Path(
        f"best_{run_config.train_path.stem}_{timestamp}"
    )
    parent_dir.mkdir(parents=True, exist_ok=True)

    for i, combo in enumerate(BEST_GRID):
        print(
            f"[best] Run {i + 1}/{len(BEST_GRID)}: {Path(parent_dir).name} (epochs={run_config.epochs})"
        )
        train_cmd(
            run_config=TrainingRunConfig(
                train_path=run_config.train_path,
                val_ratio=run_config.val_ratio,
                layers=combo["layers"],
                epochs=run_config.epochs,
                learning_rate=combo["learning_rate"],
                seed=run_config.seed,
                batch_size=combo["batch_size"],
                optimizer=combo["optimizer"],
                patience=combo["patience"],
                parent_dir=parent_dir,
            ),
        )

    subdirs = [
        s
        for s in sorted(Path(parent_dir).iterdir())
        if s.is_dir() and (s / "history.json").exists()
    ]
    if not subdirs:
        raise RuntimeError("No completed runs with history.json found")
    histories = load_histories(subdirs)

    if run_config.test_paths:
        n = len(run_config.test_paths)
        print(
            f"[best] Evaluating all runs on {n} test set(s) (equal-weight average): {', '.join(str(p) for p in run_config.test_paths)}"
        )
        _add_test_metrics_to_histories(histories, run_config.test_paths)
        print("[best] Ranking by test metrics (less bias than validation).")
    print()
    print_best_summary(histories)
    write_best_summary(parent_dir, histories)
    # Return best by recall for backward compatibility
    return rank_runs(histories)[0][0]
