from pathlib import Path

import matplotlib.pyplot as plt

from .schemas import TrainingHistory


def _history_series(hist: TrainingHistory, metric_key: str) -> list[float]:
    """Return a history series by alias key from the validated model."""
    values = hist.model_dump(by_alias=True).get(metric_key, [])
    return values if isinstance(values, list) else []


def _run_label(folder: Path) -> str:
    """Label for a run curve: basename of folder."""
    return folder.name


def save_learning_curves(
    history: TrainingHistory,
    output_dir: Path = Path("figures/training"),
) -> None:
    output_path = output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history.train_loss) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_loss, label="train_loss")
    plt.plot(epochs, history.val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_accuracy, label="train_acc")
    plt.plot(epochs, history.val_accuracy, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.train_precision, label="train_precision")
    plt.plot(epochs, history.val_precision, label="val_precision")
    plt.plot(epochs, history.train_recall, label="train_recall")
    plt.plot(epochs, history.val_recall, label="val_recall")
    plt.plot(epochs, history.train_f1, label="train_f1")
    plt.plot(epochs, history.val_f1, label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path / "precision_recall_f1_curve.png", dpi=150, bbox_inches="tight"
    )
    plt.close()


def plot_comparison(
    histories: list[tuple[Path, TrainingHistory]],
    save_path: Path | None = None,
    plot_accuracy: bool = True,
    plot_f1: bool = False,
    plot_recall: bool = False,
    plot_precision: bool = False,
    plot_val: bool = False,
    plot_train: bool = False,
    plot_val_train: bool = False,
) -> None:
    """Plot learning curves for multiple runs (one figure per metric)."""
    show_val = plot_val or plot_val_train
    show_train = plot_train or plot_val_train
    if not show_val and not show_train:
        show_val = show_train = True

    metrics: list[tuple[str, str, str, str]] = []  # (title, ylabel, train_key, val_key)
    if plot_accuracy:
        metrics.append(
            ("Accuracy comparison", "Accuracy", "history_train_acc", "history_val_acc")
        )
    if plot_f1:
        metrics.append(("F1 comparison", "F1", "history_train_f1", "history_val_f1"))
    if plot_recall:
        metrics.append(
            (
                "Recall comparison",
                "Recall",
                "history_train_recall",
                "history_val_recall",
            )
        )
    if plot_precision:
        metrics.append(
            (
                "Precision comparison",
                "Precision",
                "history_train_precision",
                "history_val_precision",
            )
        )

    if not metrics:
        return

    save_dir: Path | None = None
    save_first_file: Path | None = None
    if save_path:
        save_dir = save_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_first_file = save_path

    for idx, (title, ylabel, train_key, val_key) in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        for run_dir, hist in histories:
            label_base = _run_label(run_dir)
            train_values = _history_series(hist, train_key)
            val_values = _history_series(hist, val_key)
            if show_train and train_values:
                train_epochs = list(range(1, len(train_values) + 1))
                plt.plot(
                    train_epochs,
                    train_values,
                    label=f"{label_base} (train)",
                    linestyle="--",
                )
            if show_val and val_values:
                val_epochs = list(range(1, len(val_values) + 1))
                plt.plot(val_epochs, val_values, label=f"{label_base} (val)")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        if save_dir is not None:
            if save_first_file is not None and idx == 0:
                plt.savefig(save_first_file, dpi=150, bbox_inches="tight")
            else:
                safe_name = title.lower().replace(" ", "_").replace("-", "_") + ".png"
                plt.savefig(save_dir / safe_name, dpi=150, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
            plt.close()
