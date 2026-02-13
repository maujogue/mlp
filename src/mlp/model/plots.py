from pathlib import Path

import matplotlib.pyplot as plt


def _run_label(folder: str) -> str:
    """Label for a run curve: basename of folder."""
    return Path(folder).name


def save_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accuracies: list[float],
    val_accuracies: list[float],
    train_precisions: list[float],
    val_precisions: list[float],
    train_recalls: list[float],
    val_recalls: list[float],
    train_f1s: list[float],
    val_f1s: list[float],
    output_dir: str = "figures/training",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross-entropy")
    plt.title("Loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="train_acc")
    plt.plot(epochs, val_accuracies, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "accuracy_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_precisions, label="train_precision")
    plt.plot(epochs, val_precisions, label="val_precision")
    plt.plot(epochs, train_recalls, label="train_recall")
    plt.plot(epochs, val_recalls, label="val_recall")
    plt.plot(epochs, train_f1s, label="train_f1")
    plt.plot(epochs, val_f1s, label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Precision, Recall, F1 curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "precision_recall_f1_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(
    histories: list[tuple[str, dict]],
    save_path: str | None = None,
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
        metrics.append(("Accuracy comparison", "Accuracy", "history_train_acc", "history_val_acc"))
    if plot_f1:
        metrics.append(("F1 comparison", "F1", "history_train_f1", "history_val_f1"))
    if plot_recall:
        metrics.append(("Recall comparison", "Recall", "history_train_recall", "history_val_recall"))
    if plot_precision:
        metrics.append(("Precision comparison", "Precision", "history_train_precision", "history_val_precision"))

    if not metrics:
        return

    save_dir: Path | None = None
    save_first_file: Path | None = None  # when user passes a file path, save first figure there
    if save_path:
        p = Path(save_path)
        if p.suffix and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".pdf"):
            save_dir = p.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            save_first_file = p
        else:
            save_dir = p
            save_dir.mkdir(parents=True, exist_ok=True)

    for idx, (title, ylabel, train_key, val_key) in enumerate(metrics):
        plt.figure(figsize=(8, 5))
        for folder, hist in histories:
            label_base = _run_label(folder)
            epochs = list(range(1, len(hist.get(train_key, [])) + 1))
            if show_train and train_key in hist:
                plt.plot(epochs, hist[train_key], label=f"{label_base} (train)", linestyle="--")
            if show_val and val_key in hist:
                plt.plot(epochs, hist[val_key], label=f"{label_base} (val)")
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
