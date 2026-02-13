from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
