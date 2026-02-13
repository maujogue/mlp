from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_learning_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accuracies: list[float],
    val_accuracies: list[float],
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
