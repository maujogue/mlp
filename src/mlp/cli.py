import argparse
from pathlib import Path

from pydantic import ValidationError

from .utils.constants import BLUE, RESET


# ---------- prepare-dataset command ----------
def main_prepare_dataset() -> None:
    parser = argparse.ArgumentParser(
        description="Fix the dataset (column titles, fill missing). Scaling is done during training."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the raw dataset file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        type=str,
        help="Output path for the prepared CSV (default: next to input with _prepared suffix)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        p = Path(args.dataset_path)
        output_path = str(p.parent / f"{p.stem}_prepared{p.suffix}")

    from .data.data_engineering import prepare_dataset_cmd

    prepare_dataset_cmd(args.dataset_path, output_path)
    print(BLUE + "Prepared " + args.dataset_path + " -> " + output_path + RESET)


# ---------- split command ----------
def main_split() -> None:
    parser = argparse.ArgumentParser(
        description="Split a prepared dataset into train and test sets"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the prepared dataset file (use mlp-prepare-dataset first)",
    )
    parser.add_argument(
        "--test-size",
        "-t",
        nargs="?",
        default=0.2,
        type=float,
        help="Proportion of dataset to use as test set (e.g., 0.2)",
    )
    args = parser.parse_args()

    from .data.data_engineering import split_cmd

    split_cmd(args.dataset_path, args.test_size)
    print(
        BLUE
        + "Split "
        + args.dataset_path
        + " into train/test (test size "
        + str(args.test_size)
        + ")"
        + RESET
    )


# ---------- train command ----------
def main_train() -> None:
    parser = argparse.ArgumentParser(description="Train the model on the dataset")
    parser.add_argument(
        "train_path",
        type=str,
        help="Path to training CSV (default: datasets/train.csv). Split into train/val internally.",
    )
    parser.add_argument(
        "--val-ratio",
        default=0.2,
        type=float,
        help="Fraction of train set to use as validation (default: 0.2)",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=[24, 24],
        type=int,
        help="Hidden layer sizes (default: 24 24)",
    )
    parser.add_argument(
        "--epochs",
        default=70,
        type=int,
        help="Number of epochs (default: 70)",
    )
    parser.add_argument(
        "--learning-rate",
        default=0.03,
        type=float,
        help="Learning rate (default: 0.03)",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--batch-size",
        default=0,
        type=int,
        help="Batch size; 0 = full dataset (default: 0)",
    )
    parser.add_argument(
        "--optimizer",
        default="sgd",
        choices=("sgd", "rmsprop"),
        type=str,
        help="Optimizer: sgd or rmsprop (default: sgd)",
    )
    parser.add_argument(
        "--patience",
        default=0,
        type=int,
        help="Early stopping: stop after N epochs without val_loss improvement; 0 = disabled (default: 0)",
    )
    parser.add_argument(
        "--best",
        nargs="*",
        default=None,
        metavar="TEST_CSV",
        help="Run hyperparameter grid and pick best run. Optional: one or more test CSV paths; if given, best models are ranked by average metrics on these sets (equal weight, less bias). Displays top 5 per metric.",
    )
    args = parser.parse_args()

    from .model.compare import run_best_search
    from .model.schemas import TrainingRunConfig
    from .model.training import train_cmd

    run_config = TrainingRunConfig(
        train_path=args.train_path,
        val_ratio=args.val_ratio,
        layers=args.layers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        patience=args.patience,
        test_paths=list(args.best) if args.best else [],
    )
    if args.best is not None:
        run_best_search(
            run_config=run_config,
        )
    else:
        train_cmd(
            run_config=run_config,
        )


# ---------- eda command ----------
def main_eda() -> None:
    parser = argparse.ArgumentParser(
        description="Run exploratory data analysis and save figures"
    )
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/train.csv",
        type=str,
        help="Path to training CSV (default: datasets/train.csv)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="figures/eda",
        type=str,
        help="Directory to save EDA figures (default: figures/eda)",
    )
    args = parser.parse_args()

    from .data.feature_engineering import run_eda

    run_eda(dataset_path=args.dataset_path, output_dir=args.output_dir)


# ---------- compare command ----------
def main_compare() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple training runs and plot learning curves"
    )
    parser.add_argument(
        "run_folders",
        nargs="+",
        type=str,
        help="Paths to run folders (each must contain history.json)",
    )
    parser.add_argument(
        "--save",
        default=None,
        type=str,
        metavar="PATH",
        help="Save figure(s) to PATH (file or directory) instead of displaying",
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        default=True,
        help="Plot accuracy curves (default: True)",
    )
    parser.add_argument(
        "--no-accuracy",
        action="store_false",
        dest="accuracy",
        help="Do not plot accuracy",
    )
    parser.add_argument(
        "--f1",
        action="store_true",
        help="Plot F1 curves",
    )
    parser.add_argument(
        "--recall",
        action="store_true",
        help="Plot recall curves",
    )
    parser.add_argument(
        "--precision",
        action="store_true",
        help="Plot precision curves",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="Plot validation metrics only",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Plot training metrics only",
    )
    parser.add_argument(
        "--val-train",
        action="store_true",
        help="Plot both validation and training metrics (default)",
    )
    parser.add_argument(
        "--test",
        "-t",
        nargs="*",
        default=None,
        type=str,
        metavar="TEST_CSV",
        help="Evaluate each run's model on one or more test CSVs; metrics averaged (equal weight). Rank by test metrics and display top 5 per metric.",
    )
    args = parser.parse_args()
    from .model.compare import compare_cmd

    compare_cmd(
        run_folders=args.run_folders,
        save_path=args.save,
        plot_accuracy=args.accuracy,
        plot_f1=args.f1,
        plot_recall=args.recall,
        plot_precision=args.precision,
        plot_val=args.val,
        plot_train=args.train,
        plot_val_train=args.val_train,
        test_paths=args.test if args.test else None,
    )


# ---------- predict command ----------
def main_predict() -> None:
    parser = argparse.ArgumentParser(description="Predict using the trained model")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/test.csv",
        type=str,
        help="Path to CSV for prediction (default: datasets/test.csv). Preprocessing (fix + scaler) is applied.",
    )
    parser.add_argument(
        "--model-path",
        required=True,
        type=str,
        help="Path to saved model",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        type=str,
        help="Optional CSV path to save predictions",
    )
    args = parser.parse_args()
    from .model.predict import predict_cmd

    predict_cmd(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_path=args.output_path,
    )
