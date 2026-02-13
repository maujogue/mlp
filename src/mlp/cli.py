import argparse
from .utils.constants import BLUE, RESET


# ---------- split command ----------
def main_split() -> None:
    parser = argparse.ArgumentParser(
        description="Split the dataset into train and test sets"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset file",
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
        + "Splitted "
        + args.dataset_path
        + " with test size "
        + str(args.test_size)
        + RESET
    )


# ---------- train command ----------
def main_train() -> None:
    parser = argparse.ArgumentParser(description="Train the model on the dataset")
    parser.add_argument(
        "--train-path",
        default="datasets/train.csv",
        type=str,
        help="Path to training CSV (default: datasets/train.csv)",
    )
    parser.add_argument(
        "--val-path",
        default="datasets/val.csv",
        type=str,
        help="Path to validation CSV (default: datasets/val.csv)",
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
        "--model-path",
        default=None,
        type=str,
        metavar="PATH",
        help="Path where the model will be saved (default: temp/opts_timestamp/model.npz)",
    )
    parser.add_argument(
        "--curves-dir",
        default=None,
        type=str,
        metavar="DIR",
        help="Directory to save learning curves (default: temp/opts_timestamp/figures)",
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
        "--min-delta",
        default=0.0,
        type=float,
        help="Early stopping: minimum change in val_loss to count as improvement (default: 0.0)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Run a small hyperparameter grid and pick the best run (val precision, then fastest)",
    )
    args = parser.parse_args()
    from .model.compare import run_best_search
    from .model.training import train_cmd

    if args.best:
        run_best_search(
            train_path=args.train_path,
            val_path=args.val_path,
            epochs=args.epochs,
            seed=args.seed,
            min_delta=args.min_delta,
        )
    else:
        train_cmd(
            train_path=args.train_path,
            val_path=args.val_path,
            layers=args.layers,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            seed=args.seed,
            model_path=args.model_path,
            curves_dir=args.curves_dir,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            patience=args.patience,
            min_delta=args.min_delta,
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
    )


# ---------- predict command ----------
def main_predict() -> None:
    parser = argparse.ArgumentParser(description="Predict using the trained model")
    parser.add_argument(
        "dataset_path",
        nargs="?",
        default="datasets/val.csv",
        type=str,
        help="Path to CSV used for prediction (default: datasets/val.csv)",
    )
    parser.add_argument(
        "--model-path",
        default="weights/model",
        type=str,
        help="Path to saved model (default: weights/model)",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        type=str,
        help="Optional CSV path to save predictions",
    )
    args = parser.parse_args()
    from .model.training import predict_cmd

    predict_cmd(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        output_path=args.output_path,
    )
