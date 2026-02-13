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
        default="weights/model",
        type=str,
        help="Path where the model will be saved (default: weights/model, viewable .txt)",
    )
    parser.add_argument(
        "--curves-dir",
        default="figures/training",
        type=str,
        help="Directory to save learning curves (default: figures/training)",
    )
    parser.add_argument(
        "--batch-size",
        default=0,
        type=int,
        help="Batch size; 0 = full dataset (default: 0)",
    )
    args = parser.parse_args()
    from .model.training import train_cmd

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
