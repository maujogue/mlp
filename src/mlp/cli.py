import argparse

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
    from .loader import load_dataset
    dataset = load_dataset(args.dataset_path)
    print(f"Splitting {args.dataset_path} with test size {args.test_size}")
    print(dataset.head())
    # TODO: Call the actual split function here


def main_train() -> None:
    parser = argparse.ArgumentParser(
        description="Train the model on the dataset"
    )
    # Add arguments here if needed (e.g., epochs, batch size)
    args = parser.parse_args()
    print("Training the model...")
    # TODO: Call the actual training function here


def main_predict() -> None:
    parser = argparse.ArgumentParser(
        description="Predict using the trained model"
    )
    # Add arguments here if needed (e.g., input file)
    args = parser.parse_args()
    print("Predicting with the trained model...")
    # TODO: Call the actual prediction function here
