import pandas as pd
import pickle
from pathlib import Path


class CustomStandardScaler:
    """
    Custom standard scaler that scales the data to have a mean of 0 and a
    standard deviation of 1.
    """

    def fit(self, X: pd.DataFrame):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        # Avoid division by zero for constant columns
        self.scale_.loc[self.scale_ == 0] = 1.0

    def transform(self, X: pd.DataFrame):
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: pd.DataFrame):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame):
        return X * self.scale_ + self.mean_


def save_scaler(scaler, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to {filepath}")


def load_scaler(filepath: Path):
    if not filepath.exists():
        raise FileNotFoundError(f"Scaler file not found: {filepath}")

    with open(filepath, "rb") as f:
        scaler = pickle.load(f)
    return scaler
