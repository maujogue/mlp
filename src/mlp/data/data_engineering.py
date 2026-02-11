import pandas as pd
from sklearn.model_selection import train_test_split
from ..utils.CustomStandardScaler import (
    CustomStandardScaler,
    save_scaler,
    load_scaler,
)
from ..utils.constants import FEATURE_COLUMNS, LABELS
from ..utils.loader import load_dataset


def fill_missing_values(df):
    df_copy = df.copy()
    for col in df_copy:
        if df_copy[col].isnull().any():
            mean_val = df_copy[col].mean()
            df_copy[col].fillna(mean_val, inplace=True)
    return df_copy


def add_column_titles(df):
    df_copy = df.copy()
    df_copy.columns = ["label"] + FEATURE_COLUMNS
    return df_copy


def fix_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy = df_copy.iloc[:, 1:]  # remove first column (id)
    df_copy = add_column_titles(df_copy)
    df_copy = fill_missing_values(df_copy)
    return df_copy


def pre_process(df, test=False):
    """
    Pre-processes the dataframe for model consumption.
    If test is False (training mode):
            - Fit scaler on features and save it.
            - Map and return labels.
    If test is True (test mode):
            - Load the fitted scaler.
            - Do not map labels (return as is if present, else None).
    """

    X = df[FEATURE_COLUMNS]
    Y = df["label"]
    Y_mapped = Y.map(LABELS)

    if not test:
        scaler = CustomStandardScaler()
        X_standardized = scaler.fit_transform(X)
        save_scaler(scaler)
        return X_standardized, Y_mapped
    else:
        scaler = load_scaler()
        X_standardized = scaler.transform(X)
        return X_standardized, Y_mapped


def split_train_validation(
    df: pd.DataFrame, split: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the input CSV into training and validation sets and saves them as
    new CSV files.
    """
    train_df, val_df = train_test_split(
        df,
        test_size=split,
        random_state=42,
        shuffle=True,
        stratify=df.iloc[:, 0],
    )
    train_df.to_csv("datasets/train.csv", index=False)
    val_df.to_csv("datasets/val.csv", index=False)

    return train_df, val_df


def split_cmd(dataset_path: str, test_size: float) -> None:
    dataset = load_dataset(dataset_path)
    dataset = fix_dataset(dataset)
    train_df, val_df = split_train_validation(dataset, test_size)
    train_df.to_csv("datasets/train.csv", index=False)
    val_df.to_csv("datasets/val.csv", index=False)

    print(f"Train set size: {train_df.shape}")
    print(f"Validation set size: {val_df.shape}")
    print(train_df.head())
    print(val_df.head())
