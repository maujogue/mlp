import numpy as np
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


def split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
	"""
	Separates the dataframe into features (X) and labels (Y).
	Returns features and mapped labels.
	"""
	X = df[FEATURE_COLUMNS]
	Y = df["label"]
	Y_mapped = Y.map(LABELS)
	return X, Y_mapped

def scale_features(df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
	"""
	Scales feature columns in-place within the provided DataFrame using CustomStandardScaler.
	Label column is preserved. The DataFrame structure is unchanged except for the scaled features.
	If training=True, fit and save scaler. If training=False, load and use saved scaler.
	"""
	df_copy = df.copy()
	if training:
		scaler = CustomStandardScaler()
		df_copy[FEATURE_COLUMNS] = scaler.fit_transform(df_copy[FEATURE_COLUMNS])
		save_scaler(scaler)
	else:
		scaler = load_scaler()
		df_copy[FEATURE_COLUMNS] = scaler.transform(df_copy[FEATURE_COLUMNS])
	return df_copy


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

	return train_df, val_df


def split_cmd(dataset_path: str, test_size: float) -> None:
	df = load_dataset(dataset_path)
	df = fix_dataset(df)

	train_df, val_df = split_train_validation(df, test_size)
	train_df = scale_features(train_df, training=True)
	val_df = scale_features(val_df, training=False)
	train_df.to_csv("datasets/train.csv", index=False)
	val_df.to_csv("datasets/val.csv", index=False)
