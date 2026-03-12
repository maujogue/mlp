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

def scale_features(
	df: pd.DataFrame, scaler_path: str, training: bool = True
) -> pd.DataFrame:
	"""
	Scale feature columns using CustomStandardScaler. Label column is preserved.
	If training=True: fit on df and save scaler to scaler_path (default: model/scaler.pkl).
	If training=False: load scaler from scaler_path and transform df.
	"""
	df_copy = df.copy()
	if training:
		scaler = CustomStandardScaler()
		df_copy[FEATURE_COLUMNS] = scaler.fit_transform(df_copy[FEATURE_COLUMNS])
		save_scaler(scaler, scaler_path)
	else:
		scaler = load_scaler(scaler_path)
		df_copy[FEATURE_COLUMNS] = scaler.transform(df_copy[FEATURE_COLUMNS])
	return df_copy


def fit_scaler_on_train_and_transform_train_val(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	scaler_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""
	Fit CustomStandardScaler on train_df feature columns, transform both train and val
	with the same parameters, save scaler to scaler_path. Returns (train_scaled, val_scaled).
	"""
	scaler = CustomStandardScaler()
	scaler.fit(train_df[FEATURE_COLUMNS])
	train_out = train_df.copy()
	val_out = val_df.copy()
	train_out[FEATURE_COLUMNS] = scaler.transform(train_df[FEATURE_COLUMNS])
	val_out[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])
	save_scaler(scaler, scaler_path)
	return train_out, val_out


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
	)

	return train_df, val_df


def prepare_dataset_cmd(dataset_path: str, output_path: str) -> None:
	"""
	Load raw dataset, fix (remove id, add column titles, fill missing values)
	and save to output_path. Does not scale; scaling is done during training.
	"""
	df = load_dataset(dataset_path)
	df = fix_dataset(df)
	df.to_csv(output_path, index=False)


def split_cmd(dataset_path: str, test_size: float) -> None:
	"""
	Load a prepared dataset (already fixed, not scaled), split into train and test
	and save train.csv and test.csv in a folder named after the dataset (sans extension).
	"""
	import os

	df = load_dataset(dataset_path)
	train_df, test_df = split_train_validation(df, test_size)

	# Get dataset name without extension, ensure folder exists
	base_name = os.path.splitext(os.path.basename(dataset_path))[0]
	target_folder = os.path.join("datasets", base_name)
	os.makedirs(target_folder, exist_ok=True)

	train_df.to_csv(os.path.join(target_folder, "train.csv"), index=False)
	test_df.to_csv(os.path.join(target_folder, "test.csv"), index=False)
