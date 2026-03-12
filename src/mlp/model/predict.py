import numpy as np
from pathlib import Path
from ..data.data_engineering import fix_dataset, split_features_labels
from ..utils.constants import FEATURE_COLUMNS, INDEX_TO_LABEL
from ..utils.loader import load_dataset
from .serialization import load_model
from .evaluation import evaluate


def _load_and_preprocess_for_predict(
    dataset_path: str, scaler_path: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load dataset, fix if needed, apply saved scaler; return (X, y) as arrays."""
    from ..data.data_engineering import scale_features

    df = load_dataset(dataset_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        df = fix_dataset(df)
    df = scale_features(df, training=False, scaler_path=scaler_path)
    X_df, y_mapped = split_features_labels(df)
    return X_df.to_numpy(dtype=np.float64), y_mapped.astype(np.int64).values


def predict_cmd(
    model_path: str,
    dataset_path: str = "datasets/test.csv",
    output_path: str | None = None,
) -> None:
    model, _ = load_model(model_path)
    model_dir = (
        Path(model_path) if Path(model_path).is_dir() else Path(model_path).parent
    )
    scaler_path = str(model_dir / "scaler.pkl")
    if not Path(scaler_path).exists():
        raise FileNotFoundError(
            f"Scaler not found at {scaler_path}. Use the full run directory as --model-path "
            "(e.g. temp/best_xxx/run_name) so the same scaler used at training is used here."
        )
    X, y = _load_and_preprocess_for_predict(dataset_path, scaler_path)

    p_arr = model.predict_proba(X)[:, 1]
    p_positive = p_arr.tolist()

    if output_path:
        df = load_dataset(dataset_path)
        df = df.copy()
        df["predicted_label"] = [
            INDEX_TO_LABEL[1 if p >= 0.5 else 0] for p in p_positive
        ]
        df["proba_m"] = p_positive
        df["proba_b"] = [1.0 - p for p in p_positive]
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    metrics = evaluate(model, X, y)
    print(f"Model: {model_dir}")
    print(f"Scaler: {scaler_path}")
    print(f"Dataset: {dataset_path} (n={len(y)})")
    print(f"binary_cross_entropy: {metrics['loss']:.6f}")
    print(f"accuracy: {metrics['accuracy']:.6f}")
    print(
        "(BCE is on this dataset only; it can differ from validation BCE if this is a different file.)"
    )
