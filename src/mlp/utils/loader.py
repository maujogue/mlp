import logging
import pandas as pd
import sys
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)

        logger.info(f"Loading dataset of dimensions {df.shape}")

        return df

    except FileNotFoundError:
        logger.error(f"File '{path}' not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"File '{path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError:
        logger.error(f"File '{path}' has bad format and cannot be parsed.")
        sys.exit(1)
    except PermissionError:
        logger.error(f"Permission denied to read file '{path}'.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{path}': {e}")
        sys.exit(1)


def _sanitize(value: Any) -> str:
    """Turn an option value into a filesystem-safe string (no spaces, path separators)."""
    if isinstance(value, list):
        return "-".join(str(v) for v in value)
    s = str(value).strip().lower()
    s = re.sub(r"[^\w\-.]", "_", s)
    return s or "default"


def build_run_dir(
    root: str = "temp",
    *,
    train_path: str = "",
    layers: list[int] | None = None,
    epochs: int = 70,
    learning_rate: float = 0.03,
    seed: int = 42,
    batch_size: int = 0,
    optimizer: str = "sgd",
    patience: int = 0,
) -> str:
    """Build a run directory name: root/option1-value1_option2-value2_<timestamp>."""
    parts = []
    if train_path:
        parts.append(f"train-{_sanitize(Path(train_path).stem)}")
    parts.append(f"layers-{_sanitize(layers or [24, 24])}")
    parts.append(f"epochs-{epochs}")
    parts.append(f"lr-{_sanitize(learning_rate)}")
    parts.append(f"seed-{seed}")
    parts.append(f"batch-{batch_size if batch_size > 0 else 'full'}")
    parts.append(f"optim-{optimizer}")
    parts.append(f"patience-{patience}")
    timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    name = "_".join(parts) + "_" + timestamp
    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)
