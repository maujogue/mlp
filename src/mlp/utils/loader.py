import logging
import pandas as pd
import sys

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
