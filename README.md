# MLP — Neural network from scratch

A **multilayer perceptron (MLP)** implemented in NumPy from scratch (no autograd) for **binary classification** of breast cell malignancy (e.g. Wisconsin Breast Cancer–style data).

## Features

- **Pure NumPy MLP**: vectorized forward/backward pass, ReLU or sigmoid hidden layers, 2-class output
- **Optimizers**: SGD and RMSprop
- **Training**: train/validation split, learning curves, early stopping, optional batch training
- **Hyperparameter search**: `--best` to run a grid and pick the best run; optional test-set ranking
- **Pipeline**: prepare dataset → split → train → predict, plus EDA and run comparison

## Requirements

- **Python** ≥ 3.13, &lt; 3.14
- **Dependencies**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pydantic`

## Installation

From the project root:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

This installs the `mlp` package and the CLI commands below.

## Dataset format

- **Input**: CSV with an ID column (dropped), then 30 feature columns and a label column (e.g. `M`/`B` for malignant/benign).
- **Prepared CSV**: use `mlp-prepare-dataset` to fix column names and fill missing values. Scaling is applied during training (and reused at predict time).

## Usage

### 1. Prepare the dataset

Fix column titles and fill missing values. Output path defaults to `<stem>_prepared.csv` next to the input.

```bash
mlp-prepare-dataset path/to/raw_data.csv [-o path/to/prepared.csv]
```

### 2. Split into train and test

```bash
mlp-split path/to/prepared.csv [--test-size 0.2]
```

Creates `train.csv` and `test.csv` in the same directory as the prepared file.

### 3. Train

Train on the training CSV (with internal train/validation split). Model and learning curves are always written under `temp/<opts>_<timestamp>/` (e.g. `model.pkl`, `scaler.pkl`, `figures/`).

```bash
mlp-train path/to/train.csv [options]
```

**Common options:**

| Option | Default | Description |
|--------|--------|-------------|
| `--layers` | `24 24` | Hidden layer sizes (space-separated) |
| `--epochs` | `70` | Number of epochs |
| `--learning-rate` | `0.03` | Learning rate |
| `--val-ratio` | `0.2` | Fraction of train used for validation |
| `--optimizer` | `sgd` | `sgd` or `rmsprop` |
| `--batch-size` | `0` | Batch size; `0` = full dataset |
| `--patience` | `0` | Early stopping (epochs without improvement); `0` = off |
| `--seed` | `42` | Random seed |

**Hyperparameter search (best run):**

```bash
mlp-train path/to/train.csv --best [test1.csv test2.csv ...]
```

Runs a grid over layers/lr/optimizer and picks the best run. If one or more test CSVs are given, best models are ranked by average metrics on those sets.

### 4. Predict

Run inference with a saved model. Preprocessing (fix + scaler) is applied automatically. Use a run directory (e.g. `temp/opts_xxx/`) or the path to `model.pkl` so the same scaler used at training is found.

```bash
mlp-predict [path/to/test.csv] --model-path path/to/run_dir_or_model.pkl [--output-path predictions.csv]
```

### 5. Exploratory data analysis (EDA)

Generate EDA figures from a training CSV:

```bash
mlp-eda [path/to/train.csv] [--output-dir figures/eda]
```

### 6. Compare runs

Compare multiple training runs (each folder must contain `history.json`) and plot learning curves:

```bash
mlp-compare run_folder1 run_folder2 ... [--save path_or_dir] [--test test1.csv test2.csv]
```

Optional: `--accuracy`, `--f1`, `--recall`, `--precision`, `--val`, `--train`, `--val-train`.

## Development

From the project root:

```bash
make format   # ruff format
make lint    # ruff check
make type    # ty check
make all     # format + lint + type
```

## Project structure

```
mlp/
├── Makefile
├── pyproject.toml
├── src/
│   └── mlp/
│       ├── cli.py                  # CLI entrypoints
│       ├── data/
│       │   ├── data_engineering.py  # prepare, split, scale, load
│       │   └── feature_engineering.py # EDA
│       ├── model/
│       │   ├── mlp_classifier.py    # MLPClassifier (NumPy)
│       │   ├── schemas.py           # Pydantic config & training history
│       │   ├── training.py          # train command
│       │   ├── predict.py           # predict command
│       │   ├── evaluation.py        # metrics, evaluate on dataset
│       │   ├── compare.py           # best search + run comparison
│       │   ├── plots.py             # learning curves
│       │   └── serialization.py    # save/load model + scaler
│       └── utils/
│           ├── constants.py         # feature names, labels
│           ├── loader.py            # dataset loading
│           └── CustomStandardScaler.py
└── README.md
```

## License

See repository or project rules. Author: Mathis.
