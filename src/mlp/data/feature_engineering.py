"""
Exploratory Data Analysis (EDA) for the breast cancer dataset.
Produces: histograms by label, boxplots/violins, correlation heatmap, PCA scatter.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..utils.constants import FEATURE_COLUMNS, LABEL_COLORS


def _load_train_df(dataset_path: str) -> pd.DataFrame:
    """Load training CSV with label + feature columns."""
    df = pd.read_csv(dataset_path)
    if "label" not in df.columns or not all(c in df.columns for c in FEATURE_COLUMNS):
        raise ValueError(
            f"Expected columns 'label' and {FEATURE_COLUMNS[:3]}... in {dataset_path}"
        )
    return df


def plot_histograms_by_label(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 24),
    bins: int = 25,
) -> None:
    """
    Plot histograms of each feature, split by label (M vs B).
    Grid layout for 30 features.
    """
    n_features = len(FEATURE_COLUMNS)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        for label_key in ("B", "M"):
            subset = df[df["label"] == label_key][col]
            ax.hist(
                subset,
                bins=bins,
                alpha=0.6,
                label=label_key,
                color=LABEL_COLORS[label_key],
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("")
        ax.legend(loc="upper right", fontsize=6)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature histograms by label (B = Benign, M = Malignant)", fontsize=14)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_violins_by_label(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (18, 24),
) -> None:
    """
    For each feature, compare distributions for M and B with violin plots.
    """
    n_features = len(FEATURE_COLUMNS)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(FEATURE_COLUMNS):
        ax = axes[i]
        sns.violinplot(
            data=df,
            x="label",
            y=col,
            hue="label",
            order=["B", "M"],
            hue_order=["B", "M"],
            palette=LABEL_COLORS,
            legend=False,
            ax=ax,
        )
        ax.set_title(col, fontsize=8)
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Feature distributions by label (B = Benign, M = Malignant)", fontsize=14
    )
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (14, 12),
) -> None:
    """
    Correlation heatmap of all features.
    Highlights redundancy and potential for dimension reduction.
    """
    corr = df[FEATURE_COLUMNS].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        xticklabels=True,
        yticklabels=True,
        annot=False,
        fmt=".1f",
    )
    ax.tick_params(axis="both", labelsize=6)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_pca_scatter(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (8, 6),
) -> None:
    """
    PCA on scaled training features; plot PC1 vs PC2 colored by label.
    """
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=figsize)
    for label_key in ("B", "M"):
        mask = y == label_key
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            c=LABEL_COLORS[label_key],
            label=label_key,
            alpha=0.7,
            edgecolors="white",
            linewidth=0.3,
        )
    ax.set_xlabel(f"PC1 ({100 * pca.explained_variance_ratio_[0]:.1f}% var)")
    ax.set_ylabel(f"PC2 ({100 * pca.explained_variance_ratio_[1]:.1f}% var)")
    ax.set_title("PCA: PC1 vs PC2 (scaled features, colored by label)")
    ax.legend()
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def run_eda(
    dataset_path: str = "datasets/train.csv",
    output_dir: str | Path = "figures/eda",
) -> None:
    """
    Run full EDA pipeline and save all figures to output_dir.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_train_df(dataset_path)

    plot_histograms_by_label(df, output_path=output_dir / "histograms_by_label.png")
    plot_violins_by_label(df, output_path=output_dir / "violins_by_label.png")
    plot_correlation_heatmap(df, output_path=output_dir / "correlation_heatmap.png")
    plot_pca_scatter(df, output_path=output_dir / "pca_scatter.png")

    print(f"EDA figures saved to {output_dir.absolute()}")
