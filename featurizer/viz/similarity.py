from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import (
    _get_feature_matrix,
    _impute_median,
    _latest_slice,
    _require,
    _zscore,
)

if TYPE_CHECKING:
    import matplotlib.figure


def _entity_matrix(self, as_of_date):
    """Single-as-of, imputed, z-scored entity × feature matrix (+ the slice)."""
    sliced, resolved = _latest_slice(self.df, self.as_of_col, as_of_date)
    matrix = _zscore(_impute_median(_get_feature_matrix(sliced, self.feature_cols)))
    return matrix, sliced, resolved


def plot_entity_embedding(
    self,
    as_of_date: str | None = None,
    method: str = "umap",
    color_by: str | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> matplotlib.figure.Figure:
    """Plot UMAP/t-SNE/PCA scatter of entity feature vectors.

    Args:
        as_of_date: Filter to specific date. If None, uses latest.
        method: Dimensionality reduction method ('umap', 'tsne', 'pca').
        color_by: Column to color points by.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    import matplotlib.pyplot as plt

    matrix, sliced, resolved = _entity_matrix(self, as_of_date)
    fig, ax = plt.subplots(figsize=figsize)
    n = len(matrix)
    if n < 3:
        ax.text(
            0.5, 0.5, f"Need >=3 entities to embed (got {n})", ha="center", va="center"
        )
        return fig

    if method == "umap":
        _require("umap")
        import umap

        coords = umap.UMAP(n_components=2, random_state=42).fit_transform(matrix)
    elif method == "tsne":
        _require("sklearn")
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, n - 1))
        coords = TSNE(
            n_components=2, random_state=42, perplexity=perplexity, init="pca"
        ).fit_transform(matrix)
    elif method == "pca":
        _require("sklearn")
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=42).fit_transform(matrix)
    else:
        raise ValueError(f"Unknown method '{method}'; use 'umap', 'tsne', or 'pca'.")

    if color_by is not None and color_by in sliced.columns:
        import pandas as pd

        values = sliced.loc[matrix.index, color_by]
        if pd.api.types.is_numeric_dtype(values):
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap="viridis", s=30)
            fig.colorbar(sc, ax=ax, label=color_by)
        else:
            codes, labels = pd.factorize(values)
            sc = ax.scatter(coords[:, 0], coords[:, 1], c=codes, cmap="tab10", s=30)
            handles = [
                plt.Line2D([], [], marker="o", linestyle="", label=str(lab))
                for lab in labels
            ]
            ax.legend(handles=handles, title=color_by, fontsize=7)
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=30, color="steelblue")

    ax.set_title(f"Entity Embedding ({method}, as_of_date={resolved})")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    return fig


def plot_entity_dendrogram(
    self,
    as_of_date: str | None = None,
    figsize: tuple[int, int] = (14, 8),
) -> matplotlib.figure.Figure:
    """Plot hierarchical clustering dendrogram of entities.

    Args:
        as_of_date: Filter to specific date. If None, uses latest.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _require("matplotlib")
    _require("scipy")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    matrix, sliced, resolved = _entity_matrix(self, as_of_date)
    fig, ax = plt.subplots(figsize=figsize)
    if len(matrix) < 2:
        ax.text(0.5, 0.5, "Need >=2 entities to cluster", ha="center", va="center")
        return fig

    if self.entity_col in sliced.columns:
        labels = sliced.loc[matrix.index, self.entity_col].astype(str).tolist()
    else:
        labels = [str(i) for i in matrix.index]

    linkage_matrix = linkage(matrix.to_numpy(), method="ward")
    dendrogram(linkage_matrix, labels=labels, ax=ax, leaf_rotation=90, leaf_font_size=7)
    ax.set_title(f"Entity Dendrogram (as_of_date={resolved})")
    ax.set_ylabel("Ward distance")
    plt.tight_layout()
    return fig
