from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import _get_feature_matrix, _require

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd


def plot_correlation_clustermap(
    self,
    method: str = "spearman",
    threshold: float | None = 0.95,
    figsize: tuple[int, int] = (14, 12),
    cmap: str = "RdBu_r",
) -> matplotlib.figure.Figure:
    """Plot hierarchically-clustered correlation heatmap.

    Args:
        method: Correlation method ('spearman', 'pearson', 'kendall').
        threshold: If set, annotate pairs above this absolute correlation.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure.
    """
    _require("seaborn")
    _require("matplotlib")
    import matplotlib.pyplot as plt
    import seaborn as sns

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    corr = matrix.corr(method=method)

    g = sns.clustermap(
        corr,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        figsize=figsize,
        linewidths=0.5,
        dendrogram_ratio=(0.1, 0.1),
    )
    g.fig.suptitle(f"Feature Correlation ({method.title()})", y=1.02, fontsize=14)

    if threshold is not None:
        import numpy as np

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        high_corr = corr.where(mask & (corr.abs() > threshold)).stack()
        if not high_corr.empty:
            print(f"\n{len(high_corr)} feature pairs above |{threshold}|:")
            for (f1, f2), val in high_corr.items():
                print(f"  {f1} <-> {f2}: {val:.3f}")

    return g.fig


def plot_redundancy_graph(
    self,
    threshold: float = 0.95,
    method: str = "spearman",
) -> object:
    """Plot network graph of highly correlated features.

    Args:
        threshold: Absolute correlation threshold for edges.
        method: Correlation method.

    Returns:
        plotly Figure (interactive).
    """
    _require("plotly")
    _require("networkx")
    import networkx as nx
    import numpy as np
    import plotly.graph_objects as go

    matrix = _get_feature_matrix(self.df, self.feature_cols)
    corr = matrix.corr(method=method)

    G = nx.Graph()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high_corr = corr.where(mask & (corr.abs() > threshold)).stack()

    for (f1, f2), val in high_corr.items():
        G.add_edge(f1, f2, weight=abs(val), correlation=val)

    if len(G.nodes) == 0:
        print(f"No feature pairs above |{threshold}| — try lowering threshold.")
        return None

    pos = nx.spring_layout(G, seed=42)

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = [f"{n} (degree={G.degree(n)})" for n in G.nodes()]
    node_color = [G.degree(n) for n in G.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n.split("(")[-1].rstrip(")") if "(" in n else n[:15] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale="YlOrRd",
            color=node_color,
            size=10,
            colorbar=dict(thickness=15, title="Degree"),
        ),
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=f"Feature Redundancy Graph (|corr| > {threshold})",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig
