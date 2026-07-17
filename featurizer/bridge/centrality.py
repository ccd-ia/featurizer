# coding: utf-8

"""Multi-metric graph centralities — one build, many columns (networkx).

:class:`CentralityBridge` computes a node's full centrality profile from a
single graph build (the ADR-0014 multi-column contract): the **cheap tier is
on by default** (degree / in / out / weighted, coreness, clustering
coefficient — all near-linear) and every heavier metric — betweenness O(V·E),
eigenvector, closeness — is **opt-in** via ``include_heavy=True`` so a feature
config never gets silently 100× slower.

Centrality is *non-local*: one future edge changes every node's score, so a
backtest cohort must rebuild the graph per as-of window from strictly pre-t₀
edges — :meth:`~featurizer.bridge.base.BridgeComputer.materialize_snapshots`
does exactly that (cost **O(windows × build)**, no snapshot-binning
approximation; the cheap default tier keeps it bounded). For a static or given
graph, :meth:`~featurizer.bridge.base.BridgeComputer.materialize_nodes` is the
single-snapshot fast path. Never compute one full-history graph and slice it —
that leaks the future (ADR-0001/ADR-0014).

networkx is an optional dependency (``pip install 'featurizer[bridge]'``).
Self-loop edges are dropped at build time (they distort degree and break
k-core computation).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from .base import MultiColumnBridge

CHEAP_METRICS = (
    "degree",
    "in_degree",
    "out_degree",
    "weighted_degree",
    "coreness",
    "clustering",
)
HEAVY_METRICS = ("betweenness", "eigenvector", "closeness")


class CentralityBridge(MultiColumnBridge):
    """Per-node centrality profile from one graph build.

    ``compute()`` reads edge rows (``source_col``, ``target_col``, optional
    ``weight_col``) from ``fit_rows`` and returns ``{node: {metric: value}}``
    for every node in the pre-t₀ graph. On an undirected graph
    (``directed=False``) ``in_degree``/``out_degree`` equal ``degree``; with
    no ``weight_col``, ``weighted_degree`` equals ``degree`` — column shape
    stays fixed so downstream configs never change.
    """

    def __init__(
        self,
        *,
        source_col: str,
        target_col: str,
        directed: bool = True,
        weight_col: Optional[str] = None,
        include_heavy: bool = False,
        name: str = "centrality",
    ) -> None:
        metrics = list(CHEAP_METRICS) + (list(HEAVY_METRICS) if include_heavy else [])
        super().__init__(name=name, value_cols=metrics)
        self.source_col = source_col
        self.target_col = target_col
        self.directed = directed
        self.weight_col = weight_col
        self.include_heavy = include_heavy

    def _build_graph(self, fit_rows: List[Dict[str, Any]]) -> Any:
        try:
            import networkx as nx  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "CentralityBridge needs networkx: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        graph = nx.DiGraph() if self.directed else nx.Graph()
        for row in fit_rows:
            src, dst = row.get(self.source_col), row.get(self.target_col)
            if src is None or dst is None or src == dst:  # drop self-loops
                continue
            if self.weight_col is not None:
                weight = row.get(self.weight_col)
                graph.add_edge(src, dst, weight=float(weight or 0.0))
            else:
                graph.add_edge(src, dst)
        return graph

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        import networkx as nx  # pyright: ignore[reportMissingImports]

        graph = self._build_graph(fit_rows)
        if graph.number_of_nodes() == 0:
            return {}

        degree = dict(graph.degree())
        if self.directed:
            in_degree = dict(graph.in_degree())
            out_degree = dict(graph.out_degree())
        else:
            in_degree = out_degree = degree
        weighted = (
            dict(graph.degree(weight="weight"))
            if self.weight_col is not None
            else degree
        )
        coreness = nx.core_number(graph)
        clustering = cast(
            Dict[Any, float],
            nx.clustering(graph.to_undirected() if self.directed else graph),
        )

        heavy: Dict[str, Dict[Any, Any]] = {}
        if self.include_heavy:
            heavy["betweenness"] = nx.betweenness_centrality(graph)
            heavy["closeness"] = nx.closeness_centrality(graph)
            try:
                heavy["eigenvector"] = nx.eigenvector_centrality(graph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                heavy["eigenvector"] = {}  # non-convergent graph -> NULLs

        out: Dict[Any, Dict[str, Any]] = {}
        for node in graph.nodes:
            metrics: Dict[str, Any] = {
                "degree": float(degree[node]),
                "in_degree": float(in_degree[node]),
                "out_degree": float(out_degree[node]),
                "weighted_degree": float(weighted[node]),
                "coreness": float(coreness[node]),
                "clustering": float(clustering[node]),
            }
            for metric, scores in heavy.items():
                value = scores.get(node)
                metrics[metric] = None if value is None else float(value)
            out[node] = metrics
        return out
