# coding: utf-8

"""Community membership + modularity via Louvain (python-louvain).

:class:`CommunityBridge` partitions the pre-t₀ graph with the Louvain method
and emits two columns per node (ADR-0014 multi-column contract):
``community_id`` — the membership label as a **categorical** variable (the
spine one-hots it through the existing ADR-0007 fixed-vocabulary path) — and
``modularity``, the partition's global modularity repeated per node (constant
within one snapshot; in snapshot-sequence mode it varies per as-of window and
the spine can trend it).

Community structure is non-local like centrality: rebuild per as-of window via
:meth:`~featurizer.bridge.base.BridgeComputer.materialize_snapshots`, never
slice one full-history partition.

SBM / MDL-surprise (the coordination engine of the taxonomy) is deliberately
**deferred**: it needs graph-tool, which is not pip-installable and would
break ``pip install 'featurizer[bridge]'``. It will land behind a separately
documented path; Louvain covers community detection in 0.9.0. python-louvain
is in the ``[bridge]`` extra. Louvain operates on undirected graphs — directed
edges are collapsed; self-loops are dropped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import MultiColumnBridge


class CommunityBridge(MultiColumnBridge):
    """Louvain community membership (categorical) + modularity per node.

    ``compute()`` reads edge rows from ``fit_rows`` and returns
    ``{node: {community_id: "c<N>", modularity: Q}}``. Labels are arbitrary
    but deterministic (``random_state=0``); they are *per-partition* names,
    not stable identities across snapshots — treat them as membership
    structure, not tracked communities.
    """

    def __init__(
        self,
        *,
        source_col: str,
        target_col: str,
        weight_col: Optional[str] = None,
        resolution: float = 1.0,
        name: str = "community",
    ) -> None:
        super().__init__(
            name=name,
            value_cols=["community_id", "modularity"],
            value_types={"community_id": "categorical"},
        )
        self.source_col = source_col
        self.target_col = target_col
        self.weight_col = weight_col
        self.resolution = resolution

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        try:
            import community as community_louvain  # pyright: ignore[reportMissingImports]
            import networkx as nx  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "CommunityBridge needs python-louvain (and networkx): "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        graph = nx.Graph()
        for row in fit_rows:
            src, dst = row.get(self.source_col), row.get(self.target_col)
            if src is None or dst is None or src == dst:  # drop self-loops
                continue
            if self.weight_col is not None:
                graph.add_edge(src, dst, weight=float(row.get(self.weight_col) or 0))
            else:
                graph.add_edge(src, dst)
        if graph.number_of_edges() == 0:
            return {}

        partition = community_louvain.best_partition(
            graph, resolution=self.resolution, random_state=0
        )
        modularity = float(community_louvain.modularity(partition, graph))
        return {
            node: {"community_id": f"c{label}", "modularity": modularity}
            for node, label in partition.items()
        }
