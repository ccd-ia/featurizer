# coding: utf-8

"""Graph φ-bridge exemplar: PageRank centrality (networkx, [bridge] extra).

φ per node = its PageRank on the graph induced by the edges knowable as-of the
cutoff. Unlike the per-row exemplars this is genuinely per-*node*, so it overrides
:meth:`materialize` to read an edge table, fit the graph on pre-t₀ edges, and
write a per-node ``(node_id, value)`` table the SQL spine joins to the node entity.

networkx is an optional dependency (``pip install 'featurizer[bridge]'``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BridgeComputer, assert_pre_t0


class PageRankBridge(BridgeComputer):
    def __init__(
        self,
        *,
        source_col: str,
        target_col: str,
        directed: bool = True,
        name: str = "pagerank",
        value_col: str = "pagerank",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="numeric")
        self.source_col = source_col
        self.target_col = target_col
        self.directed = directed

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, float]:
        """PageRank per node over the graph built from ``fit_rows`` edges."""
        try:
            import networkx as nx  # pyright: ignore[reportMissingImports]
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "PageRankBridge needs networkx: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        graph = nx.DiGraph() if self.directed else nx.Graph()
        for row in fit_rows:
            src, dst = row.get(self.source_col), row.get(self.target_col)
            if src is not None and dst is not None:
                graph.add_edge(src, dst)
        if graph.number_of_nodes() == 0:
            return {}
        return {node: float(score) for node, score in nx.pagerank(graph).items()}

    def materialize(  # type: ignore[override]
        self,
        conn: Any,
        *,
        edge_table: str,
        output_table: str,
        node_col: str = "node_id",
        causal_col: Optional[str] = None,
        fit_before: Any = None,
    ) -> str:
        """Read ``edge_table``, fit PageRank on pre-t₀ edges, write per-node φ."""
        select_cols = [self.source_col, self.target_col]
        if causal_col and causal_col not in select_cols:
            select_cols.append(causal_col)
        with conn.cursor() as cur:
            cur.execute(f"select {', '.join(select_cols)} from {edge_table}")
            names = [d.name for d in cur.description]
            rows = [dict(zip(names, r)) for r in cur.fetchall()]

        if causal_col and fit_before is not None:
            fit_rows = [
                r
                for r in rows
                if r.get(causal_col) is not None and r[causal_col] <= fit_before
            ]
            assert_pre_t0(fit_rows, fit_before, causal_col)
        else:
            fit_rows = rows

        scores = self.compute(rows, fit_rows=fit_rows)
        node_type = self._carry_type(self.source_col, rows)
        with conn.cursor() as cur:
            cur.execute(
                f"create temp table {output_table} "
                f"({node_col} {node_type}, {self.value_col} double precision) "
                "on commit drop"
            )
            cur.executemany(
                f"insert into {output_table} values (%s, %s)",
                list(scores.items()),
            )
        return output_table
