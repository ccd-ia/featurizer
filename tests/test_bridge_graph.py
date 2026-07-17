"""Unit tests for the graph φ-bridges (no database).

Centralities are checked against hand-computed values on a drawn graph — a
triangle (A, B, C) with a pendant node D attached to A — and the snapshot
sequence is proven causal by feeding edges spanning two as-of dates and
asserting the earlier window excludes the later edge entirely. Community
detection is checked on two cliques joined by a single bridge edge.

networkx / python-louvain sit in the dev dependency group precisely so these
tests execute (not skip) under plain ``uv sync``.
"""

from __future__ import annotations

from datetime import date

import pytest

from featurizer.bridge import CentralityBridge, CommunityBridge
from featurizer.bridge.centrality import CHEAP_METRICS, HEAVY_METRICS

TRIANGLE_PLUS_PENDANT = [
    {"src": "A", "dst": "B"},
    {"src": "B", "dst": "C"},
    {"src": "A", "dst": "C"},
    {"src": "A", "dst": "D"},
]


def _undirected(**kw):
    return CentralityBridge(source_col="src", target_col="dst", directed=False, **kw)


# --------------------------------------------------------------------------- #
# Cheap tier: hand-computed values
# --------------------------------------------------------------------------- #


def test_cheap_tier_is_the_default_column_set():
    bridge = _undirected()
    assert bridge.value_cols == list(CHEAP_METRICS)
    heavy = _undirected(include_heavy=True)
    assert heavy.value_cols == list(CHEAP_METRICS) + list(HEAVY_METRICS)


def test_centralities_match_hand_computation():
    phi = _undirected().compute(TRIANGLE_PLUS_PENDANT, fit_rows=TRIANGLE_PLUS_PENDANT)
    assert phi["A"]["degree"] == 3.0
    assert phi["B"]["degree"] == 2.0
    assert phi["D"]["degree"] == 1.0
    # Undirected: in/out equal total degree (fixed column shape).
    assert phi["A"]["in_degree"] == phi["A"]["out_degree"] == 3.0
    # Clustering: A sees one edge (B-C) among its 3 neighbour pairs; B and C
    # sit in a closed triangle; a pendant has no pairs.
    assert phi["A"]["clustering"] == pytest.approx(1 / 3)
    assert phi["B"]["clustering"] == pytest.approx(1.0)
    assert phi["D"]["clustering"] == 0.0
    # k-core: the triangle is the 2-core, the pendant is 1-core.
    assert phi["A"]["coreness"] == 2.0
    assert phi["D"]["coreness"] == 1.0
    # No weight column declared -> weighted_degree mirrors degree.
    assert phi["A"]["weighted_degree"] == 3.0


def test_heavy_metrics_are_opt_in_with_hand_values():
    cheap = _undirected().compute(TRIANGLE_PLUS_PENDANT, fit_rows=TRIANGLE_PLUS_PENDANT)
    assert "betweenness" not in cheap["A"]

    phi = _undirected(include_heavy=True).compute(
        TRIANGLE_PLUS_PENDANT, fit_rows=TRIANGLE_PLUS_PENDANT
    )
    # A carries the only paths B-D and C-D: raw 2, normalized 2/((4-1)(4-2)/2).
    assert phi["A"]["betweenness"] == pytest.approx(2 / 3)
    assert phi["D"]["betweenness"] == 0.0
    # A is adjacent to everyone: closeness 1.
    assert phi["A"]["closeness"] == pytest.approx(1.0)
    assert phi["D"]["closeness"] == pytest.approx(3 / 5)
    assert phi["A"]["eigenvector"] > phi["D"]["eigenvector"]


def test_directed_in_out_degree():
    bridge = CentralityBridge(source_col="src", target_col="dst", directed=True)
    phi = bridge.compute(
        [
            {"src": "A", "dst": "B"},
            {"src": "A", "dst": "C"},
            {"src": "B", "dst": "A"},
        ],
        fit_rows=[
            {"src": "A", "dst": "B"},
            {"src": "A", "dst": "C"},
            {"src": "B", "dst": "A"},
        ],
    )
    assert phi["A"]["out_degree"] == 2.0
    assert phi["A"]["in_degree"] == 1.0
    assert phi["A"]["degree"] == 3.0
    assert phi["C"]["out_degree"] == 0.0


def test_weighted_degree_uses_the_weight_column():
    bridge = _undirected(weight_col="w")
    phi = bridge.compute(
        [{"src": "A", "dst": "B", "w": 2.0}, {"src": "A", "dst": "C", "w": 0.5}],
        fit_rows=[
            {"src": "A", "dst": "B", "w": 2.0},
            {"src": "A", "dst": "C", "w": 0.5},
        ],
    )
    assert phi["A"]["degree"] == 2.0
    assert phi["A"]["weighted_degree"] == pytest.approx(2.5)


def test_self_loops_dropped_and_empty_graph_is_empty():
    bridge = _undirected()
    assert bridge.compute([], fit_rows=[]) == {}
    assert (
        bridge.compute([{"src": "A", "dst": "A"}], fit_rows=[{"src": "A", "dst": "A"}])
        == {}
    )
    phi = bridge.compute(
        [{"src": "A", "dst": "A"}, {"src": "A", "dst": "B"}],
        fit_rows=[{"src": "A", "dst": "A"}, {"src": "A", "dst": "B"}],
    )
    assert phi["A"]["degree"] == 1.0


# --------------------------------------------------------------------------- #
# Snapshot sequence: the pre-t₀ graph excludes the later edge
# --------------------------------------------------------------------------- #

DATED_EDGES = [
    {"src": "A", "dst": "B", "ts": date(2020, 1, 1)},
    {"src": "B", "dst": "C", "ts": date(2020, 2, 1)},
    {"src": "A", "dst": "C", "ts": date(2020, 3, 1)},
    {"src": "A", "dst": "D", "ts": date(2020, 9, 1)},  # future at the June cut
]


def test_snapshot_sequence_excludes_future_edges_per_window():
    bridge = _undirected()
    snaps = bridge.compute_snapshots(
        DATED_EDGES,
        as_of_dates=[date(2020, 6, 1), date(2020, 12, 1)],
        causal_col="ts",
    )
    june, december = date(2020, 6, 1), date(2020, 12, 1)
    # June window: the triangle only — D does not exist yet.
    assert ("D", june) not in snaps
    assert snaps[("A", june)]["degree"] == 2.0
    assert snaps[("A", june)]["clustering"] == pytest.approx(1.0)
    # December window: the pendant has arrived and dilutes A's clustering.
    assert snaps[("A", december)]["degree"] == 3.0
    assert snaps[("A", december)]["clustering"] == pytest.approx(1 / 3)
    assert snaps[("D", december)]["degree"] == 1.0


# --------------------------------------------------------------------------- #
# Community membership
# --------------------------------------------------------------------------- #

TWO_CLIQUES = [
    {"src": s, "dst": d}
    for s, d in [
        ("a1", "a2"),
        ("a1", "a3"),
        ("a2", "a3"),
        ("b1", "b2"),
        ("b1", "b3"),
        ("b2", "b3"),
        ("a3", "b1"),  # the bridge edge
    ]
]


def test_community_membership_separates_the_cliques():
    bridge = CommunityBridge(source_col="src", target_col="dst")
    phi = bridge.compute(TWO_CLIQUES, fit_rows=TWO_CLIQUES)
    a_side = {phi[n]["community_id"] for n in ("a1", "a2", "a3")}
    b_side = {phi[n]["community_id"] for n in ("b1", "b2", "b3")}
    assert len(a_side) == 1 and len(b_side) == 1
    assert a_side != b_side
    assert all(cid.startswith("c") for cid in a_side | b_side)
    # Global modularity, repeated per node, positive for real structure.
    modularities = {v["modularity"] for v in phi.values()}
    assert len(modularities) == 1
    assert modularities.pop() > 0


def test_community_membership_is_categorical_in_emit_yaml():
    bridge = CommunityBridge(source_col="src", target_col="dst")
    fragment = bridge.emit_yaml(
        output_table="t",
        pk="node_id",
        parent_alias="nodes",
        parent_key="node_id",
        fk="node_id",
    )
    assert fragment["entity"]["variables"] == {
        "community_id": {"type": "categorical"},
        "modularity": {"type": "numeric"},
    }


def test_community_empty_graph_is_empty():
    bridge = CommunityBridge(source_col="src", target_col="dst")
    assert bridge.compute([], fit_rows=[]) == {}
