"""Tests for graph features over an edge-table entity."""

import tempfile

import yaml

from featurizer import Featurizer
from featurizer.primitives import ERGraph


def _render(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return Featurizer(handle.name).query


def _config(*, weight=None, timestamp="created_at", features=None) -> dict:
    edge = {"node": "users", "source": "follower_id", "target": "followee_id"}
    if weight:
        edge["weight"] = weight
    if timestamp:
        edge["timestamp"] = timestamp
    if features:
        edge["features"] = features
    return {
        "target": "users",
        "max_depth": 1,
        "intervals": [],
        "aggregations": ["mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "users", "table": "users", "id": "user_id"},
            {"alias": "follows", "table": "follows", "edge": edge},
        ],
    }


def test_edge_entity_parsed_into_spec():
    graph = ERGraph(_config()["entities"], relationships=None)
    users = graph.entities["users"]
    edges = graph.get_edges_for_node(users)
    assert len(edges) == 1
    spec = edges[0]
    assert spec.source == "follower_id"
    assert spec.target == "followee_id"
    assert spec.timestamp == "created_at"
    assert spec.node == "users"


def test_graph_cte_and_degree_features_rendered():
    sql = _render(_config())
    assert "follows_graph_for_users as (" in sql
    for metric in ("OUT_DEGREE", "IN_DEGREE", "DEGREE"):
        assert f'"{metric}(users.follows)"' in sql
    # Causal bound present when the edge has a timestamp.
    assert "created_at <= aod.as_of_date" in sql


def test_graph_without_timestamp_is_static_no_causal_bound():
    sql = _render(_config(timestamp=None))
    graph_cte = sql[sql.index("follows_graph_for_users as (") :]
    graph_cte = graph_cte[: graph_cte.index("group by node_id")]
    assert "aod.as_of_date" not in graph_cte


def test_weighted_degree_features_when_weight_present():
    sql = _render(_config(weight="weight"))
    assert '"WEIGHTED_OUT_DEGREE(users.follows)"' in sql
    assert '"WEIGHTED_IN_DEGREE(users.follows)"' in sql
    assert "sum(weight) filter (where direction = 'out')" in sql


# --------------------------------------------------------------------------- #
# M1b-2 recursive graph families
# --------------------------------------------------------------------------- #

ALL_FAMILIES = [
    "degree",
    "reciprocity",
    "k_hop_2",
    "clustering",
    "common_neighbours",
    "jaccard",
    "adamic_adar",
]


def test_default_features_are_degree_only():
    sql = _render(_config())
    assert "follows_graph_for_users as (" in sql
    for cte in (
        "_recip_for_",
        "_nbrs_for_",
        "_k2_for_",
        "_clust_for_",
        "_linkpred_for_",
    ):
        assert cte not in sql


def test_edge_spec_carries_requested_families():
    config = _config(features=["degree", "jaccard"])
    graph = ERGraph(config["entities"], relationships=None)
    spec = graph.get_edges_for_node(graph.entities["users"])[0]
    assert spec.features == ["degree", "jaccard"]


def test_all_families_render_their_ctes_and_columns():
    sql = _render(_config(features=ALL_FAMILIES))
    expected = {
        "follows_graph_for_users as (": '"DEGREE(users.follows)"',
        "follows_recip_for_users as (": '"RECIPROCITY(users.follows)"',
        "follows_k2_for_users as (": '"K_HOP_2_COUNT(users.follows)"',
        "follows_clust_for_users as (": '"CLUSTERING_COEFF(users.follows)"',
        "follows_linkpred_for_users as (": '"JACCARD_MEAN(users.follows)"',
    }
    for cte, column in expected.items():
        assert cte in sql, f"missing CTE {cte!r}"
        assert column in sql, f"missing column {column!r}"
    assert '"COMMON_NEIGHBOURS_MEAN(users.follows)"' in sql
    assert '"ADAMIC_ADAR_MEAN(users.follows)"' in sql
    # The recursive families share one undirected neighbour CTE.
    assert sql.count("follows_nbrs_for_users as (") == 1


def test_recursive_families_carry_the_causal_bound():
    sql = _render(_config(features=ALL_FAMILIES))
    nbrs = sql[sql.index("follows_nbrs_for_users as (") :]
    nbrs = nbrs[: nbrs.index(") u where")]
    assert nbrs.count("created_at <= aod.as_of_date") == 2  # both union arms
    recip = sql[sql.index("follows_recip_for_users as (") :]
    recip = recip[: recip.index("group by")]
    assert "e.created_at <= aod.as_of_date" in recip
    assert "r.created_at <= aod.as_of_date" in recip


def test_static_graph_families_have_no_causal_bound():
    sql = _render(_config(timestamp=None, features=ALL_FAMILIES))
    nbrs = sql[sql.index("follows_nbrs_for_users as (") :]
    nbrs = nbrs[: nbrs.index(") u where")]
    assert "aod.as_of_date" not in nbrs


def test_subset_of_linkpred_families_renders_only_requested_columns():
    sql = _render(_config(features=["jaccard"]))
    assert '"JACCARD_MEAN(users.follows)"' in sql
    assert "COMMON_NEIGHBOURS_MEAN" not in sql
    assert "ADAMIC_ADAR_MEAN" not in sql


def test_unknown_family_is_rejected_by_validation():
    import pytest

    config = _config(features=["degree", "jacard"])  # typo
    with pytest.raises(Exception) as excinfo:
        _render(config)
    assert "jacard" in str(excinfo.value)
