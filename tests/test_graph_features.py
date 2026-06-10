"""Tests for graph (degree) features over an edge-table entity."""

import tempfile

import yaml

from featurizer import Featurizer
from featurizer.primitives import ERGraph


def _render(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return Featurizer(handle.name).query


def _config(*, weight=None, timestamp="created_at") -> dict:
    edge = {"node": "users", "source": "follower_id", "target": "followee_id"}
    if weight:
        edge["weight"] = weight
    if timestamp:
        edge["timestamp"] = timestamp
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
