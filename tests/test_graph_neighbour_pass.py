"""DB-free SQL-shape guards for the native 1-hop graph pass (0.9.0 Phase 4).

The generated CTE must carry BOTH causal bounds of the taxonomy's cheap tier —
the edge timestamp and the neighbour state's temporal index, each cut at
``aod.as_of_date`` — and must stay strictly 1-hop: the edge table is never
joined to itself, so no neighbour-of-neighbour aggregate (the canonical
temporal-GNN leakage) can be expressed. Validation quality (typo suggestions,
required-key errors) is covered alongside.
"""

from __future__ import annotations

import tempfile

import yaml

from featurizer import Featurizer
from featurizer.validation import ConfigValidator


def _render(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path).query


def _segment(sql: str, start_marker: str, end_marker: str) -> str:
    start = sql.index(start_marker)
    end = sql.index(end_marker, start)
    return sql[start:end]


def _graph_config(
    directed: bool = True,
    intervals: list[str] | None = None,
    features: list[str] | None = None,
    **spec_overrides,
) -> dict:
    spec: dict = {
        "name": "contacts",
        "left": "facilities",
        "edges": {
            "table": "contact_edges",
            "source": "src_id",
            "target": "dst_id",
            "timestamp": "contacted_at",
        },
        "directed": directed,
    }
    if features is not None:
        spec["features"] = features
    spec.update(spec_overrides)
    return {
        "target": "facilities",
        "max_depth": 1,
        "intervals": intervals or [],
        "aggregations": ["count"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "facilities",
                "table": "facilities",
                "id": "facility_id",
                "temporal_ix": "valid_at",
                "variables": {
                    "risk_score": {"type": "numeric"},
                    "flagged": {"type": "boolean"},
                },
            }
        ],
        "graph_relationships": [spec],
    }


CTE = "graph_rel_contacts_for_facilities"


def test_cte_is_defined_and_joined_by_id():
    sql = _render(_graph_config())
    assert f"{CTE} as (" in sql
    assert f"{CTE}.node_id = facilities.facility_id" in sql


def test_both_causal_bounds_are_present():
    """The leakage guard: pre-t₀ edges AND pre-t₀ neighbour states."""
    sql = _render(_graph_config())
    cte = _segment(sql, f"{CTE} as (", "facilities_synth as (")
    assert "e.contacted_at <= aod.as_of_date" in cte  # edge bound
    assert "n.valid_at <= aod.as_of_date" in cte  # neighbour-state bound


def test_stays_one_hop_only():
    """The edge table is never joined to itself — no 2-hop is expressible."""
    sql = _render(_graph_config())
    assert "join contact_edges" not in sql
    # Directed: one incidence arm in the degree subquery, one in the
    # neighbour subquery — exactly two scans, both plain FROMs.
    assert sql.count("from contact_edges e") == 2
    # The neighbour join targets the state table on the 1-hop neighbour id.
    assert "inner join facilities n on n.facility_id = inc.nbr" in sql


def test_undirected_unions_both_incidence_directions():
    directed = _segment(
        _render(_graph_config(directed=True)), f"{CTE} as (", "facilities_synth as ("
    )
    assert "union all" not in directed
    undirected = _segment(
        _render(_graph_config(directed=False)), f"{CTE} as (", "facilities_synth as ("
    )
    assert "union all" in undirected
    assert "e.dst_id as node_id" in undirected  # the reversed arm


def test_degree_windowed_per_configured_interval():
    sql = _render(_graph_config(intervals=["P3M"]))
    assert '"DEGREE(contacts)"' in sql
    assert '"DEGREE(contacts|interval=P3M)"' in sql  # the house naming style
    cte = _segment(sql, f"{CTE} as (", "facilities_synth as (")
    assert "daterange((aod.as_of_date - interval 'P3M')::date" in cte
    assert "@> inc.ts::date" in cte


def test_neighbour_columns_default_to_declared_variable_types():
    """measures default to the right entity's numeric variables and shares to
    its boolean ones; the share casts the flag to int for an avg share."""
    sql = _render(_graph_config())
    assert '"NEIGHBOUR_MEAN(contacts.risk_score)"' in sql
    assert '"NEIGHBOUR_SHARE(contacts.flagged)"' in sql
    assert "avg(n.risk_score)" in sql
    assert "avg((n.flagged)::int)" in sql


def test_degree_only_selection_skips_the_neighbour_join():
    sql = _render(_graph_config(features=["degree"]))
    assert '"DEGREE(contacts)"' in sql
    assert "NEIGHBOUR_MEAN" not in sql
    assert "inner join facilities n" not in sql


def test_explicit_measures_and_shares_override_defaults():
    sql = _render(_graph_config(measures=["risk_score"], shares=[]))
    assert '"NEIGHBOUR_MEAN(contacts.risk_score)"' in sql
    assert "NEIGHBOUR_SHARE" not in sql


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _validate(config: dict):
    return ConfigValidator().validate(config)


def test_validation_requires_edge_timestamp():
    config = _graph_config()
    del config["graph_relationships"][0]["edges"]["timestamp"]
    result = _validate(config)
    assert any(
        "edges.timestamp" in e.location and "as-of" in (e.suggestion or "")
        for e in result.errors
    )


def test_validation_suggests_entity_for_typo():
    config = _graph_config()
    config["graph_relationships"][0]["left"] = "facilites"
    result = _validate(config)
    assert any(
        "unknown entity 'facilites'" in e.message
        and "facilities" in (e.suggestion or "")
        for e in result.errors
    )


def test_validation_rejects_two_hop_family_with_reason():
    config = _graph_config(features=["degree", "k2_neighbour_mean"])
    result = _validate(config)
    assert any(
        "k2_neighbour_mean" in e.message and "1-hop only" in e.message
        for e in result.errors
    )


def test_validation_suggests_family_for_typo():
    config = _graph_config(features=["neighbour_men"])
    result = _validate(config)
    assert any(
        "neighbour_men" in e.message and "neighbour_mean" in (e.suggestion or "")
        for e in result.errors
    )


def test_validation_checks_measure_columns_against_right_entity():
    config = _graph_config(measures=["risk_scor"])
    result = _validate(config)
    assert any(
        "risk_scor" in e.message and "risk_score" in (e.suggestion or "")
        for e in result.errors
    )


def test_validation_warns_on_non_boolean_share():
    config = _graph_config(shares=["risk_score"])
    result = _validate(config)
    assert any(
        "risk_score" in w.message and "boolean" in w.message for w in result.warnings
    )


def test_valid_config_passes_validation():
    result = _validate(_graph_config(intervals=["P3M"]))
    assert not [e for e in result.errors if "graph_relationships" in (e.location or "")]
