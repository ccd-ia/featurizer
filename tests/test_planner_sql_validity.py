"""DB-free guards for SQL-validity bugs in the planner.

These assert on the *shape* of the generated SQL so the regressions stay caught
even when no PostgreSQL is configured (the integration harness skips then).
They cover:

- Bug #1: the transform CTE used to re-render aggregate *definitions*
  (``avg(amount)``) against the synth CTE, which only exposes those columns by
  their aggregated *name*. Passthrough/synth-level features must be referenced
  by name.
- Bug #2: a child reached at the max_depth boundary was aggregated but never
  materialized, so the aggregation CTE referenced an undefined
  ``<child>_transform``. Every reached entity must emit its transform CTE.
"""

import re
import tempfile

import yaml

from featurizer import Featurizer


def _render(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path).query


def _parent_child_config(max_depth: int = 2) -> dict:
    return {
        "target": "customers",
        "max_depth": max_depth,
        "intervals": [],
        "aggregations": ["sum", "mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def _segment(sql: str, start_marker: str, end_marker: str) -> str:
    """Return the substring of ``sql`` between two markers (start inclusive)."""
    start = sql.index(start_marker)
    end = sql.index(end_marker, start)
    return sql[start:end]


def test_child_transform_cte_is_defined_at_depth_boundary():
    """Bug #2: the aggregated child must emit its own transform CTE."""
    sql = _render(_parent_child_config(max_depth=2))
    assert "from orders_transform" in sql  # the aggregation references it
    assert "orders_transform as (" in sql  # ...and it is actually defined


def test_single_entity_transform_cte_is_defined():
    """Bug #2 (single-entity): the target's transform CTE must exist."""
    sql = _render(
        {
            "target": "u",
            "max_depth": 1,
            "intervals": [],
            "aggregations": ["count"],
            "transformations": ["identity"],
            "entities": [
                {
                    "alias": "u",
                    "table": "u",
                    "id": "uid",
                    "temporal_ix": "ts",
                    "variables": {"v": {"type": "numeric"}},
                }
            ],
        }
    )
    assert "u_transform as (" in sql


def test_transform_cte_references_aggregates_by_name_not_definition():
    """Bug #1: passthrough aggregates render by name, not re-rendered definition."""
    sql = _render(_parent_child_config(max_depth=2))
    transform = _segment(sql, "customers_transform as (", "from customers_synth")

    # Passthrough by name (the fix).
    assert '"MEAN(orders.amount)" as "MEAN(orders.amount)"' in transform

    # The aggregate definition must NOT be re-rendered against the synth CTE
    # (which has no `amount` column). This is the exact invalid SQL of bug #1.
    assert "avg( amount )" not in transform
    assert "sum( amount )" not in transform


def _asof_config() -> dict:
    return {
        "target": "patients",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["mean"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "patients",
                "table": "patients",
                "id": "patient_id",
                "temporal_ix": "registered_at",
                "variables": {"age": {"type": "numeric"}},
            },
            {
                "alias": "care_plans",
                "table": "care_plans",
                "id": "plan_id",
                "temporal_ix": "effective_at",
                "variables": {
                    "patient_id": {"type": "index"},
                    "risk_score": {"type": "numeric"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "care_plans", "key": "patient_id"},
                "child": {"entity": "patients", "key": "patient_id"},
                "temporal": {"mode": "as_of", "grace": "P14D"},
            }
        ],
    }


def test_asof_join_key_is_projected_through_source_transform():
    """Bug #4: the as-of join key (a `type: index` variable) must be projected."""
    sql = _render(_asof_config())
    transform = _segment(sql, "care_plans_transform as (", "from care_plans_synth")
    assert "patient_id" in transform


def test_asof_grace_clause_is_dialect_safe():
    """Bug #5: grace bound is `source >= target - interval`, valid for date cols."""
    flat = " ".join(_render(_asof_config()).split())
    assert "- interval 'P14D'" in flat
    assert "care_plans_transform.effective_at >= patients.registered_at" in flat
    # The old `date - date <= interval` form (invalid for date columns) is gone.
    assert "registered_at - care_plans_transform.effective_at" not in flat


def test_identifier_columns_are_not_duplicated():
    """Bug #6: a PK that doubles as an FK is projected once, not twice."""
    sql = _render(_asof_config())
    synth = _segment(sql, "patients_synth as (", "from patients")
    assert synth.count("patients.patient_id") == 1


def test_interval_windows_cast_event_column_to_date():
    """Bug #7: ``daterange @> <event_col>`` is invalid when the temporal_ix is
    a timestamp (``operator does not exist: daterange @> timestamp``); every
    interval window must cast the event column to date. Found by the realistic
    DonorsChoose dataset, whose ``donation_timestamp`` is a timestamp."""
    config = _parent_child_config(max_depth=2)
    config["intervals"] = ["P1M"]
    config["aggregations"] = ["sum", "count", "recency", "gap_mean", "kl_drift"]
    config["entities"][1]["variables"]["status"] = {"type": "categorical"}
    sql = _render(config)

    contained = re.findall(r"@>\s*([\w.\"]+(?:::\w+)?)", sql)
    assert contained, "no interval windows were generated"
    uncast = [token for token in contained if not token.endswith("::date")]
    assert not uncast, f"interval windows without ::date cast: {uncast!r}"


def _peer_config(measures: bool = True) -> dict:
    """A facilities->inspections config with a peer_group on facilities."""
    config = _parent_child_config(max_depth=2)
    config["target"] = "facilities"
    config["aggregations"] = ["count"]
    facilities, inspections = config["entities"]
    facilities.update(
        {
            "alias": "facilities",
            "table": "facilities",
            "id": "license_no",
            "temporal_ix": "first_seen",
            "variables": {"facility_type": {"type": "categorical"}},
        }
    )
    if measures:
        facilities["variables"]["risk_score"] = {"type": "numeric"}
    peer_spec = {"by": "facility_type"}
    if measures:
        peer_spec["measures"] = ["risk_score"]
    facilities["peer_groups"] = [peer_spec]
    inspections.update(
        {
            "alias": "inspections",
            "table": "inspections",
            "id": "inspection_id",
            "temporal_ix": "inspection_date",
            "variables": {"results": {"type": "categorical"}},
        }
    )
    config["relationships"] = [
        {
            "parent": {"entity": "facilities", "key": "license_no"},
            "child": {"entity": "inspections", "key": "license_no"},
        }
    ]
    return config


def test_peer_group_cte_is_defined_and_joined_by_column():
    """The peer CTE exists and joins to the entity on the ``by`` column."""
    sql = _render(_peer_config())
    assert "peer_facility_type_for_facilities as (" in sql
    assert "group by e2.facility_type" in sql
    assert "g on g.grp = e.facility_type" in sql
    # Joined back to the entity by its id (synth-level join).
    assert "peer_facility_type_for_facilities.node_id = facilities.license_no" in sql


def test_peer_group_is_causally_bounded():
    """Peer membership and the peer child stream are both cut at the as-of date."""
    flat = " ".join(_render(_peer_config()).split())
    # Membership of the peer set is bounded.
    assert "where e2.first_seen <= aod.as_of_date" in flat
    # The shared per-peer event-count CTE is bounded on the child temporal_ix.
    assert "peer_evt_inspections_for_facilities as (" in flat
    assert "where c.inspection_date <= aod.as_of_date" in flat


def test_peer_group_is_leave_one_out():
    """Every peer aggregate divides by the leave-one-out count (n - in_grp)."""
    flat = " ".join(_render(_peer_config()).split())
    # Leave-one-out denominator guarded against the singleton group (n-1 == 0).
    assert "nullif((g.n - (case when e.first_seen <= aod.as_of_date" in flat
    # The pctile correlated subquery excludes the ego itself.
    assert "p.license_no <> e.license_no" in flat
    # No token-collision between the measure compare and the causal bound.
    assert "e.risk_scoreand" not in flat
    assert "e.risk_score and p.first_seen <= aod.as_of_date" in flat


def test_peer_group_emits_expected_families():
    """All six peer feature families are projected with stable names."""
    sql = _render(_peer_config())
    for family in (
        "PEER_GROUP_SIZE(facilities by facility_type)",
        "PEER_MEAN(facilities.risk_score by facility_type)",
        "EGO_MINUS_PEER_MEAN(facilities.risk_score by facility_type)",
        "PEER_ZSCORE(facilities.risk_score by facility_type)",
        "PEER_PCTILE(facilities.risk_score by facility_type)",
        "PEER_EVENT_RATE(facilities.inspections by facility_type)",
    ):
        assert family in sql, f"missing peer feature {family!r}"


def test_peer_group_without_measures_still_emits_size_and_rate():
    """With no numeric measures, only group-size and event-rate are emitted."""
    sql = _render(_peer_config(measures=False))
    assert "PEER_GROUP_SIZE(facilities by facility_type)" in sql
    assert "PEER_EVENT_RATE(facilities.inspections by facility_type)" in sql
    assert "PEER_MEAN(" not in sql
