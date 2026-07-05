"""DB-free SQL-shape regressions for relationships whose parent/child key
column names differ.

Every shipped example and fixture uses ``parent_key == child_key`` (usually
also == the parent id), which masked a family of bugs: the relationship-emitting
builders referenced the *other* side's key name on a table that doesn't have it.

- Aggregation (backward): the aggs CTE reads ``<child>_transform``, so it must
  project and GROUP BY ``child_key`` (the FK column the child actually carries);
  the synth join already compares ``cte.child_key = parent.parent_key``.
- Direct transfer (forward): the CTE reads ``<parent>_transform``, so it must
  project ``parent_key`` and the join must compare
  ``cte.parent_key = child.child_key``.
- As-of (lateral): the WHERE correlates
  ``<parent>_transform.parent_key = child.child_key`` — which requires the
  parent's transform to project ``parent_key`` even when it is not the parent's
  primary id.
"""

import tempfile

import yaml

from featurizer import Featurizer


def _render(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        path = handle.name
    return Featurizer(path).query


def _differing_keys_aggregation_config() -> dict:
    """customers.customer_id (parent) <- orders.buyer_id (child FK)."""
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": ["P1M"],
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
                "child": {"entity": "orders", "key": "buyer_id"},
            }
        ],
    }


def _segment(sql: str, start_marker: str, end_marker: str) -> str:
    start = sql.index(start_marker)
    end = sql.index(end_marker, start)
    return sql[start:end]


class TestAggregationDirection:
    def test_aggs_cte_projects_and_groups_by_child_key(self):
        sql = _render(_differing_keys_aggregation_config())
        cte = _segment(sql, "orders_aggs_for_customers as (", "customers_synth as (")
        assert "orders_transform.buyer_id" in cte
        assert "group by buyer_id" in cte
        # The parent-side key name must not be referenced on the child stream.
        assert "orders_transform.customer_id" not in cte
        assert "group by customer_id" not in cte

    def test_synth_join_compares_columns_both_sides_output(self):
        sql = _render(_differing_keys_aggregation_config())
        assert "orders_aggs_for_customers.buyer_id = customers.customer_id" in sql

    def test_child_key_is_carried_through_child_transform(self):
        sql = _render(_differing_keys_aggregation_config())
        synth = _segment(sql, "orders_synth as (", "orders_transform as (")
        assert "orders.buyer_id" in synth


class TestDirectTransferDirection:
    """Forward direction: the child target pulls the parent's variables."""

    def _config(self) -> dict:
        # Parent's join column (customer_ref) is NOT its primary id (cust_pk):
        # the CTE must project the relationship's parent_key, not the id.
        return {
            "target": "orders",
            "max_depth": 2,
            "intervals": [],
            "aggregations": ["sum"],
            "transformations": ["identity"],
            "entities": [
                {
                    "alias": "customers",
                    "table": "customers",
                    "id": "cust_pk",
                    "variables": {"score": {"type": "numeric"}},
                },
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
                    "parent": {"entity": "customers", "key": "customer_ref"},
                    "child": {"entity": "orders", "key": "buyer_id"},
                }
            ],
        }

    def test_direct_cte_projects_parent_key(self):
        sql = _render(self._config())
        cte = _segment(
            sql, "customers_direct_transfers_for_orders as (", "orders_synth as ("
        )
        assert "customer_ref" in cte

    def test_direct_join_pairs_parent_key_with_child_key(self):
        sql = _render(self._config())
        assert (
            "customers_direct_transfers_for_orders.customer_ref = orders.buyer_id"
            in sql
        )

    def test_parent_key_is_carried_through_parent_transform(self):
        sql = _render(self._config())
        synth = _segment(sql, "customers_synth as (", "customers_transform as (")
        assert "customers.customer_ref" in synth


class TestAsOfDirection:
    """As-of lateral: parent lookup pulled onto the timestamped child target."""

    def _config(self) -> dict:
        # care_plans' join column (patient_ref) is not its id (plan_id): the
        # lateral WHERE needs care_plans_transform.patient_ref projected.
        return {
            "target": "patients",
            "max_depth": 2,
            "intervals": [],
            "aggregations": ["sum"],
            "transformations": ["identity"],
            "entities": [
                {
                    "alias": "patients",
                    "table": "patients",
                    "id": "patient_id",
                    "temporal_ix": "admission_date",
                    "variables": {"age": {"type": "numeric"}},
                },
                {
                    "alias": "care_plans",
                    "table": "care_plans",
                    "id": "plan_id",
                    "temporal_ix": "plan_date",
                    "variables": {"cost": {"type": "numeric"}},
                },
            ],
            "relationships": [
                {
                    "parent": {"entity": "care_plans", "key": "patient_ref"},
                    "child": {"entity": "patients", "key": "patient_id"},
                    "temporal": {"mode": "as_of", "grace": "P7D"},
                }
            ],
        }

    def test_lateral_correlates_parent_key_to_child_key(self):
        sql = _render(self._config())
        assert "care_plans_transform.patient_ref = patients.patient_id" in sql

    def test_parent_key_projected_in_parent_synth(self):
        sql = _render(self._config())
        synth = _segment(sql, "care_plans_synth as (", "care_plans_transform as (")
        assert "care_plans.patient_ref" in synth


class TestEqualKeysUnchanged:
    """Guard: when parent_key == child_key == id (every shipped config), the
    rendered SQL keeps its existing shape."""

    def test_equal_keys_render_shape(self):
        config = _differing_keys_aggregation_config()
        config["relationships"][0]["child"]["key"] = "customer_id"
        sql = _render(config)
        cte = _segment(sql, "orders_aggs_for_customers as (", "customers_synth as (")
        assert "orders_transform.customer_id" in cte
        assert "group by customer_id" in cte
        assert "orders_aggs_for_customers.customer_id = customers.customer_id" in sql
