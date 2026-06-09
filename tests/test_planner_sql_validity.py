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
