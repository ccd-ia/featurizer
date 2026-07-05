"""Parallel relationships, diamond topologies, and the naming contract.

Before v0.5.0 the traversal guard (``if child in self._path: continue``)
silently dropped every relationship after the first that reached an
already-built entity: the second of two parallel customers->orders
relationships vanished (5 of 10 features), and in a diamond a<-b<-d / a<-c<-d
the d-aggregations never flowed through c at all.

Now: entities build once, every relationship is consumed, parallel
relationships require distinct ``name:`` (validation error otherwise), and the
naming alias keeps unambiguous configs byte-identical.
"""

import tempfile

import yaml

from featurizer import Featurizer, validate_config


def _write(config: dict) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return handle.name


def _render(config: dict, **kwargs) -> str:
    return Featurizer(_write(config), **kwargs).query


def _parallel_config(named: bool = True) -> dict:
    rels = [
        {
            "parent": {"entity": "customers", "key": "customer_id"},
            "child": {"entity": "orders", "key": "buyer_id"},
        },
        {
            "parent": {"entity": "customers", "key": "customer_id"},
            "child": {"entity": "orders", "key": "seller_id"},
        },
    ]
    if named:
        rels[0]["name"] = "purchases"
        rels[1]["name"] = "sales"
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
        "relationships": rels,
    }


class TestValidationAmbiguityGuard:
    def test_unnamed_parallel_relationships_error(self):
        result = validate_config(_write(_parallel_config(named=False)))
        assert not result.is_valid
        assert any("parallel relationships" in e.message.lower() for e in result.errors)

    def test_named_parallel_relationships_validate(self):
        result = validate_config(_write(_parallel_config(named=True)))
        assert result.is_valid

    def test_duplicate_relationship_names_error(self):
        config = _parallel_config(named=True)
        config["relationships"][1]["name"] = "purchases"
        result = validate_config(_write(config))
        assert not result.is_valid
        assert any("Duplicate relationship name" in e.message for e in result.errors)

    def test_name_colliding_with_entity_alias_errors(self):
        config = _parallel_config(named=True)
        config["relationships"][0]["name"] = "orders"
        result = validate_config(_write(config))
        assert not result.is_valid
        assert any("collides with an entity alias" in e.message for e in result.errors)

    def test_non_identifier_name_errors(self):
        config = _parallel_config(named=True)
        config["relationships"][0]["name"] = "as-buyer"
        result = validate_config(_write(config))
        assert not result.is_valid
        assert any("not a valid identifier" in e.message for e in result.errors)


class TestParallelRelationshipsSynthesis:
    def test_both_legs_produce_features(self):
        f = Featurizer(_write(_parallel_config(named=True)))
        names = [x.name for x in f._plan.target_output_features]
        purchases = [n for n in names if "purchases.amount" in n]
        sales = [n for n in names if "sales.amount" in n]
        # sum/mean x {no-interval, P1M} per leg.
        assert len(purchases) == 4
        assert len(sales) == 4
        assert len(names) == len(set(names)), "duplicate output feature names"

    def test_two_distinct_ctes_and_joins(self):
        sql = _render(_parallel_config(named=True))
        assert "purchases_aggs_for_customers as (" in sql
        assert "sales_aggs_for_customers as (" in sql
        assert "purchases_aggs_for_customers.buyer_id = customers.customer_id" in sql
        assert "sales_aggs_for_customers.seller_id = customers.customer_id" in sql

    def test_each_cte_groups_by_its_own_key(self):
        sql = _render(_parallel_config(named=True))
        purchases = sql[
            sql.index("purchases_aggs_for_customers as (") : sql.index(
                "sales_aggs_for_customers as ("
            )
        ]
        assert "group by buyer_id" in purchases
        assert "seller_id" not in purchases


class TestDiamondTopology:
    """a <- b <- d and a <- c <- d: d's aggregations must flow up BOTH paths."""

    def _config(self) -> dict:
        def entity(alias: str, temporal: str, var: str) -> dict:
            return {
                "alias": alias,
                "table": alias,
                "id": f"{alias}_id",
                "temporal_ix": temporal,
                "variables": {var: {"type": "numeric"}},
            }

        return {
            "target": "a",
            "max_depth": 3,
            "intervals": [],
            "aggregations": ["sum"],
            "transformations": ["identity"],
            "entities": [
                entity("a", "a_ts", "xa"),
                entity("b", "b_ts", "xb"),
                entity("c", "c_ts", "xc"),
                entity("d", "d_ts", "xd"),
            ],
            "relationships": [
                {
                    "parent": {"entity": "a", "key": "a_id"},
                    "child": {"entity": "b", "key": "a_id"},
                },
                {
                    "parent": {"entity": "a", "key": "a_id"},
                    "child": {"entity": "c", "key": "a_id"},
                },
                {
                    "parent": {"entity": "b", "key": "b_id"},
                    "child": {"entity": "d", "key": "b_id"},
                },
                {
                    "parent": {"entity": "c", "key": "c_id"},
                    "child": {"entity": "d", "key": "c_id"},
                },
            ],
        }

    def test_second_path_is_not_dropped(self):
        f = Featurizer(_write(self._config()))
        names = [x.name for x in f._plan.target_output_features]
        via_b = [n for n in names if "b.SUM(d.xd)" in n]
        via_c = [n for n in names if "c.SUM(d.xd)" in n]
        assert via_b, "d-aggregations missing via b"
        assert via_c, "d-aggregations missing via c (the dropped diamond path)"
        assert len(names) == len(set(names))

    def test_both_agg_ctes_emitted(self):
        sql = _render(self._config())
        assert "d_aggs_for_b as (" in sql
        assert "d_aggs_for_c as (" in sql


class TestSelfRelationship:
    """Baseline: an entity related to itself is a true cycle for the planner —
    the relationship is skipped (the entity's CTEs are mid-build when the
    backward edge is examined) and the plan still renders. Graph features via
    ``edge:`` tables remain the supported path for self-referential structure."""

    def test_self_relationship_renders_without_crash(self):
        config = {
            "target": "employees",
            "max_depth": 2,
            "intervals": [],
            "aggregations": ["sum"],
            "transformations": ["identity"],
            "entities": [
                {
                    "alias": "employees",
                    "table": "employees",
                    "id": "employee_id",
                    "temporal_ix": "hired_at",
                    "variables": {"salary": {"type": "numeric"}},
                }
            ],
            "relationships": [
                {
                    "parent": {"entity": "employees", "key": "employee_id"},
                    "child": {"entity": "employees", "key": "manager_id"},
                }
            ],
        }
        sql = _render(config, validate=False)
        assert "employees_transform" in sql
        # The cyclic aggregation is skipped, not half-emitted.
        assert "employees_aggs_for_employees" not in sql


class TestNameStability:
    """The ADR-0007 contract: unambiguous configs keep byte-identical names."""

    # Frozen v0.4.2 output for the canonical single-relationship config below
    # (sum/mean over amount, P1M interval, identity transform). If this list
    # ever changes, downstream Parquet columns and persisted feature-importance
    # keys break — the design is wrong, not this test.
    V042_NAMES = [
        '"MEAN(orders.amount)"',
        '"MEAN(orders.amount|interval=P1M)"',
        '"SUM(orders.amount)"',
        '"SUM(orders.amount|interval=P1M)"',
        "score",
    ]

    def test_unambiguous_config_names_unchanged(self):
        config = {
            "target": "customers",
            "max_depth": 2,
            "intervals": ["P1M"],
            "aggregations": ["sum", "mean"],
            "transformations": ["identity"],
            "entities": [
                {
                    "alias": "customers",
                    "table": "customers",
                    "id": "customer_id",
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
                    "parent": {"entity": "customers", "key": "customer_id"},
                    "child": {"entity": "orders", "key": "customer_id"},
                }
            ],
        }
        f = Featurizer(_write(config))
        names = sorted(x.name for x in f._plan.target_output_features)
        assert names == sorted(self.V042_NAMES)

    def test_unambiguous_cte_names_unchanged(self):
        sql = _render(_parallel_config(named=True))
        # Named relationships opt in to new CTE names; the entity-pair form
        # must NOT appear for them.
        assert "orders_aggs_for_customers" not in sql
