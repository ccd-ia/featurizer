"""Distribution-shape aggregators: valid pure-aggregate SQL + default-set hygiene.

Stress-testing against triage-pg's donorschoose schema (many numeric columns)
executed the advanced aggregators for the first time and exposed a cluster of
never-run bugs:

* ``skewness`` / ``kurtosis`` used ``x - avg(x)`` (a bare, un-grouped column in a
  GROUP BY aggregate CTE) and the ``**`` operator PostgreSQL does not have. They
  are rewritten as pure-aggregate raw-moment formulas.
* ``z_score`` / ``min_max_scale`` are per-ROW normalizations, not reductions —
  their SQL references a bare column and cannot be an aggregation. The
  advanced-aggregator hardening pass DROPPED them entirely (they are redundant
  with the ``cross_entity_zscore`` / ``cross_entity_percentile`` transformers).
* ``mean_deviation`` (mean absolute deviation) nested aggregates
  (``sum(abs(x - avg(x)))``), which PostgreSQL forbids; it was rewritten as a
  two-pass ``SubqueryAggregator`` and RESTORED to the default set.
"""

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS
from featurizer.primitives.utils import get_aggregations, list_aggregations

_DROPPED = ["z_score", "min_max_scale"]


def _num_agg(name: str) -> str:
    parent = Entity(alias="p", table="p", id="pid")
    child = Entity(
        alias="c",
        table="c",
        id="cid",
        temporal_ix="ts",
        variables={"num": {"type": "numeric"}},
    )
    rel = Relationship(parent=parent, child=child, parent_key="pid", child_key="cid")
    num = next(f for f in child.features if f.name == "num")
    return get_aggregations([name])[name](
        parent, child, num, relationship=rel
    ).definition


def test_skewness_kurtosis_are_pure_aggregate_moments():
    for name in ("skewness", "kurtosis"):
        d = _num_agg(name)
        assert "power(" in d and "var_pop(num)" in d, (name, d)
        assert "**" not in d, (name, d)  # PostgreSQL has no ** operator
        assert "- avg(num))" not in d  # no bare un-grouped column term
        assert d.count("(") == d.count(")"), f"unbalanced: {name}: {d}"


def test_perrow_aggregators_are_dropped_entirely():
    # z_score / min_max_scale are per-row normalizations, invalid as reductions —
    # dropped from the registry (not merely default-excluded).
    registered = set(list_aggregations())
    for name in _DROPPED:
        assert name not in DEFAULT_AGGREGATIONS, name
        assert name not in registered, name


def test_mean_deviation_restored_as_subquery_reduction():
    from featurizer.primitives.aggregations import SubqueryAggregator

    assert "mean_deviation" in DEFAULT_AGGREGATIONS  # back in the default set
    agg = get_aggregations(["mean_deviation"])["mean_deviation"]
    assert isinstance(agg, SubqueryAggregator)  # two-pass, no nested aggregates
    d = _num_agg("mean_deviation")
    assert "SELECT AVG(m_sub.num)" in d and "AVG(ABS(sub.num - m.mean_val))" in d
