"""Distribution-shape aggregators: valid pure-aggregate SQL + default-set hygiene.

Stress-testing against triage-pg's donorschoose schema (many numeric columns)
executed the advanced aggregators for the first time and exposed a cluster of
never-run bugs:

* ``skewness`` / ``kurtosis`` used ``x - avg(x)`` (a bare, un-grouped column in a
  GROUP BY aggregate CTE) and the ``**`` operator PostgreSQL does not have. They
  are rewritten as pure-aggregate raw-moment formulas.
* ``z_score`` / ``min_max_scale`` are per-ROW normalizations, not reductions —
  their SQL references a bare column and cannot be an aggregation. They are
  removed from the default set (redundant with the ``cross_entity_zscore``
  transformer) but stay registered.
* ``mean_deviation`` (mean absolute deviation) nests aggregates
  (``sum(abs(x - avg(x)))``), which PostgreSQL forbids; removed from the default
  set pending a SubqueryAggregator rewrite.
"""

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS
from featurizer.primitives.utils import get_aggregations, list_aggregations

_EXCLUDED_FROM_DEFAULT = ["z_score", "min_max_scale", "mean_deviation"]


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


def test_perrow_and_nested_aggregators_left_the_default_set():
    for name in _EXCLUDED_FROM_DEFAULT:
        assert name not in DEFAULT_AGGREGATIONS, name


def test_excluded_aggregators_stay_registered():
    # Still discoverable / requestable by name (just not default-active).
    registered = set(list_aggregations())
    for name in _EXCLUDED_FROM_DEFAULT:
        assert name in registered, name
        assert name in get_aggregations([name])
