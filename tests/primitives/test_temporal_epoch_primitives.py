"""Temporal aggregators must difference epoch-days, not raw temporal values.

Regression guard for the wide-primitive SQL bugs found by stress-testing against
triage-pg's live datasets:

* ``event_rate`` / ``time_span`` / ``cross_type_latency`` emitted
  ``EXTRACT(EPOCH FROM max - min)`` — invalid on ``date`` columns
  (``date - date`` is an integer) → ``extract(unknown, integer)``.
* ``gap_*`` / ``burstiness`` differenced raw temporal values into a ``gap`` then
  aggregated it — ``STDDEV(interval)`` is undefined on ``timestamp`` columns.
* ``geometric_mean`` emitted malformed, unbalanced SQL using base-10 ``log``.

The fix extracts epoch seconds per side and divides by 86400 (days), which is
numeric for both ``date`` and ``timestamp`` columns.
"""

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations


def _graph():
    parent = Entity(alias="cust", table="c", id="cid")
    child = Entity(
        alias="ord",
        table="o",
        id="oid",
        temporal_ix="ts",
        variables={"amt": {"type": "numeric"}},
    )
    rel = Relationship(parent=parent, child=child, parent_key="cid", child_key="cid")
    return parent, child, rel


def _agg(name, feature_name):
    parent, child, rel = _graph()
    feature = (
        child.temporal_ix
        if feature_name == "ts"
        else next(f for f in child.features if f.name == feature_name)
    )
    agg = get_aggregations([name])[name]
    return agg(parent, child, feature, relationship=rel).definition


TEMPORAL_SPAN_AGGS = ["event_rate", "time_span"]
GAP_AGGS = ["gap_mean", "gap_stddev", "gap_min", "gap_max", "gap_cv", "burstiness"]


def test_span_aggs_extract_epoch_per_side_in_days():
    for name in TEMPORAL_SPAN_AGGS:
        d = _agg(name, "ts")
        assert "EXTRACT(EPOCH FROM max(ts))" in d, (name, d)
        assert "EXTRACT(EPOCH FROM min(ts))" in d, (name, d)
        assert "/ 86400.0" in d, (name, d)
        # The type-fragile raw form must be gone.
        assert "EXTRACT(EPOCH FROM max(ts) - min(ts))" not in d, (name, d)


def _gap_sql(name):
    """SQL carrying the gap computation: the shared pre-pass for a migrated gap
    aggregator (ADR-0010), else the correlated definition."""
    parent, child, rel = _graph()
    result = get_aggregations([name])[name](
        parent, child, child.temporal_ix, relationship=rel
    )
    if result.preagg is not None:
        return result.preagg.prepass_sql
    return result.definition


def test_gap_aggs_difference_epoch_days():
    # The gap family is set-based (ADR-0010): the epoch-day gap now lives in the
    # shared window pre-pass, differencing the partitioned LAG — still epoch-days
    # (never a raw interval), which is the property this regression guards.
    for name in GAP_AGGS:
        d = _gap_sql(name)
        assert "EXTRACT(EPOCH FROM ord_transform.ts)" in d, (name, d)
        assert "EXTRACT(EPOCH FROM LAG(ord_transform.ts)" in d, (name, d)
        assert "/ 86400.0" in d, (name, d)
        # No raw temporal subtraction left to yield an interval.
        assert "ord_transform.ts - LAG" not in d, (name, d)


def test_geometric_mean_is_positive_domain_natural_log():
    d = _agg("geometric_mean", "amt")
    assert "exp(avg(ln(" in d
    assert "min(amt) > 0" in d  # outer positive-domain guard
    assert "else null end" in d
    # Base-10 log and the old unbalanced/negative branch are gone.
    assert "log(" not in d
    assert d.count("(") == d.count(")"), f"unbalanced parens: {d}"
