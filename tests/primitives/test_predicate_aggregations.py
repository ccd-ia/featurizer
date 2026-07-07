"""Tests for predicate-driven aggregators (Phase 6)."""

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations, list_aggregations


def _setup(with_temporal=True):
    parent = Entity(alias="customers", table="c", id="customer_id")
    kwargs = dict(
        alias="events",
        table="e",
        id="event_id",
        variables={
            "event_type": {
                "type": "categorical",
                "predicates": {"a": "order", "b": "deliver", "terminal": "cancel"},
            },
            "plain_cat": {"type": "categorical"},
        },
    )
    if with_temporal:
        kwargs["temporal_ix"] = "ts"
    child = Entity(**kwargs)
    rel = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    return parent, child, rel


def _feat(child, name):
    return next(f for f in child.features if f.name == name)


# --------------------------------------------------------------------------- #
# Predicate carriage (abstractions)
# --------------------------------------------------------------------------- #


def test_variable_carries_predicates():
    _, child, _ = _setup()
    assert _feat(child, "event_type").predicates == {
        "a": "order",
        "b": "deliver",
        "terminal": "cancel",
    }


def test_variable_without_predicates_is_empty():
    _, child, _ = _setup()
    assert _feat(child, "plain_cat").predicates == {}


# --------------------------------------------------------------------------- #
# right_censoring_indicator
# --------------------------------------------------------------------------- #


def test_right_censoring_registered():
    assert "right_censoring_indicator" in list_aggregations()


def test_right_censoring_fires_with_terminal_predicate():
    parent, child, rel = _setup()
    agg = get_aggregations(["right_censoring_indicator"])["right_censoring_indicator"]
    result = agg(parent, child, _feat(child, "event_type"), relationship=rel)
    assert result is not None
    assert "FILTER (WHERE sub.event_type = 'cancel')" in result.definition
    assert "= 0)::int" in result.definition
    assert "<= aod.as_of_date" in result.definition  # causal bound


def test_right_censoring_skips_without_terminal():
    parent, child, rel = _setup()
    agg = get_aggregations(["right_censoring_indicator"])["right_censoring_indicator"]
    assert agg(parent, child, _feat(child, "plain_cat"), relationship=rel) is None


# --------------------------------------------------------------------------- #
# cross_type_latency
# --------------------------------------------------------------------------- #


def test_cross_type_latency_registered():
    assert "cross_type_latency" in list_aggregations()


def test_cross_type_latency_fires_with_a_b_predicates():
    parent, child, rel = _setup()
    agg = get_aggregations(["cross_type_latency"])["cross_type_latency"]
    result = agg(parent, child, _feat(child, "event_type"), relationship=rel)
    assert result is not None
    assert "a.event_type = 'order'" in result.definition
    assert "b.event_type = 'deliver'" in result.definition
    # Latency in days, epoch-extracted per side so it is numeric for both date
    # and timestamp columns (raw ``MIN(b.ts) - a.ts`` breaks on date columns).
    assert (
        "(EXTRACT(EPOCH FROM MIN(b.ts)) - EXTRACT(EPOCH FROM a.ts)) / 86400.0"
        in result.definition
    )
    # both sides causally bounded
    assert "a.ts <= aod.as_of_date" in result.definition
    assert "b.ts <= aod.as_of_date" in result.definition


def test_cross_type_latency_skips_without_predicates():
    parent, child, rel = _setup()
    agg = get_aggregations(["cross_type_latency"])["cross_type_latency"]
    assert agg(parent, child, _feat(child, "plain_cat"), relationship=rel) is None


def test_cross_type_latency_requires_temporal_ix():
    parent, child, rel = _setup(with_temporal=False)
    agg = get_aggregations(["cross_type_latency"])["cross_type_latency"]
    assert agg(parent, child, _feat(child, "event_type"), relationship=rel) is None


def test_cross_type_latency_interval_uses_daterange():
    parent, child, rel = _setup()
    agg = get_aggregations(["cross_type_latency"])["cross_type_latency"]
    result = agg(
        parent, child, _feat(child, "event_type"), interval="P1W", relationship=rel
    )
    assert result.definition.count("daterange") == 2  # a and b windows
    assert "P1W" in result.definition
