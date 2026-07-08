"""Tests for the M1c Markov sequence gap-fill aggregators.

Covers ``recurrence_interval``, ``markov_conditional_entropy``,
``max_transition_prob`` (sequence reductions over the transition matrix) and
``first_passage_time`` (predicate-driven).
"""

import pytest

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.utils import get_aggregations, list_aggregations

SEQUENCE = ["recurrence_interval", "markov_conditional_entropy", "max_transition_prob"]


def _setup(with_temporal=True):
    parent = Entity(alias="customers", table="c", id="customer_id")
    kwargs = dict(
        alias="events",
        table="e",
        id="event_id",
        variables={
            "status": {
                "type": "categorical",
                "predicates": {"target": "failed"},
            },
            "plain_cat": {"type": "categorical"},
            "amount": {"type": "numeric"},
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


def _full_sql(result):
    """Definition plus, for a migrated (pre-agg) aggregator, its shared pre-pass
    — where the ordering / transition-matrix / causal-bound SQL now lives."""
    prepass = result.preagg.prepass_sql if result.preagg is not None else ""
    return f"{result.definition} {prepass}"


@pytest.mark.parametrize("name", SEQUENCE + ["first_passage_time"])
def test_registered(name):
    assert name in list_aggregations()


@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_fires_and_is_causally_bounded(name):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(parent, child, _feat(child, "plain_cat"), relationship=rel)
    assert result is not None and result.definition is not None
    assert "<= aod.as_of_date" in _full_sql(result)


@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_interval_variant_uses_daterange(name):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    result = agg(
        parent, child, _feat(child, "plain_cat"), interval="P1W", relationship=rel
    )
    assert result is not None
    full = _full_sql(result)
    assert "daterange" in full and "P1W" in full
    # Bug #7 guard: the event column inside the window is date-cast.
    assert "@>" in full and "ts::date" in full


@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_rejects_numeric_features(name):
    parent, child, rel = _setup()
    agg = get_aggregations([name])[name]
    assert agg(parent, child, _feat(child, "amount"), relationship=rel) is None


@pytest.mark.parametrize("name", SEQUENCE)
def test_sequence_requires_temporal_ix(name):
    parent, child, rel = _setup(with_temporal=False)
    agg = get_aggregations([name])[name]
    assert agg(parent, child, _feat(child, "plain_cat"), relationship=rel) is None


@pytest.mark.parametrize("name", SEQUENCE + ["first_passage_time"])
def test_requires_relationship(name):
    parent, child, _ = _setup()
    agg = get_aggregations([name])[name]
    feat = "status" if name == "first_passage_time" else "plain_cat"
    assert agg(parent, child, _feat(child, feat)) is None


def test_recurrence_interval_partitions_by_state():
    parent, child, rel = _setup()
    agg = get_aggregations(["recurrence_interval"])["recurrence_interval"]
    result = agg(parent, child, _feat(child, "plain_cat"), relationship=rel)
    # Set-based (ADR-0010): the same-state LAG partitions by (child key, value).
    full = _full_sql(result).lower()
    assert (
        "partition by events_transform.customer_id, events_transform.plain_cat" in full
    )
    assert "order by events_transform.ts" in full
    assert "avg(gap)" in result.definition.lower()


def test_markov_conditional_entropy_uses_conditional_probability():
    parent, child, rel = _setup()
    agg = get_aggregations(["markov_conditional_entropy"])["markov_conditional_entropy"]
    result = agg(parent, child, _feat(child, "plain_cat"), relationship=rel)
    # joint weight p(i,j) times log of the *conditional* p(j|i)
    full = _full_sql(result)
    assert "freq::float / total" in full
    assert "LN(freq::float / row_total)" in full
    # row-conditional total is the per-(key, prev) partition in the pre-pass
    assert "partition by customer_id, prev" in full.lower()


def test_max_transition_prob_reduces_conditional_matrix():
    parent, child, rel = _setup()
    agg = get_aggregations(["max_transition_prob"])["max_transition_prob"]
    result = agg(parent, child, _feat(child, "plain_cat"), relationship=rel)
    assert "MAX(freq::float / row_total)" in result.definition


def test_first_passage_time_fires_with_target_predicate():
    parent, child, rel = _setup()
    agg = get_aggregations(["first_passage_time"])["first_passage_time"]
    result = agg(parent, child, _feat(child, "status"), relationship=rel)
    assert result is not None
    assert "FILTER (WHERE sub.status = 'failed')" in result.definition
    assert "MIN(sub.ts)" in result.definition
    assert "<= aod.as_of_date" in result.definition


def test_first_passage_time_skips_without_target():
    parent, child, rel = _setup()
    agg = get_aggregations(["first_passage_time"])["first_passage_time"]
    assert agg(parent, child, _feat(child, "plain_cat"), relationship=rel) is None


def test_first_passage_time_requires_temporal_ix():
    parent, child, rel = _setup(with_temporal=False)
    agg = get_aggregations(["first_passage_time"])["first_passage_time"]
    assert agg(parent, child, _feat(child, "status"), relationship=rel) is None
