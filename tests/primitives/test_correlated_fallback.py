"""Every correlated-subquery fallback renders, and renders causally (issue #83).

ADR-0010 moved most ``SubqueryAggregator`` families onto a set-based pre-pass,
and the planner always takes it when ``_build_preagg`` returns a spec. The
correlated ``_build_subquery_expression`` each of them keeps is then dead code
in every rendered query, so nothing renders it, and nothing noticed when
``sequence_entropy``'s called ``_timeline`` with one argument too many: the
fallback raised ``TypeError`` the first time a test turned the pre-pass off.

This sweep calls the fallback directly, for every registered aggregation that
is a ``SubqueryAggregator`` (from the registry, not a hand list), on a dated
child, over the whole history and over an interval. A case the aggregator's
own ``__call__`` declines (the drift pair has no whole-history feature) is
not a case: the planner never asks for it.

The fallback must return SQL, and every read of the child in it must carry the
backward causal bound: ``<= aod.as_of_date`` over the whole history, or a
``daterange`` window over the requested interval whose upper bound is the
as-of date (or, for the drift pair's baseline, the as-of date minus the
interval).

The values the fallback computes are compared with the pre-pass's in
``tests/integration/test_correlated_fallback_parity.py``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest

from featurizer.primitives.abstractions import Entity, Feature, Relationship
from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.utils import get_aggregations, list_aggregations
from tests.test_aggregation_quoting import PREDICATES

INTERVALS = (None, "P30D")
WHOLE_HISTORY_BOUND = re.compile(r"<= aod\.as_of_date\b")


def _entities() -> tuple[Entity, Entity, Relationship]:
    """The parent / dated child of ``tests/primitives/test_aggregations.py``,
    plus what the spatial and predicate-driven aggregators need: a
    ``spatial_ix`` and predicates on the categorical."""
    parent = Entity(alias="customers", table="analytics.customers", id="customer_id")
    child = Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        temporal_ix="ordered_at",
        spatial_ix={"lat": "lat", "lon": "lon"},
        variables={
            "amount": {"type": "numeric"},
            "category": {"type": "categorical", "predicates": PREDICATES},
        },
    )
    relationship = Relationship(
        parent=parent, child=child, parent_key="customer_id", child_key="customer_id"
    )
    return parent, child, relationship


PARENT, CHILD, RELATIONSHIP = _entities()
SUBQUERY_AGGREGATORS: dict[str, SubqueryAggregator] = {
    name: agg
    for name, agg in sorted(get_aggregations(list_aggregations()).items())
    if isinstance(agg, SubqueryAggregator)
}


def _input(input_type: str) -> Feature | None:
    """The child column an aggregator of ``input_type`` reduces.

    An ``index`` aggregation (the gap family, ``burstiness``, the spatial
    family) reads the temporal index, which is the feature the planner hands it.
    """
    if input_type == "index":
        return CHILD.temporal_ix
    return next((f for f in CHILD.features if f.type == input_type), None)


def _cases() -> Iterator[Any]:
    for name, agg in SUBQUERY_AGGREGATORS.items():
        for input_type in agg.input_types:
            feature = _input(input_type)
            for interval in INTERVALS:
                if feature is not None and (
                    agg(
                        PARENT,
                        CHILD,
                        feature,
                        interval=interval,
                        relationship=RELATIONSHIP,
                    )
                    is None
                ):
                    continue
                yield pytest.param(
                    name,
                    input_type,
                    interval,
                    id=f"{name}-{input_type}-{interval or 'all'}",
                )


def _window_bound(interval: str) -> re.Pattern[str]:
    """A ``daterange`` over ``interval`` whose upper bound is at or before the
    as-of date: ``aod.as_of_date``, or ``aod.as_of_date - interval '...'``."""
    literal = re.escape(f"interval '{interval}'")
    return re.compile(
        rf"daterange\(\(aod\.as_of_date - [^)]*{literal}\)\)?::date, "
        rf"\(?aod\.as_of_date\b"
    )


@pytest.mark.parametrize(("name", "input_type", "interval"), list(_cases()))
def test_the_correlated_fallback_renders_with_its_causal_bound(
    name: str, input_type: str, interval: str | None
) -> None:
    agg = SUBQUERY_AGGREGATORS[name]
    feature = _input(input_type)
    assert feature is not None, f"the child declares no {input_type} column"

    sql = agg._build_subquery_expression(feature, CHILD, RELATIONSHIP, interval)

    assert isinstance(sql, str) and sql.strip(), f"{name} rendered no SQL"
    reads = len(re.findall(rf"\bFROM {CHILD.alias}_transform \w+", sql, re.I))
    assert reads, f"{name}'s correlated subquery reads no child rows"
    bound = WHOLE_HISTORY_BOUND if interval is None else _window_bound(interval)
    bounds = len(bound.findall(sql))
    assert bounds >= reads, (
        f"{name} reads the child {reads} time(s) but bounds only {bounds} of "
        f"them by {'the as-of date' if interval is None else interval}"
    )


def test_every_registered_subquery_aggregator_has_a_case() -> None:
    """A ``SubqueryAggregator`` that declines every case on this child would be
    swept by nothing."""
    covered = {case.values[0] for case in _cases()}
    assert covered == set(SUBQUERY_AGGREGATORS)
