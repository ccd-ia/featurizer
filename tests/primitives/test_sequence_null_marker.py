"""No sequence primitive reads a LAG's NULL as "no previous row" (ADR-0020).

``LAG(x)`` is NULL for the first row and for a row after a NULL value, so a
filter on it cannot tell the two apart (issue #86). ADR-0020 takes "no previous
row" from a row number computed in the same ``WINDOW`` as the LAG, which also
keeps the two in one sort (ADR-0018). This DB-free guard renders both paths of
every sequence primitive that reads a LAG of the value, the set-based pre-pass
with its reduction and the correlated fallback, and checks the SQL shape:

- no ``prev IS NOT NULL`` or ``lag_<i> IS NOT NULL``;
- a ``row_number()`` over the same named window as every LAG of the value.

``recurrence_interval`` is not here: its LAG reads the timestamp within one
value's partition, and a NULL gap there means only "first occurrence".
``tests/integration/test_null_state.py`` checks the values.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from featurizer.primitives.abstractions import Entity, Relationship
from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.utils import get_aggregations

# The primitives whose LAG reads the categorical value. MISMATCHES holds a
# strict xfail per primitive whose SQL breaks the rule; #86 found nine.
READS_A_LAG = (
    "longest_streak",
    "markov_conditional_entropy",
    "max_transition_prob",
    "ngram_2_freq",
    "ngram_3_freq",
    "rework_count",
    "sequence_entropy",
    "state_volatility",
    "time_in_current_state",
    "transition_matrix_summary",
)
MISMATCHES: dict[str, str] = {}

NULL_AS_FIRST_ROW = re.compile(r"\b(prev|lag_\d+)\s+is\s+not\s+null\b", re.I)
ROW_NUMBER_WINDOW = re.compile(r"\brow_number\(\)\s+over\s+(\w+)", re.I)
LAG_WINDOW = re.compile(r"\blag\([^)]*\)\s+over\s+(\w+|\()", re.I)

PARENT = Entity(alias="customers", table="analytics.customers", id="customer_id")
CHILD = Entity(
    alias="orders",
    table="analytics.orders",
    id="order_id",
    temporal_ix="ordered_at",
    variables={"category": {"type": "categorical"}},
)
RELATIONSHIP = Relationship(
    parent=PARENT, child=CHILD, parent_key="customer_id", child_key="customer_id"
)
FEATURE = next(f for f in CHILD.features if f.type == "categorical")


def _paths(name: str, interval: str | None) -> dict[str, str]:
    agg = get_aggregations([name])[name]
    assert isinstance(agg, SubqueryAggregator)
    spec = agg._build_preagg(FEATURE, CHILD, RELATIONSHIP, interval)
    assert spec is not None, f"{name} has no set-based pre-pass"
    prepass = " ".join(
        part
        for part in (spec.prepass_sql, spec.reduction, spec.reduction_where)
        if part
    )
    fallback = agg._build_subquery_expression(FEATURE, CHILD, RELATIONSHIP, interval)
    return {"pre-pass": prepass, "fallback": fallback}


def _cases() -> list[Any]:
    return [
        pytest.param(
            name,
            interval,
            id=f"{name}-{interval or 'all'}",
            marks=(
                [pytest.mark.xfail(strict=True, reason=MISMATCHES[name])]
                if name in MISMATCHES
                else []
            ),
        )
        for name in READS_A_LAG
        for interval in (None, "P30D")
    ]


@pytest.mark.parametrize(("name", "interval"), _cases())
def test_no_previous_row_is_a_row_number_from_the_lag_window(
    name: str, interval: str | None
) -> None:
    for path, sql in _paths(name, interval).items():
        found = NULL_AS_FIRST_ROW.search(sql)
        assert not found, f"{name} {path}: {found.group(0)!r} reads a LAG's NULL"
        lags = LAG_WINDOW.findall(sql)
        assert lags, f"{name} {path} reads no LAG"
        numbered = set(ROW_NUMBER_WINDOW.findall(sql))
        assert numbered, f"{name} {path} has no row_number() over a named window"
        unnumbered = [w for w in lags if w not in numbered]
        assert not unnumbered, (
            f"{name} {path}: a LAG over {unnumbered} has no row number in its window"
        )


def test_every_sequence_primitive_with_a_value_lag_is_guarded() -> None:
    """A categorical SubqueryAggregator whose SQL lags the value is listed."""
    from tests._sequence_reference import REFERENCE

    assert set(READS_A_LAG) == set(REFERENCE) - {"recurrence_interval"}
