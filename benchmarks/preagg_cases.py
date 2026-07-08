"""Pure (DB-free) specification of the golden-value cases.

Single source of truth shared by the golden-value *capture* script
(:mod:`benchmarks.capture_golden`) and the *verification* test
(``tests/integration/test_preagg_value_equality.py``) so the two can never test
different things. Imports only from ``featurizer`` and the standard library —
no database, no ``tests`` import.

A "case" pins down one measurable unit: a single subquery aggregator, run over
one fixture, with one temporal column type and one interval choice. Its golden
value is the full result set of ``[aggregator, count]`` over that fixture.
"""

from __future__ import annotations

import datetime as _dt
from typing import Any, Dict, List, Tuple

from featurizer.primitives.aggregations import SubqueryAggregator
from featurizer.primitives.utils import get_aggregations

# Aggregators that only fire under special config (predicates / spatial_ix /
# two-window drift) — kept in lockstep with the execution harness'
# ``_NEEDS_SPECIAL_CONFIG``. They stay on the correlated path (plan Phase 3e),
# so they are out of scope for golden capture / value-equality here.
NEEDS_SPECIAL_CONFIG = frozenset(
    {
        "all",
        "any",
        "bbox_area",
        "distance_travelled",
        "radius_of_gyration",
        "spatial_std",
        "first_passage_time",
        "cross_type_latency",
        "right_censoring_indicator",
        "kl_drift",
        "wasserstein_drift",
    }
)


def subquery_aggregators() -> List[str]:
    """Every registered aggregator whose instance is a ``SubqueryAggregator``.

    Introspected from the live registry — the authoritative count the plan's
    "~26" is validated against, not a hardcoded list.
    """
    registry = get_aggregations()
    return sorted(
        name for name, agg in registry.items() if isinstance(agg, SubqueryAggregator)
    )


def migratable_aggregators() -> List[str]:
    """Subquery aggregators eligible for the set-based rewrite.

    The full subquery set minus the special-config families that stay
    correlated. These are the aggregators whose values are frozen as golden.
    """
    return sorted(set(subquery_aggregators()) - NEEDS_SPECIAL_CONFIG)


# --- Fixtures -------------------------------------------------------------

# The 6-group edge fixture, copied verbatim from
# tests/integration/test_all_aggregators_execution.py so degenerate-group
# semantics (single row, constant, zero/negative, avg-zero, repeated category)
# are frozen identically. (child_key, ts, num, cat).
EDGE_ROWS: List[Tuple[int, str, float, str]] = [
    (1, "2023-01-01", 5.0, "a"),
    (2, "2023-02-01", 3.0, "x"),
    (2, "2023-02-02", 3.0, "x"),
    (3, "2023-03-01", 0.0, "p"),
    (3, "2023-03-05", -2.0, "q"),
    (4, "2023-04-01", 1.0, "m"),
    (4, "2023-04-03", 4.0, "n"),
    (4, "2023-04-06", 9.0, "m"),
    (4, "2023-04-10", 16.0, "n"),
    (5, "2023-05-01", -4.0, "s"),
    (5, "2023-05-02", 4.0, "s"),
    (6, "2023-06-01", 2.0, "z"),
    (6, "2023-06-02", 2.0, "z"),
    (6, "2023-06-03", 2.0, "z"),
]
EDGE_AS_OF = "2024-01-01"
EDGE_KEYS = list(range(1, 7))

_CATS = ("a", "b", "c", "d")


def dense_rows() -> List[Tuple[int, str, float, str]]:
    """A denser, fully-deterministic fixture: 50 keys, tie-free per key.

    Built with pure integer arithmetic (no RNG) so capture and verification
    reproduce byte-identical data. Within a key the day offsets are strictly
    increasing, so timestamps never tie — the one property the window-vs-
    correlated equivalence needs for sequence aggregators (see the plan's
    tie-ordering note).

    Each key gets an early spread (Jan..Oct) plus a December cluster of 4-6
    events, so a P1M window over the 2023-12-31 as-of captures a non-empty,
    multi-row tail — the interval cases must exercise real in-window values, not
    just NULLs (a mostly-empty window can't catch a boundary bug in the rewrite).
    """
    rows: List[Tuple[int, str, float, str]] = []
    base = _dt.date(2023, 1, 1)
    dec_first = (_dt.date(2023, 12, 1) - base).days  # 334

    def _emit(key: int, j: int, abs_day: int) -> None:
        ts = (base + _dt.timedelta(days=abs_day)).isoformat()
        num = float(((key * 7 + j * 13) % 97) - 20)  # spans negative..positive
        cat = _CATS[(key + j) % len(_CATS)]
        rows.append((key, ts, num, cat))

    for key in range(1, 51):
        j = 0
        abs_day = 0
        # Early spread: 5..8 events, gaps 15..34 days → stays Jan..Sep (< Dec 1).
        for _ in range(5 + (key % 4)):
            abs_day += 15 + ((key + j) % 20)
            _emit(key, j, abs_day)
            j += 1
        # December cluster: 4..6 events, gaps 2..4 days, starting Dec 1 → all
        # land in December (334 + 6*4 = 358 = Dec 25), so a P1M window over the
        # 2023-12-31 as-of captures a multi-row, non-degenerate tail.
        abs_day = dec_first
        for _ in range(4 + (key % 3)):
            abs_day += 2 + ((key + j) % 3)
            _emit(key, j, abs_day)
            j += 1
    return rows


DENSE_AS_OF = "2023-12-31"
DENSE_KEYS = list(range(1, 51))

FIXTURES: Dict[str, Dict[str, Any]] = {
    "edge": {"rows": EDGE_ROWS, "as_of": EDGE_AS_OF, "keys": EDGE_KEYS},
    "dense": {"rows": dense_rows(), "as_of": DENSE_AS_OF, "keys": DENSE_KEYS},
}


def config(agg: str, ts_type: str, interval: str | None) -> Dict[str, Any]:
    """A featurizer config selecting ``[agg, count]`` over the standard shape.

    Mirrors the execution harness' entity/relationship shape (parent ``p`` keyed
    on ``pid``; child ``c`` with temporal ``ts``, numeric ``num``, categorical
    ``cat``) so every non-special aggregator fires. ``count`` keeps the aggs CTE
    non-empty when ``agg`` produces a column for only one input type.
    """
    return {
        "target": "p",
        "max_depth": 2,
        "intervals": [interval] if interval else [],
        "aggregations": [agg, "count"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "p", "table": "p", "id": "pid", "variables": {}},
            {
                "alias": "c",
                "table": "c",
                "id": None,
                "temporal_ix": "ts",
                "variables": {
                    "num": {"type": "numeric"},
                    "cat": {"type": "categorical"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "p", "key": "pid"},
                "child": {"entity": "c", "key": "pid"},
            }
        ],
    }


# The case matrix. Edge fixture: no-interval only (its purpose is degenerate
# groups; intervals over it are mostly-empty and low-signal). Dense fixture:
# no-interval + P1M, both temporal types — the real value coverage.
_CASE_MATRIX: List[Tuple[str, str, str | None]] = [
    ("edge", "date", None),
    ("edge", "timestamp", None),
    ("dense", "date", None),
    ("dense", "timestamp", None),
    ("dense", "date", "P1M"),
    ("dense", "timestamp", "P1M"),
]


def case_id(agg: str, fixture: str, ts_type: str, interval: str | None) -> str:
    return f"{agg}|{fixture}|{ts_type}|{interval or 'all'}"


def cases() -> List[Dict[str, Any]]:
    """Every (aggregator × fixture × ts_type × interval) case to freeze.

    Returned as plain dicts so both the capture script and the test iterate one
    structure. The config is built lazily by the caller via :func:`config`.
    """
    out: List[Dict[str, Any]] = []
    for agg in migratable_aggregators():
        for fixture, ts_type, interval in _CASE_MATRIX:
            out.append(
                {
                    "id": case_id(agg, fixture, ts_type, interval),
                    "agg": agg,
                    "fixture": fixture,
                    "ts_type": ts_type,
                    "interval": interval,
                }
            )
    return out
