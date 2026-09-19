# coding: utf-8

"""The single source of truth for the as-of (point-in-time) boundary.

Point-in-time correctness is featurizer's cardinal rule: a feature computed for
an ``as_of_date`` may read only data knowable *at or before* that date. Every
CTE the planner and the aggregation primitives emit therefore cuts on the
entity's temporal column against ``aod.as_of_date``. Historically that cut was
hand-spelled as ``<= aod.as_of_date`` (and ``daterange(..., '[]')`` for interval
windows) in ~8 string templates — same invariant, many spellings, exactly the
shape where a future edit flips one site and not the others.

This module defines the boundary *once*:

* :data:`DEFAULT_BOUNDARY` / the ``inclusive`` vs ``exclusive`` modes,
* :func:`causal_predicate` — the scalar ``<col> <op> aod.as_of_date`` fragment,
* :func:`daterange_bound` — the matching ``daterange(..., <bound>)`` literal for
  interval windows,
* :func:`cohort_predicate` — the optional cut that pairs each as-of date with
  its own entities, and :func:`as_of_dates_source`, the relation ``aod``
  ranges over (issue #10).

Mode plumbing without circular imports
--------------------------------------
The aggregation primitives are module-level singletons created at import time
and shared across every :class:`~featurizer.Featurizer` instance, so the mode
cannot be stored on them per run. Instead the active boundary is held in a
:class:`contextvars.ContextVar` that the planner sets for the duration of a
``plan()`` call (see :func:`use_boundary`). The helper functions read the
current value, so both ``planner.py`` and ``primitives/aggregations.py`` get the
same operator without threading a parameter through every primitive signature.

This module imports nothing from the rest of the package, so it is safe to
import from both ``planner`` and ``primitives.aggregations`` (the latter is a
dependency of the former) with no risk of an import cycle.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Literal

# ``inclusive`` keeps the pre-existing behaviour: an event dated exactly on the
# as_of_date is knowable. ``exclusive`` treats such an event as not-yet-knowable
# (it must be strictly before the cutoff).
AsOfBoundary = Literal["inclusive", "exclusive"]

VALID_BOUNDARIES: tuple[AsOfBoundary, ...] = ("inclusive", "exclusive")
DEFAULT_BOUNDARY: AsOfBoundary = "inclusive"

# Scalar comparison operator per mode.
_OPERATOR: dict[AsOfBoundary, str] = {"inclusive": "<=", "exclusive": "<"}

# Upper-bound inclusivity flag for a PostgreSQL ``daterange``. The lower bound is
# always closed (``[``); only the upper bound tracks the boundary mode: closed
# (``]``) when inclusive, open (``)``) when exclusive.
_RANGE_BOUND: dict[AsOfBoundary, str] = {"inclusive": "[]", "exclusive": "[)"}


_active_boundary: ContextVar[AsOfBoundary] = ContextVar(
    "featurizer_as_of_boundary", default=DEFAULT_BOUNDARY
)


def current_boundary() -> AsOfBoundary:
    """Return the boundary mode in effect for the current render."""
    return _active_boundary.get()


@contextlib.contextmanager
def use_boundary(boundary: AsOfBoundary) -> Iterator[None]:
    """Bind ``boundary`` as the active mode for the duration of the block.

    The planner wraps its ``plan()`` traversal in this so that every primitive
    invoked underneath — including the shared aggregator singletons — reads the
    same operator. Restores the previous value on exit (so nested or concurrent
    renders do not leak into one another).
    """
    if boundary not in _RANGE_BOUND:
        raise ValueError(
            f"Unknown as_of_boundary {boundary!r}; expected one of "
            f"{', '.join(VALID_BOUNDARIES)}."
        )
    token = _active_boundary.set(boundary)
    try:
        yield
    finally:
        _active_boundary.reset(token)


def operator(boundary: AsOfBoundary | None = None) -> str:
    """Return the scalar comparison operator (``<=`` or ``<``)."""
    return _OPERATOR[boundary or current_boundary()]


def causal_predicate(
    col: str, *, prefix: str = "", boundary: AsOfBoundary | None = None
) -> str:
    """Render the canonical ``<col> <op> aod.as_of_date`` causal cut.

    Args:
        col: The (already alias-qualified) temporal column, e.g. ``c.ordered_at``.
        prefix: Optional leading keyword such as ``"where"`` or ``"and"``. When
            given the result is ``" <prefix> <col> <op> aod.as_of_date"`` (with a
            leading space, matching the surrounding builders); when empty the
            result is the bare ``"<col> <op> aod.as_of_date"``.
        boundary: Override the active mode (defaults to :func:`current_boundary`).

    Every builder writes the column on the *left* and ``aod.as_of_date`` on the
    right, so the invariant reads identically everywhere.
    """
    op = operator(boundary)
    predicate = f"{col} {op} aod.as_of_date"
    if prefix:
        return f" {prefix} {predicate}"
    return predicate


#: The caller's table of as-of dates, bound to ``aod`` in every rendered query.
AS_OF_DATES_TABLE = "as_of_dates"


def cohort_predicate(target_id: str, id_column: str, *, prefix: str = "") -> str:
    """Render the paired-cohort cut for one as-of date (issue #10).

    By default ``as_of_dates`` holds dates only and every target row is emitted
    under every date. When the config declares ``as_of_dates: {id_column: …}``
    the table holds ``(as_of_date, id)`` pairs, and this predicate keeps the
    target rows paired with the date ``aod`` is currently on::

        <target id> in (select _cohort.<id_column> from as_of_dates _cohort
                        where _cohort.as_of_date = aod.as_of_date)

    It sits beside :func:`causal_predicate` because it has the same reach:
    ``aod`` is in scope inside every CTE of the lateral, so the cut can be
    applied where the target is *read*, before anything is computed for a row
    that would be thrown away.

    A semi-join, not ``<target id> = aod.<id_column>`` with ``aod`` ranging over
    the pairs. That shape evaluates the lateral once per *pair*, and every CTE
    PostgreSQL cannot inline is recomputed each time: measured on a 65-aggregation
    config it had not finished after 1,520 s where the dense query took 72 s.
    This one evaluates once per *date*, as the dense query does, so its cost
    stays close to dense: measured 4% and 8% slower on that same config, and
    2.8x faster on a 147-feature one. It also makes a duplicated pair harmless.

    Args:
        target_id: The already-quoted target id column, qualified as the
            surrounding query needs it.
        id_column: The already-quoted column of ``as_of_dates`` holding the id.
        prefix: Optional leading keyword, as in :func:`causal_predicate`.
    """
    predicate = (
        f"{target_id} in (select _cohort.{id_column} from {AS_OF_DATES_TABLE} "
        f"_cohort where _cohort.as_of_date = aod.as_of_date)"
    )
    if prefix:
        return f" {prefix} {predicate}"
    return predicate


def as_of_dates_source(*, paired: bool) -> str:
    """The relation ``aod`` ranges over: one row per as-of date.

    Unpaired, that is the caller's table, by its bare name as it always was.
    Paired, the table repeats each date once per id, so ``aod`` ranges over its
    distinct dates. Every statement that binds ``aod`` goes through here — the
    query's outer spine and the temp-table preamble alike. The preamble is the
    one that silently breaks otherwise: it cross-joins the dates and groups by
    them, so a repeated date multiplies every ``count`` and ``sum``.
    """
    if paired:
        return f"(select distinct as_of_date from {AS_OF_DATES_TABLE})"
    return AS_OF_DATES_TABLE


def daterange_bound(boundary: AsOfBoundary | None = None) -> str:
    """Return the ``daterange`` upper-bound literal (``'[]'`` or ``'[)'``)."""
    return _RANGE_BOUND[boundary or current_boundary()]


def daterange_window(
    interval: str,
    *,
    column: str | None = None,
    boundary: AsOfBoundary | None = None,
) -> str:
    """Render an interval ``daterange`` window anchored at ``aod.as_of_date``.

    With ``column`` it returns the full containment test
    ``daterange(...) @> <column>::date``; without it, just the ``daterange(...)``
    expression. The upper-bound inclusivity follows the active boundary mode.
    """
    bound = daterange_bound(boundary)
    window = (
        f"daterange((aod.as_of_date - interval '{interval}')::date, "
        f"aod.as_of_date::date, '{bound}')"
    )
    if column is not None:
        return f"{window} @> {column}::date"
    return window
