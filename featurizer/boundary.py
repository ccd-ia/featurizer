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
  interval windows.

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
