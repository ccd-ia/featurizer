"""Set-based pre-aggregation specs for the correlated-subquery aggregator tier.

An aggregator that returns a :class:`PreAggSpec` from ``_build_preagg`` is routed
by the planner into a *companion CTE* — one window pre-pass over the child stream
reduced by a plain ``GROUP BY``, replacing the per-target-row correlated subquery
that does not scale to full-cohort materialization (ADR-0009 / ADR-0010).

The division of labour:

- the **aggregator** owns the family-specific SQL — the ``prepass_sql`` (a full
  inner subquery over ``<child>_transform`` that derives per-row columns) and its
  own ``reduction`` (an aggregate over those derived columns, which becomes the
  feature's ``definition``);
- the **planner** owns the wrapping — it groups every family member sharing a
  ``(family_key, interval)`` into one CTE, projects the join key, applies the
  optional ``reduction_where``, and registers the CTE with the existing join /
  synth-source / sharding / materialization machinery.

This module imports only from :mod:`featurizer.boundary` and the stdlib, so it
introduces no import cycle with the primitives or the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ..boundary import causal_predicate, daterange_window

if TYPE_CHECKING:
    from .abstractions import Feature


@dataclass(frozen=True)
class PreAggSpec:
    """One family member's contribution to a companion pre-aggregation CTE.

    All members sharing a ``(family_key, interval)`` MUST carry byte-identical
    ``prepass_sql`` and ``reduction_where`` (the planner asserts this) — they
    differ only in ``reduction``, the per-member aggregate that reads the shared
    pre-pass. ``interval`` is part of the group key and the CTE name (one CTE per
    interval, so each applies its own window filter before the window function).
    """

    family_key: str
    interval: Optional[str]
    prepass_sql: str
    reduction: str
    reduction_where: str = ""


def causal_where(
    feature: "Feature", interval: Optional[str], *, column: Optional[str] = None
) -> str:
    """Standalone ``WHERE`` bounding the child stream to the as-of cut.

    The pre-pass reads the child stream directly (no correlation), so it needs
    the causal / interval bound as a *leading* ``WHERE`` clause rather than the
    ``and``-prefixed fragment the correlated form appends. Returns ``""`` when
    the entity has no ``temporal_ix`` (no time axis to bound), matching the
    correlated path's behaviour.
    """
    tix = getattr(feature.entity, "temporal_ix", None) if feature.entity else None
    if tix is None:
        return ""
    col = column if column is not None else tix.name
    if interval:
        return f"where {daterange_window(interval, column=col)}"
    return causal_predicate(col, prefix="where")
