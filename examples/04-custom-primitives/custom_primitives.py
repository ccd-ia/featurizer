"""Custom aggregation and transformation primitives for financial analytics.

These demonstrate the current primitive API and emit PostgreSQL-valid SQL:

- An **aggregation** subclasses ``Aggregator`` (or a richer base such as
  ``OrderedSetAggregator``) and overrides ``_build_aggregate_expression`` to
  return the SQL aggregate; the base wraps it into a ``Feature``. Interval
  variants pass ``interval`` so the expression can add the ``daterange`` filter.
- A **transformation** subclasses ``Transformer`` and overrides
  ``_build_transformer_call`` to return a scalar SQL expression over the
  current row; the base wraps it into a ``Feature``.

Register *instances* (not classes) via ``register_aggregation`` /
``register_transformer``, then select them by name in ``config.yaml``.
"""

from featurizer.primitives.aggregations import Aggregator, OrderedSetAggregator
from featurizer.primitives.transformations import Transformer
from featurizer.primitives.utils import register_aggregation, register_transformer

# ============================================================================
# Custom Aggregations
# ============================================================================


class Range(Aggregator):
    """Spread of a value: ``max(x) - min(x)``.

    PostgreSQL's ``FILTER`` clause attaches to a single aggregate, so for an
    interval window each of ``max`` and ``min`` is filtered independently.
    """

    def __init__(self):
        super().__init__(name="range")

    def _agg(self, fn, feature, interval):
        expr = f"{fn}({feature.name})"
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = (
                f"daterange((aod.as_of_date - interval '{interval}')::date, "
                f"aod.as_of_date::date, '[]')"
            )
            expr += f" filter (where {daterange} @> {event_date}::date)"
        return expr

    def _build_aggregate_expression(self, feature, interval=None):
        return f"({self._agg('max', feature, interval)} - {self._agg('min', feature, interval)})"


class Percentile95(OrderedSetAggregator):
    """95th percentile via the ordered-set base (handles WITHIN GROUP + interval)."""

    def __init__(self):
        super().__init__(name="p95", aggregate="percentile_cont", direct_argument=0.95)


# ============================================================================
# Custom Transformations
# ============================================================================


class Log1p(Transformer):
    """Natural log with a +1 offset: ``ln(x + 1)``.

    Guarded with a CASE: the logarithm is undefined for ``x + 1 <= 0``, and this
    data has negative values (withdrawals), so those map to NULL rather than
    raising. (Without the guard PostgreSQL errors on the whole query.)
    """

    def __init__(self):
        super().__init__(name="log1p")

    def _build_transformer_call(self, feature):
        return f"case when {feature.name} + 1 > 0 then ln({feature.name} + 1) end"


class ZScore(Transformer):
    """Z-score standardization across the rows in scope (``OVER ()``)."""

    def __init__(self):
        super().__init__(name="zscore")

    def _build_transformer_call(self, feature):
        return (
            f"(({feature.name} - avg({feature.name}) over ()) "
            f"/ nullif(stddev({feature.name}) over (), 0))"
        )


class BinCount(Transformer):
    """Discretize into 5 equal-width bins (0–4) across the rows in scope."""

    def __init__(self):
        super().__init__(name="bin", output_type="numeric")

    def _build_transformer_call(self, feature):
        f = feature.name
        return (
            f"case when {f} is null then null "
            f"else cast(floor(5 * ({f} - min({f}) over ()) "
            f"/ nullif(max({f}) over () - min({f}) over (), 0)) as integer) end"
        )


# ============================================================================
# Registration
# ============================================================================


def register_all_custom_primitives():
    """Register all custom primitives with the feature system."""
    register_aggregation("range", Range())
    register_aggregation("p95", Percentile95())

    register_transformer("log1p", Log1p())
    register_transformer("zscore", ZScore())
    register_transformer("bin", BinCount())

    print("✓ Registered custom primitives:")
    print("  Aggregations: range, p95")
    print("  Transformations: log1p, zscore, bin")
