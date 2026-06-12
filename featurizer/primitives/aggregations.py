# coding: utf-8

"""Aggregation primitives and registry wiring.

Each aggregator registers itself through `register_aggregation` so new primitives
can be discovered without editing the featurizer core.

Aggregation primitives are applied when traversing backward relationships
(parent ← child). They reduce multiple rows from the child entity to a single
value for each parent entity row.

Available aggregations:
    - Basic: sum, min, max, mean, stddev, variance, count, nunique
    - Boolean: all, any
    - Ordered-set: median, mode
    - Statistical: min_max_scale, mean_deviation, z_score, skewness, kurtosis
    - Mean variants: harmonic_mean, geometric_mean

Example usage:
    >>> from featurizer.primitives.utils import get_aggregations
    >>> aggs = get_aggregations(["sum", "mean", "median"])
    >>> for name, agg in aggs.items():
    ...     print(f"{name}: {agg}")

Most aggregations support temporal interval filtering when the entity has a
temporal_ix defined. This allows computing aggregates over specific time windows
(e.g., "sum of orders in the last 7 days").
"""

import hashlib

from .abstractions import Feature, SpatialIx
from .utils import register_aggregation


class Aggregator:
    """Base class for aggregation functions.

    Aggregators reduce multiple rows to a single value, following PostgreSQL's
    aggregate function semantics. They are applied when traversing backward
    relationships in the entity graph.

    From the PostgreSQL docs:
        "An aggregate function reduces multiple inputs to a single output value,
        such as the sum or average of the inputs."

    Attributes:
        name: Unique identifier for this aggregation (e.g., 'sum', 'mean').
        aggregate: SQL aggregate function name (defaults to name).
        input_types: List of feature types this aggregation accepts.
        output_type: Type of the resulting feature.
        distinct: If True, applies DISTINCT to the aggregate.
        order_by: Optional ORDER BY clause for ordered-set aggregates.
        stackable: If True, can be composed with other primitives.

    Example:
        >>> # Create a custom aggregation
        >>> class ProductAgg(Aggregator):
        ...     def __init__(self):
        ...         super().__init__(name='product')
        ...     def _build_aggregate_expression(self, feature, interval):
        ...         return f"EXP(SUM(LN({feature.name})))"
        >>> register_aggregation('product', ProductAgg())

    SQL Generation:
        The aggregator generates SQL like:
        - Basic: `SUM(amount)` or `AVG(price)`
        - With interval: `SUM(amount) FILTER (WHERE daterange(...) @> date)`
        - With DISTINCT: `COUNT(DISTINCT category)`
    """

    def __init__(
        self,
        name,
        aggregate=None,
        input_types=["numeric"],
        output_type="numeric",
        distinct=False,
        order_by=None,
        stackable=True,
    ):
        self.name = name
        self.aggregate = aggregate if aggregate else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.distinct = (
            distinct  # If DISTINCT is specified in addition to an order_by_clause,
        )
        # then all the ORDER BY expressions must match regular arguments of
        # the aggregate; that is, you cannot sort on
        # an expression that is not included in the DISTINCT list .
        self.order_by = order_by  # ' ORDER BY expression' i.e. input columns
        # self.filter_by = filter_by  # filter' FILTER WHERE :filter'
        self.stackable = stackable

    @staticmethod
    def _build_name(name, feature, interval):
        name = f"{str.upper(name)}({feature.entity.alias}.{feature.name}"
        interval = f"|interval={interval})" if interval else ")"
        name = (name + interval).replace('"', "")
        # PostgreSQL truncates identifiers to 63 bytes (NAMEDATALEN - 1).
        # Two long names sharing a 63-byte prefix (e.g. the P6M and P1Y
        # interval variants of the same feature) would silently collide into
        # one ambiguous column, so cap long names with a stable hash suffix.
        if len(name.encode()) > 63:
            digest = hashlib.md5(name.encode()).hexdigest()[:8]
            name = f"{name[:54]}~{digest}"
        return f'"{name}"'

    def _build_aggregate_expression(self, feature, interval):
        expression = feature.name
        aggregate_expression = [
            f"{self.aggregate}({'distinct' if self.distinct else ''} {expression}"
        ]
        if self.order_by and feature.sort:
            # order by clause
            aggregate_expression.append(f"order by {feature.sort})")
        else:
            aggregate_expression.append(")")
        if interval:
            # filter by clause
            event_date = feature.entity.temporal_ix.name
            daterange = f" daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]') "
            aggregate_expression.append(
                f" filter (where {daterange} @>  {event_date}::date) "
            )
        return " ".join(aggregate_expression)

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.type == "key":
            agg_feature = feature
        elif feature.type not in self.input_types:
            # We don't do anything
            agg_feature = None
        else:
            agg_feature = Feature(
                name=self._build_name(self.name, feature, interval=interval),
                type=self.output_type,
                definition=self._build_aggregate_expression(feature, interval),
                parents=feature,
                entity=parent,
                stack_depth=feature.stack_depth + 1,
            )
        return agg_feature


class Zscore(Aggregator):
    """Z-score (standard score) aggregation.

    Computes how many standard deviations a value is from the mean.
    Useful for identifying outliers and normalizing distributions.

    SQL: (ABS(value - AVG(value)) / STDDEV(value))
    """

    def __init__(self):
        super().__init__(name="zscore")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"(abs({feature.name} - avg({feature.name})) / stddev({feature.name})"


class Skewness(Aggregator):
    """Skewness aggregation - measure of distribution asymmetry.

    Positive skewness indicates a right-tailed distribution.
    Negative skewness indicates a left-tailed distribution.
    Values near zero indicate a symmetric distribution.

    SQL: ((value - AVG(value)) / STDDEV(value))^3
    """

    def __init__(self):
        super().__init__(name="skewness")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"({feature.name} - avg({feature.name})) / stddev({feature.name})**3"


class Kurtosis(Aggregator):
    """Kurtosis aggregation - measure of distribution tailedness.

    High kurtosis indicates heavy tails (more outliers).
    Low kurtosis indicates light tails (fewer outliers).
    Normal distribution has kurtosis of 3.

    SQL: ((value - AVG(value)) / STDDEV(value))^4
    """

    def __init__(self):
        super().__init__(name="kurtosis")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"({feature.name} - avg({feature.name})) / stddev({feature.name})**4"


class MinMaxScale(Aggregator):
    """Min-max normalization aggregation.

    Scales values to a 0-1 range based on min and max.
    Useful for comparing features with different scales.

    SQL: (value - MIN(value)) / (MAX(value) - MIN(value))
    """

    def __init__(self):
        super().__init__(name="min_max_scale")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"1.0*({feature.name} - min({feature.name})/(max({feature.name}) - min({feature.name}))"


class AverageDeviation(Aggregator):
    """Mean absolute deviation aggregation.

    Measures average distance from the mean. More robust to outliers
    than standard deviation.

    SQL: SUM(ABS(value - AVG(value))) / COUNT(value)
    """

    def __init__(self):
        super().__init__(name="mean_deviation")

    def _build_aggregate_expression(self, feature, interval=None):
        return (
            f"(sum(abs({feature.name} - avg({feature.name}))) / count({feature.name}))"
        )


class HarmonicMean(Aggregator):
    """
    It is the appropriate when dealing with rates and prices

    From the wikipedia:
    The harmonic mean of a list of numbers tends strongly toward
    the least elements of the list, it tends (compared to the arithmetic mean)
    to mitigate the impact of large outliers and aggravate
    the impact of small ones
    """

    def __init__(self):
        super().__init__(name="harmonic_mean")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"(count({feature.name}) / sum(1.0/{feature.name}))"


class GeometricMean(Aggregator):
    """
    Is a better measure of central tendency than a simple arithmetic
    mean when you are analyzing change over time

    From the wikipedia:
    This makes the geometric mean the only correct
    mean when averaging normalized results; that is,
    results that are presented as ratios to reference values
    """

    def __init__(self):
        super().__init__(name="geometric_mean")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"""(
        case
        when {feature.name} > 0
        then
        exp(avg(log({feature.name}))
        else
        (-1.0)^count(*)*exp(avg(log(abs({feature.name})))
        end
        )
        """


sum = Aggregator(name="sum")
min = Aggregator(name="min")
max = Aggregator(name="max")
mean = Aggregator(name="mean", aggregate="avg")
stddev = Aggregator(name="stddev")
var = Aggregator(name="variance")
count = Aggregator(name="count", input_types=["categorical", "index"])
all = Aggregator(
    name="all", aggregate="bool_and", input_types=["boolean"], output_type="boolean"
)
any = Aggregator(
    name="any", aggregate="bool_or", input_types=["boolean"], output_type="boolean"
)
nunique = Aggregator(
    name="nunique",
    aggregate="count",
    input_types=["categorical", "index"],
    distinct=True,
)
min_max_scale = MinMaxScale()
mean_deviation = AverageDeviation()
z_score = Zscore()
skewness = Skewness()
kurtosis = Kurtosis()
harmonic_mean = HarmonicMean()
geometric_mean = GeometricMean()

DEFAULT_AGGREGATIONS = {
    "sum": sum,
    "min": min,
    "max": max,
    "mean": mean,
    "stddev": stddev,
    "variance": var,
    "count": count,
    "all": all,
    "any": any,
    "nunique": nunique,
    "min_max_scale": min_max_scale,
    "mean_deviation": mean_deviation,
    "z_score": z_score,
    "skewness": skewness,
    "kurtosis": kurtosis,
    "harmonic_mean": harmonic_mean,
    "geometric_mean": geometric_mean,
}

for _name, _agg in DEFAULT_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


class OrderedSetAggregator(Aggregator):
    """Ordered-set aggregate functions (e.g., median, mode, percentiles).

    These aggregates require an ORDER BY clause within the aggregate call
    (WITHIN GROUP syntax) because their computation depends on row ordering.

    Common examples:
        - PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) -- median
        - MODE() WITHIN GROUP (ORDER BY category) -- most frequent value

    From PostgreSQL docs:
        "For an ordered-set aggregate, the order_by_clause is written inside
        WITHIN GROUP (...). The expressions are evaluated once per input row,
        sorted per the ORDER BY requirements, and fed to the aggregate function."

    Attributes:
        direct_argument: Value passed before WITHIN GROUP (e.g., 0.5 for median).

    SQL Pattern:
        AGGREGATE(direct_arg) WITHIN GROUP (ORDER BY expression)
    """

    def __init__(
        self,
        name,
        aggregate=None,
        direct_argument=None,
        input_types=["numeric"],
        output_type="numeric",
        filter=None,
        stackable=True,
    ):
        self.order_by = True
        self.direct_argument = direct_argument
        self.filter = filter
        super().__init__(
            name=name,
            aggregate=aggregate,
            input_types=input_types,
            output_type=output_type,
            order_by=self.order_by,
            stackable=stackable,
        )

    def _build_aggregate_expression(self, feature, interval=None):
        expression = feature.name
        if self.direct_argument:
            aggregate_expression = [f"{self.aggregate}({self.direct_argument})"]
        else:
            aggregate_expression = [f"{self.aggregate}()"]
        aggregate_expression.append(f"within group(order by {expression})")
        filters = []
        if self.filter and feature.specials:
            filters.append(f"{feature.name} = {feature.specials}")
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filters.append(f"{daterange} @> {event_date}::date")
        if filters:
            aggregate_expression.append(f" filter (where {' and '.join(filters)})")
        return " ".join(aggregate_expression)


median = OrderedSetAggregator(
    name="median", aggregate="percentile_cont", direct_argument=0.5
)
mode = OrderedSetAggregator(name="mode", input_types=["categorical"])

register_aggregation("median", median)
register_aggregation("mode", mode)

p10 = OrderedSetAggregator(name="p10", aggregate="percentile_cont", direct_argument=0.1)
p25 = OrderedSetAggregator(
    name="p25", aggregate="percentile_cont", direct_argument=0.25
)
p75 = OrderedSetAggregator(
    name="p75", aggregate="percentile_cont", direct_argument=0.75
)
p90 = OrderedSetAggregator(name="p90", aggregate="percentile_cont", direct_argument=0.9)
p95 = OrderedSetAggregator(
    name="p95", aggregate="percentile_cont", direct_argument=0.95
)
p99 = OrderedSetAggregator(
    name="p99", aggregate="percentile_cont", direct_argument=0.99
)


class IQR(Aggregator):
    """Interquartile range: P75 - P25."""

    def __init__(self):
        super().__init__(name="iqr")

    def _build_aggregate_expression(self, feature, interval=None):
        p75 = f"percentile_cont(0.75) within group(order by {feature.name})"
        p25 = f"percentile_cont(0.25) within group(order by {feature.name})"
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
            p75 += filt
            p25 += filt
        return f"({p75}) - ({p25})"


iqr = IQR()


class CoefficientOfVariation(Aggregator):
    """Coefficient of variation: STDDEV / AVG."""

    def __init__(self):
        super().__init__(name="cv")

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return f"stddev({feature.name}){filt} / NULLIF(avg({feature.name}){filt}, 0)"


cv = CoefficientOfVariation()


class Range(Aggregator):
    """Range: MAX - MIN."""

    def __init__(self):
        super().__init__(name="range")

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return f"max({feature.name}){filt} - min({feature.name}){filt}"


range_agg = Range()


class EventRate(Aggregator):
    """Events per unit time: COUNT / time span in seconds."""

    def __init__(self):
        super().__init__(name="event_rate", input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, **kwargs):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(parent, child, feature, interval=interval)

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return (
            f"count(*){filt} / NULLIF(EXTRACT(EPOCH FROM "
            f"max({feature.name}){filt} - min({feature.name}){filt}), 0)"
        )


event_rate = EventRate()


class TimeSpan(Aggregator):
    """Time span in seconds between first and last event."""

    def __init__(self):
        super().__init__(name="time_span", input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, **kwargs):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(parent, child, feature, interval=interval)

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return (
            f"EXTRACT(EPOCH FROM max({feature.name}){filt} - min({feature.name}){filt})"
        )


time_span = TimeSpan()


class Recency(Aggregator):
    """Days since the most recent event at or before the as-of date.

    ``aod.as_of_date - max(event_ts)`` — the single highest-value as-of-state
    feature. Backward-only by construction: ``max`` runs over rows the
    aggregation CTE already cut at ``aod.as_of_date >= temporal_ix``. Fires only
    on the entity's temporal_ix.
    """

    def __init__(self, name="recency"):
        super().__init__(name=name, input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, **kwargs):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(parent, child, feature, interval=interval)

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return f"(aod.as_of_date::date - (max({feature.name}){filt})::date)"


class Tenure(Aggregator):
    """Days since the first observed event (age in system).

    ``aod.as_of_date - min(event_ts)``. Backward-only; fires on temporal_ix.
    """

    def __init__(self, name="tenure"):
        super().__init__(name=name, input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, **kwargs):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(parent, child, feature, interval=interval)

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return f"(aod.as_of_date::date - (min({feature.name}){filt})::date)"


class InterEventHazard(Aggregator):
    """Events per day over the observed lifespan: count / (aod - first event).

    A cheap backward-only hazard proxy. Fires on temporal_ix.
    """

    def __init__(self, name="inter_event_hazard_proxy"):
        super().__init__(name=name, input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, **kwargs):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(parent, child, feature, interval=interval)

    def _build_aggregate_expression(self, feature, interval=None):
        filt = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            filt = f" filter (where {daterange} @> {event_date}::date)"
        return (
            f"(count(*){filt})::float / "
            f"NULLIF((aod.as_of_date::date - (min({feature.name}){filt})::date), 0)"
        )


recency = Recency()
tenure = Tenure()
age_in_system = Tenure(name="age_in_system")
inter_event_hazard_proxy = InterEventHazard()

# --- SubqueryAggregator infrastructure ---


class SubqueryAggregator(Aggregator):
    """Base class for aggregations requiring correlated subqueries.

    Used when the aggregation needs access to per-row data within the
    GROUP BY context (e.g., inter-event gap statistics, entropy).
    Overrides __call__ to pass child entity and relationship info
    to _build_subquery_expression.
    """

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.type == "key":
            return feature
        if feature.type not in self.input_types:
            return None
        if relationship is None:
            return None
        definition = self._build_subquery_expression(
            feature, child, relationship, interval
        )
        if definition is None:
            return None
        return Feature(
            name=self._build_name(self.name, feature, interval=interval),
            type=self.output_type,
            definition=definition,
            entity=parent,
            parents=feature,
            stack_depth=feature.stack_depth + 1,
        )

    @staticmethod
    def _causal_filter(feature, interval, *, alias="sub"):
        """Backward causal bound for a correlated subquery on the child stream.

        Returns a SQL fragment beginning with ' and ' that bounds the subquery
        to rows at or before the as-of date:
        - with an interval: the daterange window (upper bound aod.as_of_date);
        - without an interval: a plain ``<= aod.as_of_date``;
        - empty when the entity has no temporal_ix (no time axis to bound).

        Without this, a correlated subquery reading ``<child>_transform`` would
        see the entity's *future* rows — the outer ``where aod.as_of_date >=
        temporal_ix`` does not reach into the subquery — leaking the label window.
        """
        tix = getattr(feature.entity, "temporal_ix", None) if feature.entity else None
        if tix is None:
            return ""
        col = f"{alias}.{tix.name}"
        if interval:
            daterange = (
                f"daterange((aod.as_of_date - interval '{interval}')::date, "
                f"aod.as_of_date::date, '[]')"
            )
            return f" and {daterange} @> {col}::date"
        return f" and {col} <= aod.as_of_date"

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        raise NotImplementedError


class GapStatAggregator(SubqueryAggregator):
    """Inter-event gap statistics via correlated subquery with LAG().

    Computes intervals between consecutive events, then applies an
    aggregate function (AVG, STDDEV, MIN, MAX) to those gaps.
    Only fires on temporal_ix features.
    """

    def __init__(self, name, gap_aggregate):
        super().__init__(name=name, input_types=["index"])
        self.gap_aggregate = gap_aggregate

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.name
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT {self.gap_aggregate}(gap) FROM ("
            f"SELECT sub.{event_col} - LAG(sub.{event_col}) OVER (ORDER BY sub.{event_col}) as gap "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") gaps WHERE gap IS NOT NULL)"
        )


gap_mean = GapStatAggregator(name="gap_mean", gap_aggregate="AVG")
gap_stddev = GapStatAggregator(name="gap_stddev", gap_aggregate="STDDEV")
gap_min = GapStatAggregator(name="gap_min", gap_aggregate="MIN")
gap_max = GapStatAggregator(name="gap_max", gap_aggregate="MAX")


class GapCV(SubqueryAggregator):
    """Coefficient of variation of inter-event gaps."""

    def __init__(self):
        super().__init__(name="gap_cv", input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.name
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT STDDEV(gap) / NULLIF(AVG(gap), 0) FROM ("
            f"SELECT sub.{event_col} - LAG(sub.{event_col}) OVER (ORDER BY sub.{event_col}) as gap "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") gaps WHERE gap IS NOT NULL)"
        )


gap_cv = GapCV()


class Burstiness(SubqueryAggregator):
    """Goh-Barabasi burstiness index: (sigma - mu) / (sigma + mu) of inter-event gaps."""

    def __init__(self):
        super().__init__(name="burstiness", input_types=["index"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.name
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT (STDDEV(gap) - AVG(gap)) / NULLIF(STDDEV(gap) + AVG(gap), 0) FROM ("
            f"SELECT sub.{event_col} - LAG(sub.{event_col}) OVER (ORDER BY sub.{event_col}) as gap "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") gaps WHERE gap IS NOT NULL)"
        )


burstiness = Burstiness()


class Entropy(SubqueryAggregator):
    """Shannon entropy of categorical distributions."""

    def __init__(self):
        super().__init__(name="entropy", input_types=["categorical"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT -SUM(freq::float / total * LN(freq::float / total)) "
            f"FROM (SELECT COUNT(*) as freq, SUM(COUNT(*)) OVER () as total "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter} "
            f"GROUP BY sub.{feature.name}) entropy_calc)"
        )


entropy = Entropy()


class HHI(SubqueryAggregator):
    """Herfindahl-Hirschman Index of categorical concentration."""

    def __init__(self):
        super().__init__(name="hhi", input_types=["categorical"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT SUM(POWER(freq::float / NULLIF(total, 0), 2)) "
            f"FROM (SELECT COUNT(*) as freq, SUM(COUNT(*)) OVER () as total "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter} "
            f"GROUP BY sub.{feature.name}) hhi_calc)"
        )


hhi = HHI()


class Gini(SubqueryAggregator):
    """Gini coefficient of numeric distributions.

    Uses the formula: (2 * SUM(i * x_i)) / (n * SUM(x_i)) - (n + 1) / n
    where x_i are values sorted in ascending order and i is the rank.
    """

    def __init__(self):
        super().__init__(name="gini", input_types=["numeric"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT (2.0 * SUM(rn * val)) / NULLIF(COUNT(*) * SUM(val), 0) "
            f"- (COUNT(*) + 1.0) / NULLIF(COUNT(*), 0) "
            f"FROM (SELECT sub.{feature.name} as val, "
            f"ROW_NUMBER() OVER (ORDER BY sub.{feature.name}) as rn "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") gini_calc)"
        )


gini = Gini()


class NgramFrequency(SubqueryAggregator):
    """Most common N-gram frequency for sequence features.

    Computes the frequency of the most common consecutive pair (bigram)
    of categorical values, ordered by temporal index.
    """

    def __init__(self, n=2):
        self.n = n
        super().__init__(name=f"ngram_{n}_freq", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.entity.temporal_ix.name
        interval_filter = self._causal_filter(feature, interval)
        lag_cols = ", ".join(
            f"LAG(sub.{feature.name}, {i}) OVER (ORDER BY sub.{event_col}) as lag_{i}"
            for i in range(1, self.n)
        )
        return (
            f"(SELECT MAX(cnt) FROM ("
            f"SELECT COUNT(*) as cnt FROM ("
            f"SELECT sub.{feature.name} as curr, {lag_cols} "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") ngrams WHERE lag_1 IS NOT NULL "
            f"GROUP BY curr, {', '.join(f'lag_{i}' for i in range(1, self.n))}"
            f") ngram_counts)"
        )


ngram_2_freq = NgramFrequency(n=2)
ngram_3_freq = NgramFrequency(n=3)


class SequenceEntropy(SubqueryAggregator):
    """Transition entropy of consecutive categorical values.

    Measures the randomness of transitions between consecutive values
    in a temporal sequence. Higher values indicate more random transitions.
    """

    def __init__(self):
        super().__init__(name="sequence_entropy", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.entity.temporal_ix.name
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT -SUM(freq::float / total * LN(freq::float / total)) "
            f"FROM (SELECT COUNT(*) as freq, SUM(COUNT(*)) OVER () as total "
            f"FROM (SELECT sub.{feature.name} as curr, "
            f"LAG(sub.{feature.name}) OVER (ORDER BY sub.{event_col}) as prev "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") transitions WHERE prev IS NOT NULL "
            f"GROUP BY curr, prev) seq_entropy_calc)"
        )


sequence_entropy = SequenceEntropy()


class LongestStreak(SubqueryAggregator):
    """Longest consecutive streak of the same categorical value."""

    def __init__(self):
        super().__init__(name="longest_streak", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        event_col = feature.entity.temporal_ix.name
        interval_filter = self._causal_filter(feature, interval)
        return (
            f"(SELECT MAX(streak_len) FROM ("
            f"SELECT COUNT(*) as streak_len FROM ("
            f"SELECT sub.{feature.name}, "
            f"ROW_NUMBER() OVER (ORDER BY sub.{event_col}) - "
            f"ROW_NUMBER() OVER (PARTITION BY sub.{feature.name} ORDER BY sub.{event_col}) as grp "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{interval_filter}"
            f") streaks GROUP BY {feature.name}, grp) streak_counts)"
        )


longest_streak = LongestStreak()


# --- Phase 5: distributional & sequence reductions (all bounded via _causal_filter) ---


class Theil(SubqueryAggregator):
    """Theil-T inequality index over positive values: mean((x/mu)*ln(x/mu))."""

    def __init__(self):
        super().__init__(name="theil", input_types=["numeric"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        col = feature.name
        return (
            f"(SELECT AVG((val / m) * LN(val / m)) FROM ("
            f"SELECT sub.{col} AS val, AVG(sub.{col}) OVER () AS m "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal} "
            f"AND sub.{col} > 0) theil_calc)"
        )


theil = Theil()


class TrimmedMean(SubqueryAggregator):
    """Symmetric trimmed mean: mean of values within [p_lo, p_hi]."""

    def __init__(self, name="trimmed_mean_10", lower=0.10, upper=0.90):
        super().__init__(name=name, input_types=["numeric"])
        self.lower = lower
        self.upper = upper

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        col = feature.name
        return (
            f"(SELECT AVG(q.val) FROM ("
            f"SELECT sub.{col} AS val FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}) q, ("
            f"SELECT percentile_cont({self.lower}) WITHIN GROUP (ORDER BY sub.{col}) AS lo, "
            f"percentile_cont({self.upper}) WITHIN GROUP (ORDER BY sub.{col}) AS hi "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}) b "
            f"WHERE q.val BETWEEN b.lo AND b.hi)"
        )


trimmed_mean_10 = TrimmedMean()


class MedianAbsoluteDeviation(SubqueryAggregator):
    """MAD: median(|x - median(x)|)."""

    def __init__(self):
        super().__init__(name="median_absolute_deviation", input_types=["numeric"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        col = feature.name
        return (
            f"(SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(q.val - b.med)) "
            f"FROM (SELECT sub.{col} AS val FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}) q, ("
            f"SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY sub.{col}) AS med "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}) b)"
        )


median_absolute_deviation = MedianAbsoluteDeviation()


class _SequenceReduction(SubqueryAggregator):
    """Base for categorical sequence reductions ordered by temporal_ix.

    Subclasses reduce a (curr, prev) transition set. Fires only on entities
    with a temporal_ix.
    """

    def __init__(self, name):
        super().__init__(name=name, input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _transitions(self, feature, child, relationship, interval):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        return (
            f"SELECT sub.{col} AS curr, "
            f"LAG(sub.{col}) OVER (ORDER BY sub.{ts}) AS prev "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}"
        )


class StateVolatility(_SequenceReduction):
    """Count of value changes (transitions where prev != curr)."""

    def __init__(self):
        super().__init__(name="state_volatility")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        transitions = self._transitions(feature, child, relationship, interval)
        return (
            f"(SELECT count(*) FROM ({transitions}) t "
            f"WHERE t.prev IS DISTINCT FROM t.curr AND t.prev IS NOT NULL)"
        )


state_volatility = StateVolatility()


class TransitionMatrixSummary(_SequenceReduction):
    """Number of distinct observed (prev -> curr) transitions."""

    def __init__(self):
        super().__init__(name="transition_matrix_summary")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        transitions = self._transitions(feature, child, relationship, interval)
        return (
            f"(SELECT count(DISTINCT (t.prev, t.curr)) FROM ({transitions}) t "
            f"WHERE t.prev IS NOT NULL)"
        )


transition_matrix_summary = TransitionMatrixSummary()


class ReworkCount(_SequenceReduction):
    """Count of self-loops (consecutive repeats, prev == curr)."""

    def __init__(self):
        super().__init__(name="rework_count")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        transitions = self._transitions(feature, child, relationship, interval)
        return f"(SELECT count(*) FROM ({transitions}) t WHERE t.prev = t.curr)"


rework_count = ReworkCount()


class TimeInCurrentState(_SequenceReduction):
    """Days since the most recent change of a categorical attribute (dwell)."""

    def __init__(self):
        super().__init__(name="time_in_current_state")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval, alias="s")
        ts = feature.entity.temporal_ix.name
        col = feature.name
        return (
            f"(aod.as_of_date::date - (SELECT max(run.ts) FROM ("
            f"SELECT s.{ts} AS ts, s.{col} AS curr, "
            f"LAG(s.{col}) OVER (ORDER BY s.{ts}) AS prev "
            f"FROM {child_table} s "
            f"WHERE s.{child_key} = {child_table}.{child_key}{causal}"
            f") run WHERE run.prev IS DISTINCT FROM run.curr)::date)"
        )


time_in_current_state = TimeInCurrentState()


class RecurrenceInterval(_SequenceReduction):
    """Mean days between consecutive occurrences of the *same* state.

    Unlike ``gap_mean`` (gaps between any two consecutive events), the LAG is
    partitioned by the categorical value, so it measures how often each state
    recurs. Timestamps are cast to date so the output is numeric days for both
    date and timestamp temporal indexes.
    """

    def __init__(self):
        super().__init__(name="recurrence_interval")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        return (
            f"(SELECT AVG(gap) FROM ("
            f"SELECT sub.{ts}::date - LAG(sub.{ts}::date) "
            f"OVER (PARTITION BY sub.{col} ORDER BY sub.{ts}) AS gap "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}"
            f") g WHERE gap IS NOT NULL)"
        )


recurrence_interval = RecurrenceInterval()


class _TransitionMatrixReduction(_SequenceReduction):
    """Base for reductions over the first-order transition matrix.

    Builds the grouped (prev, curr) frequency matrix with joint and
    row-conditional totals; subclasses reduce it to a scalar.
    """

    def _matrix(self, feature, child, relationship, interval):
        transitions = self._transitions(feature, child, relationship, interval)
        return (
            f"SELECT t.prev, t.curr, COUNT(*) AS freq, "
            f"SUM(COUNT(*)) OVER () AS total, "
            f"SUM(COUNT(*)) OVER (PARTITION BY t.prev) AS row_total "
            f"FROM ({transitions}) t WHERE t.prev IS NOT NULL "
            f"GROUP BY t.prev, t.curr"
        )


class MarkovConditionalEntropy(_TransitionMatrixReduction):
    """First-order Markov entropy rate H(X_t | X_{t-1}), in nats.

    ``-SUM p(i,j) * ln p(j|i)`` over the observed transition matrix: 0 for a
    perfectly predictable chain, ``ln(k)`` for a uniform one over k states.
    The existing ``sequence_entropy`` is the *joint* (prev, curr) entropy;
    this is the conditional entropy the Markov taxonomy actually calls for.
    """

    def __init__(self):
        super().__init__(name="markov_conditional_entropy")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        matrix = self._matrix(feature, child, relationship, interval)
        return (
            f"(SELECT -SUM((freq::float / total) * LN(freq::float / row_total)) "
            f"FROM ({matrix}) m)"
        )


markov_conditional_entropy = MarkovConditionalEntropy()


class MaxTransitionProbability(_TransitionMatrixReduction):
    """Predictability score: the largest conditional transition probability.

    ``MAX p(j|i)`` over the observed matrix — 1.0 when at least one state
    always transitions to the same successor.
    """

    def __init__(self):
        super().__init__(name="max_transition_prob")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        matrix = self._matrix(feature, child, relationship, interval)
        return f"(SELECT MAX(freq::float / row_total) FROM ({matrix}) m)"


max_transition_prob = MaxTransitionProbability()


class FirstPassageTime(SubqueryAggregator):
    """Days from the entity's first event to the first occurrence of a target
    state; NULL if the target state has not occurred by the as-of date.

    Requires a ``target`` predicate value and a temporal_ix, e.g.
    ``results: {type: categorical, predicates: {target: Fail}}``.
    """

    def __init__(self):
        super().__init__(name="first_passage_time", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if "target" not in feature.predicates:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        target = feature.predicates["target"]
        return (
            f"(SELECT (MIN(sub.{ts}) FILTER (WHERE sub.{col} = '{target}'))::date "
            f"- MIN(sub.{ts})::date "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal})"
        )


first_passage_time = FirstPassageTime()


class _NumericStreamReduction(SubqueryAggregator):
    """Base for numeric reductions needing temporal ordering (ACF, VR, cosinor)."""

    def __init__(self, name):
        super().__init__(name=name, input_types=["numeric"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )


class AutoCorrelation(_NumericStreamReduction):
    """Lag-k autocorrelation: corr(x_t, x_{t-k}) over the backward window."""

    def __init__(self, k=1):
        super().__init__(name=f"acf_{k}")
        self.k = k

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        return (
            f"(SELECT corr(val, lagk) FROM ("
            f"SELECT sub.{col} AS val, "
            f"LAG(sub.{col}, {self.k}) OVER (ORDER BY sub.{ts}) AS lagk "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}"
            f") t WHERE lagk IS NOT NULL)"
        )


acf_1 = AutoCorrelation(k=1)


class VarianceRatio(_NumericStreamReduction):
    """Variance ratio: var(value) / var(first difference) over the window."""

    def __init__(self):
        super().__init__(name="variance_ratio")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        return (
            f"(SELECT var_samp(val) / NULLIF(var_samp(d), 0) FROM ("
            f"SELECT sub.{col} AS val, "
            f"sub.{col} - LAG(sub.{col}) OVER (ORDER BY sub.{ts}) AS d "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}"
            f") t)"
        )


variance_ratio = VarianceRatio()


class CosinorAmplitude(_NumericStreamReduction):
    """Cosinor amplitude over a fixed period (orthogonal-basis approximation).

    ``sqrt(regr_slope(x, sin)^2 + regr_slope(x, cos)^2)`` with the sin/cos basis
    built from the event timestamp. Exact when the basis columns are
    uncorrelated over the window; otherwise a seasonal-strength approximation.
    Backward-only.
    """

    def __init__(self, name="cosinor_amplitude_weekly", period_seconds=7 * 86400):
        super().__init__(name=name)
        self.period_seconds = period_seconds

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        child_key = relationship.child_key
        child_table = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        ts = feature.entity.temporal_ix.name
        col = feature.name
        omega = f"2 * pi() * extract(epoch from sub.{ts}) / {self.period_seconds}"
        return (
            f"(SELECT sqrt(power(regr_slope(val, s), 2) + power(regr_slope(val, c), 2)) "
            f"FROM (SELECT sub.{col} AS val, sin({omega}) AS s, cos({omega}) AS c "
            f"FROM {child_table} sub "
            f"WHERE sub.{child_key} = {child_table}.{child_key}{causal}"
            f") t)"
        )


cosinor_amplitude = CosinorAmplitude()


# --- Two-window distributional drift (recent vs prior baseline window) ---


class TwoWindowDriftAggregator(SubqueryAggregator):
    """Drift between the recent window ``[t0-W, t0]`` and the prior baseline
    window ``[t0-2W, t0-W)``.

    Interval-only: the interval IS the window width W, so it returns None on the
    planner's non-interval pass. Backward-safe because the baseline window's
    upper bound is ``t0-W`` (strictly in the past) and the recent window's is t0.
    """

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if interval is None:
            return None
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    @staticmethod
    def _windows(feature, interval):
        ts = feature.entity.temporal_ix.name
        recent = (
            f"daterange((aod.as_of_date - interval '{interval}')::date, "
            f"aod.as_of_date::date, '[]') @> sub.{ts}::date"
        )
        baseline = (
            f"daterange((aod.as_of_date - (2 * interval '{interval}'))::date, "
            f"(aod.as_of_date - interval '{interval}')::date, '[)') @> sub.{ts}::date"
        )
        return recent, baseline


class KLDrift(TwoWindowDriftAggregator):
    """KL divergence of the recent vs baseline categorical distribution.

    Summed over the shared category support (categories absent from the baseline
    are dropped by the join — a documented simplification of full KL).
    """

    def __init__(self):
        super().__init__(name="kl_drift", input_types=["categorical"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        col = feature.name
        recent, baseline = self._windows(feature, interval)

        def dist(win):
            return (
                f"SELECT sub.{col} AS v, count(*)::float / SUM(count(*)) OVER () AS p "
                f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck} AND {win} GROUP BY sub.{col}"
            )

        return (
            f"(SELECT COALESCE(SUM(r.p * LN(r.p / NULLIF(b.p, 0))), 0) "
            f"FROM ({dist(recent)}) r JOIN ({dist(baseline)}) b ON r.v = b.v)"
        )


kl_drift = KLDrift()


class WassersteinDrift(TwoWindowDriftAggregator):
    """Drift as the L1 distance between recent and baseline quantiles.

    A coarse Wasserstein-1 proxy on a fixed quantile grid (p10/p50/p90) — kept
    to constant fractions so the generated SQL is unambiguously valid.
    """

    def __init__(self):
        super().__init__(name="wasserstein_drift", input_types=["numeric"])

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        col = feature.name
        recent, baseline = self._windows(feature, interval)

        def quants(win):
            return (
                f"SELECT percentile_cont(0.1) WITHIN GROUP (ORDER BY sub.{col}) AS q10, "
                f"percentile_cont(0.5) WITHIN GROUP (ORDER BY sub.{col}) AS q50, "
                f"percentile_cont(0.9) WITHIN GROUP (ORDER BY sub.{col}) AS q90 "
                f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck} AND {win}"
            )

        return (
            f"(SELECT ABS(r.q10 - b.q10) + ABS(r.q50 - b.q50) + ABS(r.q90 - b.q90) "
            f"FROM ({quants(recent)}) r, ({quants(baseline)}) b)"
        )


wasserstein_drift = WassersteinDrift()


# --- Predicate-driven aggregators (event-type semantics) ---


class RightCensoringIndicator(SubqueryAggregator):
    """1 if a terminal event has NOT occurred by t0 (right-censored), else 0.

    Requires the feature to declare a ``terminal`` predicate value, e.g.
    ``event_type: {type: categorical, predicates: {terminal: cancel}}``.
    """

    def __init__(self):
        super().__init__(name="right_censoring_indicator", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if "terminal" not in feature.predicates:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        col = feature.name
        terminal = feature.predicates["terminal"]
        return (
            f"(SELECT (count(*) FILTER (WHERE sub.{col} = '{terminal}') = 0)::int "
            f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck}{causal})"
        )


right_censoring_indicator = RightCensoringIndicator()


class CrossTypeLatency(SubqueryAggregator):
    """Mean time (seconds) from an A-typed event to the next B-typed event.

    Requires ``a`` and ``b`` predicate values on the feature and a temporal_ix:
    ``event_type: {type: categorical, predicates: {a: order, b: deliver}}``.
    """

    def __init__(self):
        super().__init__(name="cross_type_latency", input_types=["categorical"])

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if "a" not in feature.predicates or "b" not in feature.predicates:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        ts = feature.entity.temporal_ix.name
        col = feature.name
        a_val = feature.predicates["a"]
        b_val = feature.predicates["b"]
        a_causal = self._causal_filter(feature, interval, alias="a")
        b_causal = self._causal_filter(feature, interval, alias="b")
        return (
            f"(SELECT AVG(lat) FROM ("
            f"SELECT EXTRACT(EPOCH FROM MIN(b.{ts}) - a.{ts}) AS lat "
            f"FROM {ct} a JOIN {ct} b "
            f"ON b.{ck} = a.{ck} AND b.{ts} > a.{ts} AND b.{col} = '{b_val}'{b_causal} "
            f"WHERE a.{ck} = {ct}.{ck} AND a.{col} = '{a_val}'{a_causal} "
            f"GROUP BY a.{ts}) lat_calc)"
        )


cross_type_latency = CrossTypeLatency()


# --- Spatial substrate (plain-SQL haversine; rides the backward traversal) ---


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (R = 6371000) between two lat/lon pairs."""
    return (
        f"2 * 6371000 * asin(sqrt("
        f"power(sin(radians({lat2} - {lat1}) / 2), 2) "
        f"+ cos(radians({lat1})) * cos(radians({lat2})) "
        f"* power(sin(radians({lon2} - {lon1}) / 2), 2)))"
    )


class SpatialAggregator(SubqueryAggregator):
    """Base for plain-SQL spatial reductions over an entity's event locations.

    Fires on the temporal_ix feature (one per entity) when the entity declares a
    plain lat/lon ``SpatialIx``; reads the lat/lon columns from it. Backward-safe
    via the inherited ``_causal_filter`` on the temporal_ix.
    """

    def __init__(self, name):
        super().__init__(name=name, input_types=["index"])

    def _spatial(self, feature):
        sx = getattr(feature.entity, "spatial_ix", None)
        if isinstance(sx, SpatialIx) and sx.lat and sx.lon:
            return sx
        return None

    def __call__(self, parent, child, feature, interval=None, *, relationship=None):
        if feature.entity is None or feature.entity.temporal_ix is None:
            return None
        if feature is not feature.entity.temporal_ix:
            return None
        if self._spatial(feature) is None:
            return None
        return super().__call__(
            parent, child, feature, interval=interval, relationship=relationship
        )


class DistanceTravelled(SpatialAggregator):
    """Total great-circle distance over consecutive events (metres)."""

    def __init__(self):
        super().__init__(name="distance_travelled")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        sx = self._spatial(feature)
        ts = feature.name
        step = _haversine_m("plat", "plon", "lat", "lon")
        return (
            f"(SELECT SUM({step}) FROM ("
            f"SELECT sub.{sx.lat} AS lat, sub.{sx.lon} AS lon, "
            f"LAG(sub.{sx.lat}) OVER (ORDER BY sub.{ts}) AS plat, "
            f"LAG(sub.{sx.lon}) OVER (ORDER BY sub.{ts}) AS plon "
            f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck}{causal}"
            f") steps WHERE plat IS NOT NULL)"
        )


class RadiusOfGyration(SpatialAggregator):
    """RMS great-circle distance of events from their centroid (metres)."""

    def __init__(self):
        super().__init__(name="radius_of_gyration")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        sx = self._spatial(feature)
        dist = _haversine_m("clat", "clon", "lat", "lon")
        return (
            f"(SELECT sqrt(AVG(power({dist}, 2))) FROM ("
            f"SELECT sub.{sx.lat} AS lat, sub.{sx.lon} AS lon, "
            f"AVG(sub.{sx.lat}) OVER () AS clat, AVG(sub.{sx.lon}) OVER () AS clon "
            f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck}{causal}"
            f") pts)"
        )


class SpatialStd(SpatialAggregator):
    """Degree-space spatial dispersion: sqrt(var(lat) + var(lon))."""

    def __init__(self):
        super().__init__(name="spatial_std")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        sx = self._spatial(feature)
        return (
            f"(SELECT sqrt(var_samp(sub.{sx.lat}) + var_samp(sub.{sx.lon})) "
            f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck}{causal})"
        )


class BoundingBoxArea(SpatialAggregator):
    """Approximate bounding-box area in m^2 (latitude-corrected degree box)."""

    def __init__(self):
        super().__init__(name="bbox_area")

    def _build_subquery_expression(self, feature, child, relationship, interval=None):
        ck = relationship.child_key
        ct = f"{child.alias}_transform"
        causal = self._causal_filter(feature, interval)
        sx = self._spatial(feature)
        return (
            f"(SELECT (max(sub.{sx.lat}) - min(sub.{sx.lat})) "
            f"* (max(sub.{sx.lon}) - min(sub.{sx.lon})) "
            f"* cos(radians(avg(sub.{sx.lat}))) * 111320 * 111320 "
            f"FROM {ct} sub WHERE sub.{ck} = {ct}.{ck}{causal})"
        )


distance_travelled = DistanceTravelled()
radius_of_gyration = RadiusOfGyration()
spatial_std = SpatialStd()
bbox_area = BoundingBoxArea()

_NEW_AGGREGATIONS = {
    "p10": p10,
    "p25": p25,
    "p75": p75,
    "p90": p90,
    "p95": p95,
    "p99": p99,
    "iqr": iqr,
    "cv": cv,
    "range": range_agg,
    "event_rate": event_rate,
    "time_span": time_span,
    "gap_mean": gap_mean,
    "gap_stddev": gap_stddev,
    "gap_min": gap_min,
    "gap_max": gap_max,
    "gap_cv": gap_cv,
    "burstiness": burstiness,
    "entropy": entropy,
    "hhi": hhi,
    "gini": gini,
    "ngram_2_freq": ngram_2_freq,
    "ngram_3_freq": ngram_3_freq,
    "sequence_entropy": sequence_entropy,
    "longest_streak": longest_streak,
}

DEFAULT_AGGREGATIONS.update(_NEW_AGGREGATIONS)

for _name, _agg in _NEW_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# --- As-of state aggregations (recency, tenure, hazard) ---

_ASOF_AGGREGATIONS = {
    "recency": recency,
    "tenure": tenure,
    "age_in_system": age_in_system,
    "inter_event_hazard_proxy": inter_event_hazard_proxy,
}

DEFAULT_AGGREGATIONS.update(_ASOF_AGGREGATIONS)

for _name, _agg in _ASOF_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# --- Distributional, sequence, and numeric-stream reductions ---

_REDUCTION_AGGREGATIONS = {
    "theil": theil,
    "trimmed_mean_10": trimmed_mean_10,
    "median_absolute_deviation": median_absolute_deviation,
    "state_volatility": state_volatility,
    "transition_matrix_summary": transition_matrix_summary,
    "rework_count": rework_count,
    "time_in_current_state": time_in_current_state,
    "recurrence_interval": recurrence_interval,
    "markov_conditional_entropy": markov_conditional_entropy,
    "max_transition_prob": max_transition_prob,
    "acf_1": acf_1,
    "variance_ratio": variance_ratio,
    "cosinor_amplitude_weekly": cosinor_amplitude,
}

DEFAULT_AGGREGATIONS.update(_REDUCTION_AGGREGATIONS)

for _name, _agg in _REDUCTION_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# --- Two-window distributional drift ---

_DRIFT_AGGREGATIONS = {
    "kl_drift": kl_drift,
    "wasserstein_drift": wasserstein_drift,
}

DEFAULT_AGGREGATIONS.update(_DRIFT_AGGREGATIONS)

for _name, _agg in _DRIFT_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# --- Predicate-driven aggregators ---

_PREDICATE_AGGREGATIONS = {
    "right_censoring_indicator": right_censoring_indicator,
    "cross_type_latency": cross_type_latency,
    "first_passage_time": first_passage_time,
}

DEFAULT_AGGREGATIONS.update(_PREDICATE_AGGREGATIONS)

for _name, _agg in _PREDICATE_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# --- Spatial aggregators (plain-SQL) ---

_SPATIAL_AGGREGATIONS = {
    "distance_travelled": distance_travelled,
    "radius_of_gyration": radius_of_gyration,
    "spatial_std": spatial_std,
    "bbox_area": bbox_area,
}

DEFAULT_AGGREGATIONS.update(_SPATIAL_AGGREGATIONS)

for _name, _agg in _SPATIAL_AGGREGATIONS.items():
    register_aggregation(_name, _agg)


# TODO: trend

# def fixed_width_buckets(self, target, variable, n_buckets=5):
#     return {f'"{str.upper({n_buckets})}_BUCKETS({ numeric_var })"': {'query': f'width_bucket({ numeric_var }, min({ numeric_var }), max({ numeric_var }), { n_buckets })'}}
