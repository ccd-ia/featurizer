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

from .abstractions import Feature
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
        name = f"{ str.upper(name) }({feature.entity.alias}.{feature.name}"
        interval = f"|interval={interval})" if interval else ")"
        name = name + interval
        return f'''"{name.replace('"', '')}"'''

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
                f" filter (where {daterange} @>  {event_date}) "
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
        return (
            f"(abs({ feature.name } - avg({ feature.name })) / stddev({ feature.name })"
        )


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
        return (
            f"({ feature.name } - avg({ feature.name })) / stddev({ feature.name })**3"
        )


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
        return (
            f"({ feature.name } - avg({ feature.name })) / stddev({ feature.name })**4"
        )


class MinMaxScale(Aggregator):
    """Min-max normalization aggregation.

    Scales values to a 0-1 range based on min and max.
    Useful for comparing features with different scales.

    SQL: (value - MIN(value)) / (MAX(value) - MIN(value))
    """

    def __init__(self):
        super().__init__(name="min_max_scale")

    def _build_aggregate_expression(self, feature, interval=None):
        return f"1.0*({ feature.name } - min({ feature.name })/(max({ feature.name }) - min({ feature.name }))"


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
            filters.append(f"{daterange} @> {event_date}")
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
            filt = f" filter (where {daterange} @> {event_date})"
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
            filt = f" filter (where {daterange} @> {event_date})"
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
            filt = f" filter (where {daterange} @> {event_date})"
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
            filt = f" filter (where {daterange} @> {event_date})"
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
            filt = f" filter (where {daterange} @> {event_date})"
        return (
            f"EXTRACT(EPOCH FROM max({feature.name}){filt} - min({feature.name}){filt})"
        )


time_span = TimeSpan()

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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_date}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_date}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            event_date = feature.entity.temporal_ix.name
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_date}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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
        interval_filter = ""
        if interval and feature.entity and feature.entity.temporal_ix:
            daterange = f"daterange((aod.as_of_date - interval '{interval}')::date, aod.as_of_date::date, '[]')"
            interval_filter = f" and {daterange} @> sub.{event_col}"
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


# TODO: trend

# def fixed_width_buckets(self, target, variable, n_buckets=5):
#     return {f'"{str.upper({n_buckets})}_BUCKETS({ numeric_var })"': {'query': f'width_bucket({ numeric_var }, min({ numeric_var }), max({ numeric_var }), { n_buckets })'}}
