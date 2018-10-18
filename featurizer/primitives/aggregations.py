# coding: utf-8

from .abstractions import Feature

class Aggregator:
    """
    Base class for aggregation functions

    From the PostgreSQL docs:

    '...represents the application of an aggregate
    function across the rows selected by a query.
    An aggregate function reduces multiple inputs to a single output value,
    such as the sum or average of the inputs.'

    """
    def __init__(self, name, aggregate=None, input_types=['numeric'], output_type='numeric', distinct = False, order_by=None, filter=None, stackable=True):
        self.name = name
        self.aggregate = aggregate if aggregate else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.distinct = distinct # If DISTINCT is specified in addition to an order_by_clause,
                                 # then all the ORDER BY expressions must match regular arguments of
                                 # the aggregate; that is, you cannot sort on
                                 # an expression that is not included in the DISTINCT list .
        self.order_by = order_by # ' ORDER BY expression' i.e. input columns
        self.filter = filter  # filter' FILTER WHERE :filter'
        self.stackable = stackable

    @staticmethod
    def _build_name(name, feature):
        name = f'{ str.upper(name) }({feature.entity.alias}.{feature.name})'
        return f'''"{name.replace('"', '')}"'''

    def _build_aggregate_expression(self, feature):
        expression = feature.name
        aggregate_expression = [f"{self.aggregate}({'distinct' if self.distinct else ''} {expression}"]
        if self.order_by and feature.sort:
            # order by clause
            aggregate_expression.append(f"order by {feature.sort})")
        else:
            aggregate_expression.append(")")
        if self.filter and feature.specials:
            # filter by clause
            aggregate_expression.append(f" filter (where {feature.name} = {feature.specials}) ")

        return ' '.join(aggregate_expression)

    def __call__(self, parent, child, feature):
        if feature.type not in self.input_types:
            # We don't do anything
            agg_feature = None
        else:
            agg_feature = Feature(name = self._build_name(self.name, feature),
                                  type=self.output_type,
                                  definition=self._build_aggregate_expression(feature),
                                  parents = feature,
                                  entity = parent,
                                  stack_depth=feature.stack_depth + 1)
        return agg_feature

class Zscore(Aggregator):
    def __init__(self):
        super().__init__(name='zscore')

    def _build_aggregate_expression(self,feature):
        return f"(abs({ feature.name } - avg({ feature.name })) / stddev({ feature.name })"


class Skewness(Aggregator):
    def __init__(self):
        super().__init__(name='skewness')

    def _build_aggregate_expression(self,feature):
        return f"({ feature.name } - avg({ feature.name })) / stddev({ feature.name })**3"


class Kurtosis(Aggregator):
    def __init__(self):
        super().__init__(name='kurtosis')

    def _build_aggregate_expression(self,feature):
        return f"({ feature.name } - avg({ feature.name })) / stddev({ feature.name })**4"

class MinMaxScale(Aggregator):
    def __init__(self):
        super().__init__(name='min_max_scale')

    def _build_aggregate_expression(self,feature):
        return f"1.0*({ feature.name } - min({ feature.name })/(max({ feature.name }) - min({ feature.name }))"

class AverageDeviation(Aggregator):
    def __init__(self):
        super().__init__(name='mean_deviation')

    def _build_aggregate_expression(self,feature):
        return f"(sum(abs({feature.name} - avg({feature.name}))) / count({feature.name}))"


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
        super().__init__(name='harmonic_mean')

    def _build_aggregate_expression(self,feature):
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
        super().__init__(name='geometric_mean')

    def _build_aggregate_expression(self,feature):
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

sum = Aggregator(name='sum')
min = Aggregator(name='min')
max = Aggregator(name='max')
mean = Aggregator(name='mean', aggregate='avg')
stddev = Aggregator(name='stddev')
var = Aggregator(name='variance')
count = Aggregator(name='count', input_types=['categorical', 'index'])
all = Aggregator(name='all', aggregate='bool_and', input_types=['boolean'], output_type='boolean')
any = Aggregator(name='anu', aggregate='bool_or', input_types=['boolean'], output_type='boolean')
nunique = Aggregator(name='nunique', aggregate='count', input_types=['categorical', 'index'], distinct=True)
min_max_scale = MinMaxScale()
mean_deviation = AverageDeviation()
z_score = Zscore()
skewness = Skewness()
kurtosis = Kurtosis()
harmonic_mean = HarmonicMean()
geometric_mean = GeometricMean()


class OrderedSetAggregator(Aggregator):
    """
    There is a subclass of aggregate functions called ordered-set aggregates for which an
    order_by_clause is required, usually because the aggregate's computation is only sensible
    in terms of a specific ordering of its input rows. Typical examples of ordered-set aggregates
    include rank and percentile calculations.
    For an ordered-set aggregate, the order_by_clause is written inside WITHIN GROUP (...),
    as shown in the final syntax alternative above.
    The expressions in the order_by_clause are evaluated once per input row just
    like regular aggregate arguments, sorted as per the order_by_clause's requirements,
    and fed to the aggregate function as input arguments.
    (This is unlike the case for a non-WITHIN GROUP order_by_clause, which is
    not treated as argument(s) to the aggregate function.)
    The argument expressions preceding WITHIN GROUP, if any, are called direct
    arguments to distinguish them from the aggregated arguments listed in the order_by_clause.
    Unlike regular aggregate arguments, direct arguments are evaluated only once
    per aggregate call, not once per input row. This means that they can contain variables
    only if those variables are grouped by GROUP BY; this restriction is the same as
    if the direct arguments were not inside an aggregate expression at all.
    Direct arguments are typically used for things like percentile fractions,
    which only make sense as a single value per aggregation calculation.
    The direct argument list can be empty; in this case, write just () not (*).
    """
    def __init__(self, name, aggregate=None, direct_argument=None, input_types=['numeric'], output_type='numeric', filter=None, stackable=True):
        self.order_by = True
        self.direct_argument = direct_argument
        super().__init__(name, aggregate, input_types, output_type, self.order_by, filter, stackable)

    def _build_aggregate_expression(self, feature):
        expression = feature.name
        if self.direct_argument:
            aggregate_expression = [f"{self.aggregate}({self.direct_argument})"]
        else:
            aggregate_expression = [f"{self.aggregate}()"]
        aggregate_expression.append(f"within group(order by {expression})")
        if self.filter and feature.specials:
            # filter by clause
            aggregate_expression.append(f" filter (where {feature.name} = {feature.specials}) ")

        return ' '.join(aggregate_expression)

median = OrderedSetAggregator(name='median', aggregate='percentile_cont', direct_argument=0.5)
mode = OrderedSetAggregator(name='mode', input_types=['categorical'])


# TODO: trend

# TODO: first, last Trasnfromaciones?
# TODO: Average time between events

# TODO: def daily_average():
#     pass

# TODO: def weekly_average():
#     pass

# TODO: def monthly_average():
#     pass

# TODO: percentage of true

# TODO: Above the average


# def fixed_width_buckets(self, target, variable, n_buckets=5):
#     return {f'"{str.upper({n_buckets})}_BUCKETS({ numeric_var })"': {'query': f'width_bucket({ numeric_var }, min({ numeric_var }), max({ numeric_var }), { n_buckets })'}}
