# coding: utf-8

class Aggregator:
    """
    Base class for aggregation functions

    From the PostgreSQL docs:

    '...represents the application of an aggregate
    function across the rows selected by a query.
    An aggregate function reduces multiple inputs to a single output value,
    such as the sum or average of the inputs.'

    """
    def __init__(self, name, func, input_type, output_type, distinct = False, order_by=None, filter=None, stackable=True):
        self.name = name
        self.func = func
        self.input_type = input_type
        self.output_type = output_type
        self.distinct = distinct # If DISTINCT is specified in addition to an order_by_clause,
                                 # then all the ORDER BY expressions must match regular arguments of
                                 # the aggregate; that is, you cannot sort on
                                 # an expression that is not included in the DISTINCT list .
        self.order_by = order_by # ' ORDER BY expression' i.e. input columns
        self.filter = filter  # filter' FILTER WHERE :filter'
        self.stackable = stackable


    def __call__(self, parent, child, feature):
        return self.func(parent, child, feature, filter)


# ## Agregations
# def _aggregator(self, target, entity, function, variable, input_type, output_type):
#     return Feature(name=f'{ str.upper(function) }({entity.alias}.{ variable.name })',
#               definition=f'{ function }({ variable.name })',
#               type=output_type)

# sum = partialmethod(_aggregator, function='sum', input_type='numeric', output_type='numeric')
# min = partialmethod(_aggregator, function='min', input_type='numeric', output_type='numeric')
# max = partialmethod(_aggregator, function='max', input_type='numeric', output_type='numeric')
# mean = partialmethod(_aggregator, function='avg', input_type='numeric', output_type='numeric')
# stddev = partialmethod(_aggregator, function='stddev', input_type='numeric', output_type='numeric')
# count = partialmethod(_aggregator, function='count', input_type='categorical', output_type='numeric')

# def z_score(self, target, variable):
#     return {f'"Z_SCORE({{ numeric_var }})"': {'definition': f'abs({{ numeric_var }} - avg({{ numeric_var }})) / stddev({{ numeric_var }})'}}

# def skewness(self, target, variable):
#     return {f'"SKEWNESS({{ numeric_var }})"': {'query':f'({{ numeric_var }} - avg({{ numeric_var }})) / stddev({{ numeric_var }})**3'}}

# def kurtosis(self, target, variable):
#     return {f'"KURTOSIS({{ numeric_var }})"': {'query':f'({{ numeric_var }} - avg({{ numeric_var }})) / stddev({{ numeric_var }})**4'}}

# def median(self, target, variable):
#     return {f'"MEDIAN({{ numeric_var }})"': {'query': f'percentile_cont(0.5) within group (order by {{ numeric_var }})'}}

# def fixed_width_buckets(self, target, variable, n_buckets=5):
#     return {f'"{str.upper({n_buckets})}_BUCKETS({{ numeric_var }})"': {'query': f'width_bucket({{ numeric_var }}, min({{ numeric_var }}), max({{ numeric_var }}), {{ n_buckets }})'}}

# def min_max_scale(self, target, variable):
#     return {f'"MIN_MAX({{ numeric_var }})': {'query': f'1.0*({{ numeric_var }} - min({{ numeric_var }})/(max({{ numeric_var }}) - min({{ numeric_var }}))'}}

# def num_unique(self, target, variable):
#     return {f'"NUM_UNIQUE({{ categorical_var }})"': {'query': f'count( distinct {{ categorical_var }})'}}

# def mode(self, target, variable):
#     return {f'"MODE({categorical_var})"': {'query': f'mode() within group (order by {{ categorical_var }})'}}


class OrderedSetAggregator(Aggregator):
    """
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
    def __init__(self, name, func, input_type, output_type,order_by=None, filter=None, direct_argument=None, stackable=True):
        super().__init__(name, func, input_type, output_type, order_by, filter, stackable)

    def __call__(self, parent, child, feature):
        return self.func(parent, child, feature, filter)
