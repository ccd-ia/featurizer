# coding: utf-8

"""Transformation primitives and registry wiring.

Unary transformers should register via `register_transformer` to remain discoverable
by the featurizer without requiring manual imports.
"""

from .abstractions import Entity, Feature
from .utils import register_transformer
from typing import Callable, Iterable, Optional, Sequence, Tuple

class Transformer:
    """
    Base class for transformation functions

    From the PostgreSQL docs:
    The syntax for a function call is the name of a function (possibly qualified
    with a schema name), followed by its argument list enclosed in parentheses:

    """
    def __init__(self, name, transformer=None, input_types=['numeric'], output_type='numeric', stackable=True):
        self.name = name
        self.transformer = transformer if transformer is not None else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.stackable = stackable

    @staticmethod
    def _build_name(name, feature):
        name = f'{ str.upper(name) }({feature.entity.alias}.{feature.name})'
        return f'''"{name.replace('"', '')}"'''

    def _build_transformer_call(self, feature):
        return f""" {self.transformer}({feature.name}) """

    def __call__(self, parent, feature):
        if feature.type == 'key' or feature.type not in self.input_types:
            return feature
        return Feature(
            name=self._build_name(self.name, feature),
            type=self.output_type,
            definition=self._build_transformer_call(feature),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )

abs = Transformer(name='abs')
exp = Transformer(name='exp')
ln = Transformer(name='ln')
log = Transformer(name='log')
# log2 = log(2, x)
# power = power(a, b) # a^b
sqrt = Transformer(name='sqrt')
cbrt = Transformer(name='cbrt')
sign = Transformer(name='sign')
num_chars = Transformer(name='num_chars', transformer='char_lenght', input_types=['text'])
# random = Transformer(name='random')  # without arguments  Should setseed(number)
ceil = Transformer(name='ceil')
floor = Transformer(name='floor')
trunc = Transformer(name='trunc')
# #round = partialmethod(_unitary, function='round')
# # power

class Identity(Transformer):
    def __init__(self):
        super().__init__(name='identity', input_types=['numeric', 'categoric', 'text'])

    @staticmethod
    def _build_name(name, feature):
        name = f'{feature.entity.alias}.{feature.name}'
        return f'''"{name.replace('"', '')}"'''

    def _build_transformer_call(self, feature):
        return f""" {feature.name} """

    def __call__(self, parent, feature):
        return feature

identity = Identity()


class DateTransformer(Transformer):
    def __init__(self, name, date_part):
        self.date_part = date_part
        super().__init__(name, input_types=['date', 'timestamp', 'index'], output_type='categorical', stackable=True)

    def _build_transformer_call(self, feature):
        return f"to_char({ feature.name }, '{self.date_part}')"

    def __call__(self, parent, feature):
        if feature.type == 'key':
            return feature
        temporal_ix = getattr(feature.entity, "temporal_ix", None)
        if feature.type == 'index' and feature is not temporal_ix:
            return feature
        if feature.type not in self.input_types:
            return feature
        return Feature(
            name=self._build_name(self.name, feature),
            type=self.output_type,
            definition=self._build_transformer_call(feature),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )

day = DateTransformer(name='day', date_part='day')
dow = DateTransformer(name='dow', date_part='ID')  # Iso week: Monday (1) to Sunday (7)
dom = DateTransformer(name='dom', date_part='DD')
doy = DateTransformer(name='doy', date_part='DDD')
year = DateTransformer(name='year', date_part='YYYY')
month = DateTransformer(name='month', date_part='M')
hour = DateTransformer(name='hour', date_part='HH24')
century = DateTransformer(name='century', date_part='CC')
quarter = DateTransformer(name='quarter', date_part='Q')
week = DateTransformer(name='week', date_part='W')
week_of_year = DateTransformer(name='week_of_year', date_part='WW')
time_zone = DateTransformer(name='tz', date_part='TZ')
tz_offset = DateTransformer(name='tz_offset', date_part='OF')


class HourlyBinning(Transformer):
    def __init__(self):
        super().__init__(name='hourly_bin', transformer=None, input_types=['date', 'timestamp'], output_type='categorical', stackable=True)

    def _build_transformer_call(self, feature):
        return f'''
        (
        case
        when extract(hour from { feature.name }) <@ int4range(0,5) then 'night'
        when extract(hour from { feature.name }) <@ int4range(5,8) then 'early_morning'
        when extract(hour from { feature.name }) <@ int4range(8,11) then 'morning'
        when extract(hour from { feature.name }) <@ int4range(11,14) then 'midday'
        when extract(hour from { feature.name }) <@ int4range(14,19) then 'afternoon'
        when extract(hour from { feature.name }) <@ int4range(19,22) then 'evening'
        when extract(hour from { feature.name }) <@ int4range(22,24) then 'night'
        )
        '''

class DailyBinning(Transformer):
    def __init__(self):
        super().__init__(name='daily_bin', transformer=None, input_types=['date', 'timestamp'], output_type='categorical', stackable=True)

    def _build_transformer_call(self, feature):
        return f'''
        (
        case
        when to_char({feature.name},'ID')::smallint <@ int4range(0,5) then 'weekday'
        when to_char({feature.name},'ID')::smallint <@ int4range(5,7) then 'weekday'
        )
        '''

hourly_binning = HourlyBinning()
daily_binning = DailyBinning()


class CyclicalDateTransformer(DateTransformer):
    def __init__(self, name, date_part, period, adjust = True):
        self.period = period
        self.adjust = adjust
        super().__init__(name=name, date_part=date_part)

    def _build_transformer_call(self, feature, trig_function):
        if self.adjust:
            return f"""{trig_function}((to_char({feature.name}, '{self.date_part}')::smallint - 1)*(2*pi()/{self.period}))"""
        else:
            return f"""{trig_function}((to_char({feature.name}, '{self.date_part}')::smallint)*(2*pi()/{self.period}))"""

    def __call__(self, parent, feature):
        if feature.type == 'key' or feature.type not in self.input_types:
            return feature
        return [
            Feature(
                name=self._build_name(self.name + '_sin', feature),
                type=self.output_type,
                definition=self._build_transformer_call(feature, trig_function='sin'),
                parents=feature,
                entity=parent,
                stack_depth=feature.stack_depth + 1,
            ),
            Feature(
                name=self._build_name(self.name + '_cos', feature),
                type=self.output_type,
                definition=self._build_transformer_call(feature, trig_function='cos'),
                parents=feature,
                entity=parent,
                stack_depth=feature.stack_depth + 1,
            ),
        ]


cyclic_hour = CyclicalDateTransformer(name='cyclic_hour', date_part='HH24', period=24, adjust=False)
cyclic_month = CyclicalDateTransformer(name='cyclic_month', date_part='MM', period=12)
cyclic_day = CyclicalDateTransformer(name='cyclic_hour', date_part='D', period=7)


class WindowFunctionTransformer:
    """
    A window function call represents the application of an aggregate-like
    function over some portion of the rows selected by a query.
    Unlike non-window aggregate calls, this is not tied to grouping of the
    selected rows into a single output row — each row remains separate in the query output.
    However the window function has access to all the rows that would
    be part of the current row's group according to the grouping specification
    (PARTITION BY list) of the window function call.
    """
    def __init__(
        self,
        name,
        function=None,
        input_types=['numeric'],
        output_type='numeric',
        order_by: Optional[Callable[[Feature], Optional[str]]] = None,
        filter=None,
        frame: Optional[Tuple[str, str]] = None,
        stackable=True,
        extra_args: Optional[Sequence[Callable[[Feature], str] | str]] = None,
    ):
        self.name = name
        self.function = function if function else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.order_by = order_by
        self.filter = filter  # filter' FILTER WHERE :filter'
        self.stackable = stackable
        self.frame = frame
        self.extra_args: Tuple[Callable[[Feature], str] | str, ...] = tuple(extra_args or ())

    @staticmethod
    def _build_name(name, feature):
        name = f'{ str.upper(name) }({feature.entity.alias}.{feature.name})'
        return f'''"{name.replace('"', '')}"'''

    def _resolve_order_by(self, feature: Feature) -> Optional[str]:
        if callable(self.order_by):
            return self.order_by(feature)
        return self.order_by

    def _resolve_args(self, feature: Feature) -> Sequence[str]:
        args = []
        for arg in self.extra_args:
            if callable(arg):
                args.append(arg(feature))
            else:
                args.append(str(arg))
        return args

    def _build_window_function_call(self, parent, feature):
        expression = feature.name
        partition = parent.id.name
        if not partition:
            return None
        window_args = [expression] + list(self._resolve_args(feature))
        window_call = [f"{ self.function }({ ', '.join(window_args) })"]
        if self.filter and feature.specials:
            # filter by clause
            window_call.append(f" filter (where {feature.name} = {feature.specials}) ")
        window_call.append(f" over (partition by { partition }")
        order_clause = self._resolve_order_by(feature)
        if order_clause:
            window_call.append(f" order by { order_clause }")
        if self.frame:
            start, end = self.frame
            window_call.append(f" rows between {start} and {end}")
        window_call.append(")")

        return ' '.join(window_call)

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        definition = self._build_window_function_call(parent, feature)
        if not definition:
            return None
        return Feature(
            name=self._build_name(self.name, feature),
            type=self.output_type,
            definition=definition,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


def _temporal_ordering(feature: Feature) -> Optional[str]:
    temporal_ix = getattr(feature.entity, "temporal_ix", None)
    if temporal_ix is None:
        return None
    return temporal_ix.name


def _build_temporal_window(function: str, parent: Entity, feature: Feature, *, args: Iterable[str] = (), frame: Optional[Tuple[str, str]] = None) -> Optional[str]:
    partition = parent.id.name if parent.id else None
    if partition is None:
        return None
    order_by = _temporal_ordering(feature)
    if order_by is None:
        return None
    args_sql = ", ".join([feature.name] + list(args))
    window_bits = [f"partition by {partition}", f"order by {order_by}"]
    if frame:
        start, end = frame
        window_bits.append(f"rows between {start} and {end}")
    window_clause = " ".join(window_bits)
    return f"{function}({args_sql}) over ({window_clause})"


def _frame_for_window(window: int) -> Optional[Tuple[str, str]]:
    if window <= 1:
        return None
    return (f"{window - 1} preceding", "current row")


def _build_percentile_window(
    parent: Entity,
    feature: Feature,
    percentile: float,
    frame: Optional[Tuple[str, str]],
) -> Optional[str]:
    partition = parent.id.name if parent.id else None
    if partition is None:
        return None
    order_by = _temporal_ordering(feature)
    if order_by is None:
        return None
    frame_clause = ""
    if frame:
        start, end = frame
        frame_clause = f" rows between {start} and {end}"
    return (
        f"percentile_cont({percentile}) within group (order by {feature.name}) "
        f"over (partition by {partition} order by {order_by}{frame_clause})"
    )


cum_sum = WindowFunctionTransformer(name='cum_sum', function='sum', order_by=_temporal_ordering)
cum_mean = WindowFunctionTransformer(name='cum_mean', function='avg', order_by=_temporal_ordering)
cum_max = WindowFunctionTransformer(name='cum_max', function='max', order_by=_temporal_ordering)
cum_min = WindowFunctionTransformer(name='cum_min', function='min', order_by=_temporal_ordering)
cum_count = WindowFunctionTransformer(name='cum_count', function='count', input_types=['categorical', 'index'], order_by=_temporal_ordering)

# All of the following act on the window frame, not in the partition
# TODO: Include any or *
first = WindowFunctionTransformer(name='first', function='first_value', input_types=['categorical', 'index', 'numeric', 'date'], order_by=_temporal_ordering)
last = WindowFunctionTransformer(name='last', function='last_value', input_types=['categorical', 'index', 'numeric'], order_by=_temporal_ordering)
#nth_value = WindowFunctionTransformer(name='nth_value', function='', input_types=['categorical', 'index', 'numeric', 'date'])

previous = WindowFunctionTransformer(name='previous', function='lag', order_by=_temporal_ordering)


class Diff:
    def __init__(self, name, input_types=['numeric'], output_type='numeric'):
        self.name = name
        self.input_types = input_types
        self.output_type = output_type

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        lag_expr = _build_temporal_window("lag", parent, feature)
        if lag_expr is None:
            return None
        return Feature(
            name=f"\"{self.name.upper()}({feature.entity.alias}.{feature.name})\"",
            type=self.output_type,
            definition=f"{feature.name} - {lag_expr}",
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


diff = Diff(name='diff')
time_since_previous = Diff(name='time_since_previous', input_types=['date', 'timestamp'], output_type='date')


class DistributionTransformer(WindowFunctionTransformer):
    def __init__(self, name, function=None, arg_func=None, input_types=['numeric'], output_type='numeric', order_by=None, frame=None, stackable=True):
        #  Only window functions that are aggregates accept a FILTER clause.
        filter = False
        self.arg_func = arg_func
        super().__init__(name, function, input_types, output_type, order_by, filter, frame, stackable)

    def _build_window_function_call(self, parent, feature):
        partition = parent.id.name
        if not partition:
            return None
        pieces = []
        if self.arg_func:
            pieces.append(f"{ self.function }({ self.arg_func })")
        else:
            pieces.append(f"{ self.function }()")
        pieces.append(f" over (partition by { partition }")
        order_clause = self._resolve_order_by(feature) if hasattr(self, "_resolve_order_by") else None
        if order_clause:
            pieces.append(f" order by { order_clause }")
        if self.frame:
            start, end = self.frame
            pieces.append(f" rows between {start} and {end}")
        pieces.append(")")
        return ' '.join(pieces)


cdf = DistributionTransformer(name='cdf', function='cum_dist', order_by=_temporal_ordering)
## relative rank of the current row: (rank - 1) / (total partition rows - 1)
percent_rank = DistributionTransformer(name='percent_rank', order_by=_temporal_ordering)
ntile = DistributionTransformer(name='ntile', arg_func=5, order_by=_temporal_ordering)

class LagTransformer:
    def __init__(self, periods: int):
        self.periods = periods
        self.name = f"lag_{periods}"
        self._input_types = ['numeric', 'categorical', 'date', 'timestamp', 'index']

    def __call__(self, parent, feature):
        if feature.type == 'key':
            return feature
        if feature.type not in self._input_types:
            return feature
        expression = _build_temporal_window("lag", parent, feature, args=[str(self.periods)])
        if expression is None:
            return None
        return Feature(
            name=f"\"LAG_{self.periods}({feature.entity.alias}.{feature.name})\"",
            type=feature.type,
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class RollingStatisticTransformer:
    def __init__(self, label: str, function: str, window: int):
        self.label = label
        self.function = function
        self.window = window
        self.name = f"{label}_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        frame = _frame_for_window(self.window)
        expression = _build_temporal_window(self.function, parent, feature, frame=frame)
        if expression is None:
            return None
        return Feature(
            name=f"\"{self.label.upper()}_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class RollingMedianTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"rolling_median_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        frame = _frame_for_window(self.window)
        expression = _build_percentile_window(parent, feature, 0.5, frame)
        if expression is None:
            return None
        return Feature(
            name=f"\"ROLLING_MEDIAN_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class RollingIQRTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"rolling_iqr_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        frame = _frame_for_window(self.window)
        p75 = _build_percentile_window(parent, feature, 0.75, frame)
        p25 = _build_percentile_window(parent, feature, 0.25, frame)
        if not p75 or not p25:
            return None
        expression = f"({p75}) - ({p25})"
        return Feature(
            name=f"\"ROLLING_IQR_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class ExponentialMovingAverageTransformer:
    def __init__(self, window: int, decay: float):
        self.window = window
        self.decay = decay
        self.name = f"ema_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        partition = parent.id.name if parent.id else None
        order_by = _temporal_ordering(feature)
        if not partition or not order_by:
            return None
        frame = _frame_for_window(self.window)
        frame_clause = ""
        if frame:
            start, end = frame
            frame_clause = f" rows between {start} and {end}"
        timestamp_expr = f"(extract(epoch from {order_by}) / 86400.0)"
        weight_expr = f"exp({self.decay} * {timestamp_expr})"
        base_window = f"partition by {partition} order by {order_by}{frame_clause}"
        numerator = f"sum({feature.name} * {weight_expr}) over ({base_window})"
        denominator = f"sum({weight_expr}) over ({base_window})"
        expression = f"{numerator} / NULLIF({denominator}, 0)"
        return Feature(
            name=f"\"EMA_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class HoltWintersLevelTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"holt_winters_level_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        frame = _frame_for_window(self.window)
        expression = _build_temporal_window("avg", parent, feature, frame=frame)
        if expression is None:
            return None
        return Feature(
            name=f"\"HOLT_WINTERS_LEVEL_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class HoltWintersTrendTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"holt_winters_trend_{window}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        order_by = _temporal_ordering(feature)
        if order_by is None:
            return None
        frame = _frame_for_window(self.window)
        expression = _build_temporal_window(
            "regr_slope",
            parent,
            feature,
            args=[order_by],
            frame=frame,
        )
        if expression is None:
            return None
        return Feature(
            name=f"\"HOLT_WINTERS_TREND_{self.window}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class PercentageChangeTransformer:
    def __init__(self, periods: int):
        self.periods = periods
        self.name = f"pct_change_{periods}"

    def __call__(self, parent, feature):
        if feature.type != 'numeric':
            return feature
        lag_expr = _build_temporal_window("lag", parent, feature, args=[str(self.periods)])
        if lag_expr is None:
            return None
        expression = f"""
        case
        when {lag_expr} is null or {lag_expr} = 0 then null
        else ({feature.name} - {lag_expr}) / {lag_expr}
        end
        """
        return Feature(
            name=f"\"PCT_CHANGE_{self.periods}({feature.entity.alias}.{feature.name})\"",
            type='numeric',
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


class BinaryTransformer(Transformer):
    def __init__(self, name, operation, input_types=['numeric'], output_type='numeric', stackable=True):
        self.operation = operation
        self.transformer = None
        super().__init__(name, self.transformer, input_types, output_type, stackable)

    @staticmethod
    def _build_name(name, feature1, feature2):
        name = f'{ str.upper(name) }({feature1.entity.alias}.{feature1.name}, {feature2.entity.alias}.{feature2.name})'
        return f'''"{name.replace('"', '')}"'''

    def _build_transformer_call(self, feature1, feature2):
        return f"{feature1.entity.alias}.{ feature1.name } { self.operation }  {feature2.entity.alias}.{ feature2.name }"

    def __call__(self, parent, feature1, feature2):
        if feature1.type not in self.input_types or feature2.type not in self.input_types:
            # Don't do anything
            trans_feature = None
        else:
            trans_feature = Feature(name=self._build_name(self.name, feature1, feature2),
                                    type=self.output_type,
                                    definition=self._build_transformer_call(feature1, feature2),
                                    parents = [feature1, feature2],
                                    entity = parent,
                                    stack_depth=feature1.stack_depth + 1)

        return trans_feature


add = BinaryTransformer(name='add', operation='+')
difference = BinaryTransformer(name='subs', operation='-')
multiply = BinaryTransformer(name='mul', operation='*')
ratio = BinaryTransformer(name='div', operation='/')
modulo = BinaryTransformer(name='mod', operation='%')
exponentiation = BinaryTransformer(name='exponentiation', operation='^')
bitwise_and = BinaryTransformer(name='bitwise_and', operation='&')
bitwise_or = BinaryTransformer(name='bitwise_or', operation='|')
bitwise_xor = BinaryTransformer(name='bitwise_xor', operation='#')
bitwise_shift_left = BinaryTransformer(name='bitwise_shift_left', operation='<<')
bitwise_shift_right = BinaryTransformer(name='bitwise_shift_right', operation='>>')

# Boolean
boolean_and = BinaryTransformer(name='and', operation='and', input_types=['boolean'], output_type='boolean')
boolean_or = BinaryTransformer(name='or', operation='or', input_types=['boolean'], output_type='boolean')

# Logical
eq = BinaryTransformer(name='eq', operation='=')
neq = BinaryTransformer(name='neq', operation='!=')
lt = BinaryTransformer(name='lt', operation='<')
gt = BinaryTransformer(name='gt', operation='>')
le = BinaryTransformer(name='le', operation='<=')
ge = BinaryTransformer(name='ge', operation='=>')
time_since = BinaryTransformer(name='time_since', operation='-', input_types=['date', 'timestamp'], output_type='date')

class IsNull(Transformer):
    def __init__(self):
        name = 'is_null'
        super().__init__(name, transformer=None, input_types=['numeric', 'categorical', 'date'], output_type='boolean', stackable=True)

    def _build_transformer_call(self, feature):
        return f"({ feature.name } is null)"

class IsInArray(Transformer):
    def __init__(self):
        name = 'in_array'
        super().__init__(name, transformer=None, input_types=['numeric', 'categorical', 'date'], output_type='boolean', stackable=True)

    def _build_transformer_call(self, feature, an_array):
        return f"({feature.name} = ANY (ARRAY {an_array})"

    def __call__(self, parent, feature, an_array):
        if feature.type not in self.input_types:
            return feature
        return Feature(
            name=self._build_name(self.name, feature),
            type=self.output_type,
            definition=self._build_transformer_call(feature, an_array),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )


isnull = IsNull()
inarray = IsInArray()


DEFAULT_TRANSFORMERS = {
    "identity": identity,
    "abs": abs,
    "exp": exp,
    "ln": ln,
    "log": log,
    "sqrt": sqrt,
    "cbrt": cbrt,
    "sign": sign,
    "num_chars": num_chars,
    "ceil": ceil,
    "floor": floor,
    "trunc": trunc,
    "day": day,
    "dow": dow,
    "dom": dom,
    "doy": doy,
    "year": year,
    "month": month,
    "hour": hour,
    "century": century,
    "quarter": quarter,
    "week": week,
    "week_of_year": week_of_year,
    "tz": time_zone,
    "tz_offset": tz_offset,
    "hourly_bin": hourly_binning,
    "daily_bin": daily_binning,
    "cyclic_hour": cyclic_hour,
    "cyclic_month": cyclic_month,
    "cyclic_day": cyclic_day,
    "cum_sum": cum_sum,
    "cum_mean": cum_mean,
    "cum_max": cum_max,
    "cum_min": cum_min,
    "cum_count": cum_count,
    "first": first,
    "last": last,
    "previous": previous,
    "diff": diff,
    "time_since_previous": time_since_previous,
    "cdf": cdf,
    "percent_rank": percent_rank,
    "ntile": ntile,
    "is_null": isnull,
    "in_array": inarray,
}

for _name, _transformer in DEFAULT_TRANSFORMERS.items():
    register_transformer(_name, _transformer)

for _periods in (1, 3, 7):
    _lag_transformer = LagTransformer(_periods)
    DEFAULT_TRANSFORMERS[_lag_transformer.name] = _lag_transformer
    register_transformer(_lag_transformer.name, _lag_transformer)

for _window in (3, 7, 14):
    _rolling_mean = RollingStatisticTransformer("rolling_mean", "avg", _window)
    _rolling_std = RollingStatisticTransformer("rolling_std", "stddev", _window)
    DEFAULT_TRANSFORMERS[_rolling_mean.name] = _rolling_mean
    DEFAULT_TRANSFORMERS[_rolling_std.name] = _rolling_std
    register_transformer(_rolling_mean.name, _rolling_mean)
    register_transformer(_rolling_std.name, _rolling_std)

for _window in (5, 7):
    _rolling_median = RollingMedianTransformer(_window)
    DEFAULT_TRANSFORMERS[_rolling_median.name] = _rolling_median
    register_transformer(_rolling_median.name, _rolling_median)

for _window in (7, 14):
    _rolling_iqr = RollingIQRTransformer(_window)
    DEFAULT_TRANSFORMERS[_rolling_iqr.name] = _rolling_iqr
    register_transformer(_rolling_iqr.name, _rolling_iqr)

for _window, _decay in ((7, 0.25), (14, 0.15)):
    _ema = ExponentialMovingAverageTransformer(_window, _decay)
    DEFAULT_TRANSFORMERS[_ema.name] = _ema
    register_transformer(_ema.name, _ema)

for _window in (7, 14):
    _hw_level = HoltWintersLevelTransformer(_window)
    _hw_trend = HoltWintersTrendTransformer(_window)
    DEFAULT_TRANSFORMERS[_hw_level.name] = _hw_level
    DEFAULT_TRANSFORMERS[_hw_trend.name] = _hw_trend
    register_transformer(_hw_level.name, _hw_level)
    register_transformer(_hw_trend.name, _hw_trend)

for _periods in (1, 3):
    _pct = PercentageChangeTransformer(_periods)
    DEFAULT_TRANSFORMERS[_pct.name] = _pct
    register_transformer(_pct.name, _pct)

#percentage above avg
#percentage trues


# def _polynomial(self, target, x_1, x_2):
#     return f'(1 + { x_1 } + { x_2 } + { x_1 }*{ x_2 } + { x_1 }**2 + { x_2 }**2)'

# def _polynomial(self, target, x, coefs):
#     poly =  ' '.join(["{:+d}*{:s}**{:d}".format(a,x,n) for n, a in enumerate(coefs)][::-1])
#     return f'{poly} as "POLYNOMIAL({x})"'



# def num_words(self, target, text_var):
#     return f'''sum(array_length(regexp_split_to_array({ text_var }, '\s'),1)) as "NUM_WORDS({ text_var })"'''
