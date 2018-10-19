# coding: utf-8

from .abstractions import Feature

class Transformer:
    """
    Base class for transformation functions

    From the PostgreSQL docs:
    The syntax for a function call is the name of a function (possibly qualified
    with a schema name), followed by its argument list enclosed in parentheses:

    """
    def __init__(self, name, transformer=None, input_types=['numeric'], output_type='numeric', stackable=True):
        self.name = name
        self.transformer = transformer if transformer else self.name
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
        if feature.type == 'key':
            trans_feature = feature
        elif feature.type not in self.input_types:
            # Don't do anything
            trans_feature = None
        else:
            trans_feature = Feature(name=self._build_name(self.name, feature),
                                    type=self.output_type,
                                    definition=self._build_transformer_call(feature),
                                    parents = feature,
                                    entity = parent,
                                    stack_depth=feature.stack_depth + 1)

        return trans_feature

abs = Transformer(name='abs')
exp = Transformer(name='exp')
ln = Transformer(name='ln')
log = Transformer(name='log')
# log2 = log(2, x)
# power = power(a, b) # a^b
sqrt = Transformer(name='sqrt')
cbrt = Transformer(name='cbrt')
sign = Transformer(name='sign')
num_chars = Transformer(name='num_chars', transformer='char_lenght')
# random = Transformer(name='random')  # without arguments  Should setseed(number)
identity = Transformer(name='identity', transformer='')

# #round = partialmethod(_unitary, function='round')
# # power


class WindowFunctionTransformer:
    """
    A window function call represents the application of an aggregate-like
    function over some portion of the rows selected by a query.
    Unlike non-window aggregate calls, this is not tied to grouping of the
    selected rows into a single output row — each row remains separate in the query output.
    However the window function hasf access to all the rows that would
    be part of the current row's group according to the grouping specification
    (PARTITION BY list) of the window function call.
    """
    def __init__(self, name, function=None, input_types=['numeric'], output_type='numeric', order_by=None, filter=None, frame_start=None, frame_end=None, stackable=True):
        self.name = name
        self.function = function if function else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.order_by = order_by
        self.filter = filter  # filter' FILTER WHERE :filter'
        self.stackable = stackable
        # The frame_clause specifies the set of rows constituting
        # the window frame, which is a subset of the current partition,
        # for those window functions that act on the frame
        # instead of the whole partition.
        # The frame can be specified in either RANGE or ROWS mode;
        # in either case, it runs from the frame_start to
        # the frame_end. If frame_end is omitted, it defaults to
        # CURRENT ROW.
        self.frame_start = frame_start
        self.frame_end = frame_end

    @staticmethod
    def _build_name(name, feature):
        name = f'{ str.upper(name) }({feature.entity.alias}.{feature.name})'
        return f'''"{name.replace('"', '')}"'''

    def _build_window_function_call(self, parent, feature):
        expression = feature.name
        partition = parent.id.name
        window_call = [f"{ self.function }({ expression })"]
        if self.filter and feature.specials:
            # filter by clause
            window_call.append(f" filter (where {feature.name} = {feature.specials}) ")
        window_call.append(f" over (partition by { partition }")
        if self.order_by and feature.sort:
            window_call.append(f" order by { self.order_by } ")
        if self.frame_start:
            if self.frame_end:
                 window_call.append(f" rows between {self.frame_start} and {self.frame_end} ")
            else:
                window_call.append(f" rows {self.frame_start} ")
        else:
            window_call.append(")")

        return ' '.join(window_call)

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            # We don't do anything
            window_feature = None
        else:
            window_feature = Feature(name = self._build_name(self.name, feature),
                                     type=self.output_type,
                                     definition=self._build_window_function_call(parent, feature),
                                     parents = feature,
                                     entity = parent,
                                     stack_depth=feature.stack_depth + 1)
        return window_feature


cum_sum = WindowFunctionTransformer(name='cum_sum', function='sum')
cum_mean = WindowFunctionTransformer(name='cum_mean', function='avg')
cum_max = WindowFunctionTransformer(name='cum_max', function='max')
cum_min = WindowFunctionTransformer(name='cum_min', function='min')
cum_count = WindowFunctionTransformer(name='cum_count', function='count', input_types=['categorical', 'index'])

# All of the following act on the window frame, not in the partition
# TODO: Include any or *
first = WindowFunctionTransformer(name='first', function='first_value', input_types=['categorical', 'index', 'numeric', 'date'])
last = WindowFunctionTransformer(name='last', function='last_value', input_types=['categorical', 'index', 'numeric'])
#nth_value = WindowFunctionTransformer(name='nth_value', function='', input_types=['categorical', 'index', 'numeric', 'date'])

previous = WindowFunctionTransformer(name='previous', function='lag')
# TODO: Add the second from current, third from current, so on

class Diff(WindowFunctionTransformer):
    def __init__(self, name, input_types=['numeric'], output_type='numeric', order_by=None, filter=None, frame_start=None, frame_end=None, stackable=True):
        function=None
        super().__init__(name, function, input_types, output_type, order_by=None, filter=None, frame_start=None, frame_end=None, stackable=True)

    def _build_window_function_call(self, parent, feature):
        expression = feature.name
        partition = parent.id.name
        window_call = [f"{expression} - lag({ expression })"]
        if self.filter and feature.specials:
            # filter by clause
            window_call.append(f" filter (where {feature.name} = {feature.specials}) ")
        window_call.append(f" over (partition by { partition }")
        if self.order_by and feature.sort:
            window_call.append(f" order by { self.order_by } ")
        if self.frame_start:
            if self.frame_end:
                 window_call.append(f" rows between {self.frame_start} and {self.frame_end} ")
            else:
                window_call.append(f" rows {self.frame_start} ")
        else:
            window_call.append(")")

        return ' '.join(window_call)


diff = Diff(name='diff')
time_since_previous = Diff(name='time_since_previous', input_types=['date', 'timestamp'], output_type='date')

class DistributionTransformer(WindowFunctionTransformer):
    def __init__(self, name, function=None, arg_func=None, input_types=['numeric'], output_type='numeric', order_by=None, frame_start=None, frame_end=None, stackable=True):
        #  Only window functions that are aggregates accept a FILTER clause.
        filter = False
        self.arg_func = arg_func
        super().__init__(name,function, input_types, output_type, order_by, filter, frame_start, frame_end, stackable)

    def _build_window_function_call(self, parent, feature):
        partition = parent.id.name
        window_call = []
        if self.arg_func:
            window_call.append(f"{ self.function }({ self.arg_func })")
        else:
            window_call.append(f"{ self.function }()")
        window_call.append(f" over (partition by { partition }")
        if self.order_by and feature.sort:
            window_call.append(f" order by { self.order_by } ")
        if self.frame_start:
            if self.frame_end:
                 window_call.append(f" rows between {self.frame_start} and {self.frame_end} ")
            else:
                window_call.append(f" rows {self.frame_start} ")
        else:
            window_call.append(")")

        return ' '.join(window_call)


cdf = DistributionTransformer(name='cdf', function='cum_dist')
## relative rank of the current row: (rank - 1) / (total partition rows - 1)
percent_rank = DistributionTransformer(name='percent_rank')
ntile = DistributionTransformer(name='ntile', arg_func=5)

class DateTransformer(Transformer):
    def __init__(self, name, date_part):
        self.date_part = date_part
        super().__init__(name, input_types=['date', 'timestamp'], output_type='categorical', stackable=True)

    def _build_transformer_call(self, feature):
        return f"to_char({ feature.name }, '{self.date_part}')"

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
        return f"({feature.entity.alias}.{ feature.name } is null)"

class IsInArray(Transformer):
    def __init__(self):
        name = 'in_array'
        super().__init__(name, transformer=None, input_types=['numeric', 'categorical', 'date'], output_type='boolean', stackable=True)

    def _build_transformer_call(self, feature, an_array):
        return f"({feature.entity.alias}.{feature.name} = ANY (ARRAY {an_array})"

    def __call__(self, parent, feature, an_array):
        if feature.type not in self.input_types:
            # Don't do anything
            trans_feature = None
        else:
            trans_feature = Feature(name=self._build_name(self.name, feature),
                                    type=self.output_type,
                                    definition=self._build_transformer_call(feature, an_array),
                                    parents = feature,
                                    entity = parent,
                                    stack_depth=feature.stack_depth + 1)

        return trans_feature

isnull = IsNull()
inarray = IsInArray()

# TODO: Transform dates to cyclical (Fourier)


# def _polynomial(self, target, x_1, x_2):
#     return f'(1 + { x_1 } + { x_2 } + { x_1 }*{ x_2 } + { x_1 }**2 + { x_2 }**2)'

# def _polynomial(self, target, x, coefs):
#     poly =  ' '.join(["{:+d}*{:s}**{:d}".format(a,x,n) for n, a in enumerate(coefs)][::-1])
#     return f'{poly} as "POLYNOMIAL({x})"'



# def num_words(self, target, text_var):
#     return f'''sum(array_length(regexp_split_to_array({ text_var }, '\s'),1)) as "NUM_WORDS({ text_var })"'''
