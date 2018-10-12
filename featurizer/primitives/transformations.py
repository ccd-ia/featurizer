# coding: utf-8

class Transformer:
    """ Base class for transformation functions """
    def __init__(self, name, func, input_type, output_type, stackable=True):
        self.name = name
        self.func = func
        self.input_type = input_type
        self.output_type = output_type
        self.stackable = stackable

    def __call__(self, entity, features):
        return self.func(entity, features)


# ## Transformations

# def _unitary(self, target, function, variable, input_type, output_type):
#     return Feature(name=f'{ str.upper(function) }({target.alias}.{ variable.name })',
#               definition=f'{ function }({ variable.name })',
#               type=output_type)


# abs = partialmethod(_unitary, function='abs', input_type='numeric', output_type='numeric')
# exp = partialmethod(_unitary, function='exp', input_type='numeric', output_type='numeric')
# ln = partialmethod(_unitary, function='ln', input_type='numeric', output_type='numeric')
# log = partialmethod(_unitary, function='log', input_type='numeric', output_type='numeric')
# sqrt = partialmethod(_unitary, function='sqrt', input_type='numeric', output_type='numeric')
# sign = partialmethod(_unitary, function='sign', input_type='numeric', output_type='numeric')
# #round = partialmethod(_unitary, function='round')
# # power

# def _cumulative(self, target, function, variable, date_var):
#     # sum, avg, max, min
#     return f'{ function }({ variable }) over (partition by { target } order by { date_var }) as "{ str.upper(function) }({ variable })"'

# def _distributions(self, target, function, variable, **kwargs):
#     # cumd_dist, percent_rank, ntile
#     return f'{ function }() over (partition by { target } order by { variable }) as "{ str.upper(function) }({ variable })"'

# cdf = partialmethod(_distributions, function='cume_dist')
# percent_rank = partialmethod(_distributions, function='percent_rank')
# # ntile = partialmethod(_distributions, function='ntile')

# def _transform_date(self, target, date_part, date_var, input_type, output_type):
#     # CC (century), YYYY (year), Q (quarter), day, W (week of month), WW (week of year), D (day of week)
#     # DD (day of month), DDD (day of year), H (hour), tz (timezone)
#     return Feature(name=f'''{ str.upper(date_part) }({ date_var.name })''',
#               definition=f'''to_char({ date_var.name }, '{ date_part }')''',
#               type=output_type)

# day = partialmethod(_transform_date, date_part='day', input_type='date', output_type='categorical')
# dow = partialmethod(_transform_date, date_part='D', input_type='date', output_type='categorical')

# def _hourly_binning(self, target, date_var, **kwargs):
#     return f'''
#     case
#     when extract(hour from { date_var }) <@ int4range(0,5) then 'night'
#     when extract(hour from { date_var }) <@ int4range(5,8) then 'early_morning'
#     when extract(hour from { date_var }) <@ int4range(8,11) then 'morning'
#     when extract(hour from { date_var }) <@ int4range(11,14) then 'midday'
#     when extract(hour from { date_var }) <@ int4range(14,19) then 'afternoon'
#     when extract(hour from { date_var }) <@ int4range(19,22) then 'evening'
#     when extract(hour from { date_var }) <@ int4range(22,24) then 'night'
#     end as "HOURLY_BIN({ date_var })"
#     '''

# def _day_binning(self, target, date_var, **kwargs):
#     return f'''
#     case
#     when to_char({ date_var},'ID')::smallint <@ int4range(0,5) then 'weekday'
#     when to_char({ date_var},'ID')::smallint <@ int4range(5,7) then 'weekday'
#     end as "DAILY_BIN({ date_var })"
#     '''

# def _is_null(self, target, variable, **kwargs):
#     return f'({ variable } is null) as { variable }_is_null'

# def _is_in_array(self, target, variable, an_array):
#     return f'{ variable } in { an_array } as "IS_IN({ variable }, { an_array })'

# def _diff(self, target, variable, date_var):
#     return f'({ variable } - lag({ variable }) over (partition by { target } order by  { date_var } desc ) as "DIFF({ variable })'

# def _time_since_previous(self, target, date_var, input_type, output_type):
#     return f'({ date_var } - lag({ date_var }) over (partition by { target } order by { date_var } desc)) as "TIME_SINCE_PREVIOUS({ date_var }'

# def num_chars(self, target, text_var, input_type='text', output_type='numeric'):
#     return Feature(name=f'NUM_CHARS({ text_var.name })',
#               definition=f'char_length({ text_var.name })',
#               type=output_type)

# def num_words(self, target, text_var):
#     return f'''sum(array_length(regexp_split_to_array({ text_var }, '\s'),1)) as "NUM_WORDS({ text_var })"'''

# def _time_since(self, target, date_var, date):
#     # For each value compute the time elapsed between it and a datetime
#     return f'{ date_var } - { date } as "TIME_SINCE({ date }, { date_var })'

# def _time_since_major_event(self, target, date_1, date_2):
#     #  For each value compute the time elapsed between it and a datetime
#     #  Closeness to major events
#     return f'{ date_1 } - { date_2 } as { date_2 }_days_before_{ date_1 }'

# def _numeric_binary(self, target, numeric_var_1, numeric_var_2, op):
#     ops = {'add': '+',
#            'substract': '-',
#            'multiply': '*',
#            'divide': '/',
#     }

#     return f'{ numeric_var_1 } { ops[op] } { numeric_var_2 } as { str.upper(op) }({ numeric_var_1 }, { numeric_var_2 })'


# def _bool_binary(self, target, bool_var_1, bool_var_2, op):
#     # ops: and / or
#     return f'{ bool_var_1 } { op } { bool_var_2 } as { str.upper(op) }({ bool_var_1 },{ bool_var_2 })'


# def _comparison(self, target, var_1, var_2, op):
#     ops = {'eq': '=',
#            'neq': '!=',
#            'lt': '<',
#            'gt': '>',
#            'le': '<=',
#            'ge': '>='
#     }
#     return f'{ var_1 } {ops[op]} { var_2 } as "{ str.upper(op) }({ var_1 },{ var_2 })"'

# def _polynomial(self, target, x_1, x_2):
#     return f'(1 + { x_1 } + { x_2 } + { x_1 }*{ x_2 } + { x_1 }**2 + { x_2 }**2)'

# def _polynomial(self, target, x, coefs):
#     poly =  ' '.join(["{:+d}*{:s}**{:d}".format(a,x,n) for n, a in enumerate(coefs)][::-1])
#     return f'{poly} as "POLYNOMIAL({x})"'


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
    def __init__(self, **kwargs):
        pass

    def __call__(self):
        pass
