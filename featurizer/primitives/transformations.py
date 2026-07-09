# coding: utf-8

"""Transformation primitives and registry wiring.

Unary transformers should register via `register_transformer` to remain discoverable
by the featurizer without requiring manual imports.

Transformation primitives are applied to features within an entity. They transform
individual values or compute window functions over partitions.

Categories of transformers:
    - Basic: identity (pass-through)
    - Math: abs, exp, ln, log, sqrt, cbrt, sign, ceil, floor, trunc
    - Text: num_chars
    - Date parts: day, dow, month, year, hour, quarter, week, etc.
    - Binning: hourly_bin, daily_bin
    - Cyclical: cyclic_hour, cyclic_month, cyclic_day (sin/cos encoding)
    - Cumulative: cum_sum, cum_mean, cum_max, cum_min, cum_count
    - Window: first, last, previous, diff, time_since_previous
    - Lag: lag_1, lag_3, lag_7
    - Rolling: rolling_mean_*, rolling_std_*, rolling_median_*, rolling_iqr_*
    - EMA: ema_7, ema_14
    - Holt-Winters: holt_winters_level_*, holt_winters_trend_*
    - Percentage change: pct_change_1, pct_change_3
    - Distribution: cdf, percent_rank, ntile
    - Boolean: is_null, in_array

Example usage:
    >>> from featurizer.primitives.utils import get_transformers
    >>> transforms = get_transformers(["abs", "lag_1", "rolling_mean_7"])
    >>> for name, t in transforms.items():
    ...     print(f"{name}: {t}")

Important: Transformers must return NEW Feature instances (never mutate input)
to preserve hashing semantics for set operations and deduplication.
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple

from .abstractions import Entity, Feature, pg_identifier
from .utils import register_transformer


def _parent_token(feature: Feature, *, use_label: bool) -> str:
    """``alias.identifier`` for one parent feature inside a transformer name.

    ``use_label=True`` substitutes the parent's full, untruncated ``label`` so
    the readable chain survives PostgreSQL's 63-byte cap at any nesting depth;
    ``use_label=False`` substitutes its possibly hash-truncated ``name`` — the
    actual column the generated SQL reads. Quotes are stripped either way.
    """
    inner = (feature.label if use_label else feature.name) or feature.name
    return f"{feature.entity.alias}.{inner}".replace('"', "")


def _name_label(prefix: str, *features: Feature) -> Tuple[str, str]:
    """``(pg_identifier name, full untruncated label)`` for a transformer output.

    Names follow ``PREFIX(alias.parent[, alias.parent2...])`` — the same grammar
    the aggregators emit — so a long transformer-wrapped name gets a
    deterministic hash-suffixed identifier (bug #8) AND a manifest ``label`` that
    maps the capped column back to its intended name. ``prefix`` is upper-cased
    here, matching the historical ``str.upper(name)`` spelling, so short names
    stay byte-identical (the ADR-0007 name-stability contract).
    """
    prefix = prefix.upper()
    name_core = ", ".join(_parent_token(f, use_label=False) for f in features)
    label_core = ", ".join(_parent_token(f, use_label=True) for f in features)
    return pg_identifier(f"{prefix}({name_core})"), f"{prefix}({label_core})"


class Transformer:
    """Base class for transformation functions.

    Transformers apply functions to individual feature values within an entity.
    They generate SQL function calls that transform column values.

    From PostgreSQL docs:
        "The syntax for a function call is the name of a function followed by
        its argument list enclosed in parentheses."

    Attributes:
        name: Unique identifier for this transformation.
        transformer: SQL function name (defaults to name).
        input_types: List of feature types this transformation accepts.
        output_type: Type of the resulting feature.
        stackable: If True, can be composed with other primitives.

    Example:
        >>> class Square(Transformer):
        ...     def __init__(self):
        ...         super().__init__(name='square')
        ...     def _build_transformer_call(self, feature):
        ...         return f"POWER({feature.name}, 2)"
        >>> register_transformer('square', Square())

    Important:
        Subclasses should return NEW Feature instances from __call__,
        never mutate the input feature.
    """

    def __init__(
        self,
        name,
        transformer=None,
        input_types=["numeric"],
        output_type="numeric",
        stackable=True,
    ):
        self.name = name
        self.transformer = transformer if transformer is not None else self.name
        self.input_types = input_types
        self.output_type = output_type
        self.stackable = stackable

    @staticmethod
    def _build_name(name, feature):
        return _name_label(name, feature)[0]

    @staticmethod
    def _build_label(name, feature):
        return _name_label(name, feature)[1]

    def _build_transformer_call(self, feature):
        return f""" {self.transformer}({feature.name}) """

    def __call__(self, parent, feature):
        if feature.type == "key" or feature.type not in self.input_types:
            return feature
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=self._build_transformer_call(feature),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class DomainGuardedTransformer(Transformer):
    """A unary transformer whose SQL function has a restricted real domain.

    ``ln``/``log`` (defined only for x > 0) and ``sqrt`` (x >= 0) raise a hard
    PostgreSQL error on out-of-domain input — ``cannot take logarithm of a
    negative number`` / ``... of zero``. Applied blindly across a feature matrix
    (e.g. to a z-score, a difference, a deviation — all legitimately signed),
    that one bad row aborts the *entire* materialization. Wrapping the call in
    ``case when <domain> then fn(x) end`` yields SQL ``NULL`` for out-of-domain
    rows instead: an honest "undefined here" that never crashes the matrix, and
    that a downstream imputer/encoder already handles like any other NULL. This
    is not swallowing an error — a non-positive input to ``ln`` is a domain
    condition, not a bug.

    Only ``_build_transformer_call`` changes; ``name``/``label`` still derive
    from ``self.name``, so the ADR-0007 output-column naming contract is
    byte-for-byte unchanged.
    """

    def __init__(self, name, *, domain, transformer=None, **kwargs):
        super().__init__(name, transformer=transformer, **kwargs)
        # ``domain`` is a predicate template over ``{x}`` (the argument SQL).
        self.domain = domain

    def _build_transformer_call(self, feature):
        x = feature.name
        return (
            f""" case when {self.domain.format(x=x)} """
            f"""then {self.transformer}({x}) end """
        )


abs = Transformer(name="abs")
exp = Transformer(name="exp")
# Domain-guarded: ln/log are defined for x > 0, sqrt for x >= 0. Out-of-domain
# rows render SQL NULL instead of aborting the whole matrix (see the class docstring).
ln = DomainGuardedTransformer(name="ln", domain="{x} > 0")
log = DomainGuardedTransformer(name="log", domain="{x} > 0")
# log2 = log(2, x)
# power = power(a, b) # a^b
sqrt = DomainGuardedTransformer(name="sqrt", domain="{x} >= 0")
cbrt = Transformer(name="cbrt")
sign = Transformer(name="sign")
num_chars = Transformer(
    name="num_chars", transformer="char_length", input_types=["text"]
)
# random = Transformer(name='random')  # without arguments  Should setseed(number)
ceil = Transformer(name="ceil")
floor = Transformer(name="floor")
trunc = Transformer(name="trunc")
# #round = partialmethod(_unitary, function='round')
# # power


class Identity(Transformer):
    def __init__(self):
        super().__init__(name="identity", input_types=["numeric", "categoric", "text"])

    @staticmethod
    def _build_name(name, feature):
        name = f"{feature.entity.alias}.{feature.name}"
        return f'''"{name.replace('"', "")}"'''

    def _build_transformer_call(self, feature):
        return f""" {feature.name} """

    def __call__(self, parent, feature):
        return feature


identity = Identity()


class DateTransformer(Transformer):
    def __init__(self, name, date_part):
        self.date_part = date_part
        super().__init__(
            name,
            input_types=["date", "timestamp", "index"],
            output_type="categorical",
            stackable=True,
        )

    def _build_transformer_call(self, feature):
        return f"to_char({feature.name}, '{self.date_part}')"

    def __call__(self, parent, feature):
        if feature.type == "key":
            return feature
        temporal_ix = getattr(feature.entity, "temporal_ix", None)
        if feature.type == "index" and feature is not temporal_ix:
            return feature
        if feature.type not in self.input_types:
            return feature
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=self._build_transformer_call(feature),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


day = DateTransformer(name="day", date_part="day")
dow = DateTransformer(name="dow", date_part="ID")  # Iso week: Monday (1) to Sunday (7)
dom = DateTransformer(name="dom", date_part="DD")
doy = DateTransformer(name="doy", date_part="DDD")
year = DateTransformer(name="year", date_part="YYYY")
month = DateTransformer(name="month", date_part="M")
hour = DateTransformer(name="hour", date_part="HH24")
century = DateTransformer(name="century", date_part="CC")
quarter = DateTransformer(name="quarter", date_part="Q")
week = DateTransformer(name="week", date_part="W")
week_of_year = DateTransformer(name="week_of_year", date_part="WW")
time_zone = DateTransformer(name="tz", date_part="TZ")
tz_offset = DateTransformer(name="tz_offset", date_part="OF")


class HourlyBinning(Transformer):
    def __init__(self):
        super().__init__(
            name="hourly_bin",
            transformer=None,
            input_types=["date", "timestamp"],
            output_type="categorical",
            stackable=True,
        )

    def _build_transformer_call(self, feature):
        return f"""
        (
        case
        when extract(hour from {feature.name}) <@ int4range(0,5) then 'night'
        when extract(hour from {feature.name}) <@ int4range(5,8) then 'early_morning'
        when extract(hour from {feature.name}) <@ int4range(8,11) then 'morning'
        when extract(hour from {feature.name}) <@ int4range(11,14) then 'midday'
        when extract(hour from {feature.name}) <@ int4range(14,19) then 'afternoon'
        when extract(hour from {feature.name}) <@ int4range(19,22) then 'evening'
        when extract(hour from {feature.name}) <@ int4range(22,24) then 'night'
        )
        """


class DailyBinning(Transformer):
    def __init__(self):
        super().__init__(
            name="daily_bin",
            transformer=None,
            input_types=["date", "timestamp"],
            output_type="categorical",
            stackable=True,
        )

    def _build_transformer_call(self, feature):
        return f"""
        (
        case
        when to_char({feature.name},'ID')::smallint <@ int4range(0,5) then 'weekday'
        when to_char({feature.name},'ID')::smallint <@ int4range(5,7) then 'weekday'
        )
        """


hourly_binning = HourlyBinning()
daily_binning = DailyBinning()


class CyclicalDateTransformer(DateTransformer):
    def __init__(self, name, date_part, period, adjust=True):
        self.period = period
        self.adjust = adjust
        super().__init__(name=name, date_part=date_part)

    def _build_transformer_call(self, feature, trig_function):
        if self.adjust:
            return f"""{trig_function}((to_char({feature.name}, '{self.date_part}')::smallint - 1)*(2*pi()/{self.period}))"""
        else:
            return f"""{trig_function}((to_char({feature.name}, '{self.date_part}')::smallint)*(2*pi()/{self.period}))"""

    def __call__(self, parent, feature):
        if feature.type == "key" or feature.type not in self.input_types:
            return feature
        sin_name, sin_label = _name_label(self.name + "_sin", feature)
        cos_name, cos_label = _name_label(self.name + "_cos", feature)
        return [
            Feature(
                name=sin_name,
                type=self.output_type,
                definition=self._build_transformer_call(feature, trig_function="sin"),
                parents=feature,
                entity=parent,
                stack_depth=feature.stack_depth + 1,
                label=sin_label,
            ),
            Feature(
                name=cos_name,
                type=self.output_type,
                definition=self._build_transformer_call(feature, trig_function="cos"),
                parents=feature,
                entity=parent,
                stack_depth=feature.stack_depth + 1,
                label=cos_label,
            ),
        ]


cyclic_hour = CyclicalDateTransformer(
    name="cyclic_hour", date_part="HH24", period=24, adjust=False
)
cyclic_month = CyclicalDateTransformer(name="cyclic_month", date_part="MM", period=12)
cyclic_day = CyclicalDateTransformer(name="cyclic_hour", date_part="D", period=7)


class WindowFunctionTransformer:
    """Window function transformer for aggregate-like operations over row partitions.

    Window functions compute values across a set of rows related to the current row,
    without collapsing rows like regular aggregates do. Each row retains its identity.

    From PostgreSQL docs:
        "A window function call represents the application of an aggregate-like
        function over some portion of the rows selected by a query. Unlike
        non-window aggregate calls, this is not tied to grouping of the selected
        rows into a single output row — each row remains separate in the query output."

    Attributes:
        name: Unique identifier for this transformation.
        function: SQL window function name.
        order_by: Callable or string for ORDER BY clause (often temporal_ix).
        frame: Tuple defining the window frame (start, end), e.g., ('3 preceding', 'current row').
        extra_args: Additional arguments to pass to the function.

    SQL Pattern:
        FUNCTION(value) OVER (PARTITION BY id ORDER BY date [ROWS BETWEEN ... AND ...])

    Examples:
        - cum_sum: SUM(value) OVER (PARTITION BY id ORDER BY date)
        - rolling_mean_7: AVG(value) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
    """

    def __init__(
        self,
        name,
        function=None,
        input_types=["numeric"],
        output_type="numeric",
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
        self.extra_args: Tuple[Callable[[Feature], str] | str, ...] = tuple(
            extra_args or ()
        )

    @staticmethod
    def _build_name(name, feature):
        return _name_label(name, feature)[0]

    @staticmethod
    def _build_label(name, feature):
        return _name_label(name, feature)[1]

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
        partition = parent.id.name if parent.id else None
        if not partition:
            return None
        window_args = [expression] + list(self._resolve_args(feature))
        window_call = [f"{self.function}({', '.join(window_args)})"]
        if self.filter and feature.specials:
            # filter by clause
            window_call.append(f" filter (where {feature.name} = {feature.specials}) ")
        window_call.append(f" over (partition by {partition}")
        order_clause = self._resolve_order_by(feature)
        if order_clause:
            window_call.append(f" order by {order_clause}")
        if self.frame:
            start, end = self.frame
            window_call.append(f" rows between {start} and {end}")
        window_call.append(")")

        return " ".join(window_call)

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        definition = self._build_window_function_call(parent, feature)
        if not definition:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=definition,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


def _temporal_ordering(feature: Feature) -> Optional[str]:
    temporal_ix = getattr(feature.entity, "temporal_ix", None)
    if temporal_ix is None:
        return None
    return temporal_ix.name


def _build_temporal_window(
    function: str,
    parent: Entity,
    feature: Feature,
    *,
    args: Iterable[str] = (),
    frame: Optional[Tuple[str, str]] = None,
) -> Optional[str]:
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


# Alias the transform CTE's source row (`from <alias>_synth <TRANSFORM_EGO_ALIAS>`)
# so a rolling ordered-set aggregate can correlate a re-scan of the same _synth
# rows against the current ("ego") row. Shared with planner._build_transform_cte.
TRANSFORM_EGO_ALIAS = "_ego"


def _build_rolling_percentile(
    parent: Entity,
    feature: Feature,
    percentile: float,
    window: int,
) -> Optional[str]:
    """A row-framed rolling percentile, as a correlated subquery.

    PostgreSQL forbids ``OVER`` on ordered-set aggregates, so a windowed
    ``percentile_cont`` cannot be a window function. Instead, for each ego row,
    re-scan the entity's ``_synth`` rows, take the ``window`` most-recent by the
    temporal index up to and including the current row, and take the percentile
    over them. Transformers are only ever applied to ``_synth`` columns, so
    ``feature.name`` is guaranteed to be a column of ``<alias>_synth``.
    """
    partition = parent.id.name if parent.id else None
    if partition is None:
        return None
    order_by = _temporal_ordering(feature)
    if order_by is None:
        return None
    synth = f"{parent.alias}_synth"
    ego = TRANSFORM_EGO_ALIAS
    return (
        f"(select percentile_cont({percentile}) within group (order by _w.v) "
        f"from (select {synth}.{feature.name} as v from {synth} "
        f"where {synth}.{partition} = {ego}.{partition} "
        f"and {synth}.{order_by} <= {ego}.{order_by} "
        f"order by {synth}.{order_by} desc limit {window}) _w)"
    )


cum_sum = WindowFunctionTransformer(
    name="cum_sum", function="sum", order_by=_temporal_ordering
)
cum_mean = WindowFunctionTransformer(
    name="cum_mean", function="avg", order_by=_temporal_ordering
)
cum_max = WindowFunctionTransformer(
    name="cum_max", function="max", order_by=_temporal_ordering
)
cum_min = WindowFunctionTransformer(
    name="cum_min", function="min", order_by=_temporal_ordering
)
cum_count = WindowFunctionTransformer(
    name="cum_count",
    function="count",
    input_types=["categorical", "index"],
    order_by=_temporal_ordering,
)

# All of the following act on the window frame, not in the partition
# TODO: Include any or *
first = WindowFunctionTransformer(
    name="first",
    function="first_value",
    input_types=["categorical", "index", "numeric", "date"],
    order_by=_temporal_ordering,
)
last = WindowFunctionTransformer(
    name="last",
    function="last_value",
    input_types=["categorical", "index", "numeric"],
    order_by=_temporal_ordering,
    # With ORDER BY and no explicit frame, PostgreSQL defaults to
    # `range between unbounded preceding and current row`, so `last_value`
    # returns the current row (i.e. `last` ≡ `identity`). Force the full
    # partition frame so it returns the partition's actual last value.
    frame=("unbounded preceding", "unbounded following"),
)
# nth_value = WindowFunctionTransformer(name='nth_value', function='', input_types=['categorical', 'index', 'numeric', 'date'])

previous = WindowFunctionTransformer(
    name="previous", function="lag", order_by=_temporal_ordering
)


class Diff:
    def __init__(self, name, input_types=["numeric"], output_type="numeric"):
        self.name = name
        self.input_types = input_types
        self.output_type = output_type

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        lag_expr = _build_temporal_window("lag", parent, feature)
        if lag_expr is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=f"{feature.name} - {lag_expr}",
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


diff = Diff(name="diff")
time_since_previous = Diff(
    name="time_since_previous", input_types=["date", "timestamp"], output_type="date"
)


class NthDiff:
    """N-th order finite difference (acceleration, jerk) via binomial lags.

    ``diff2 = x - 2*lag1 + lag2`` (acceleration); ``diff3 = x - 3*lag1 +
    3*lag2 - lag3`` (jerk). Backward-only: built only from lags over the
    entity's temporal order.
    """

    def __init__(self, name, order, input_types=["numeric"], output_type="numeric"):
        self.name = name
        self.order = order
        self.input_types = input_types
        self.output_type = output_type

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        lags = {}
        for k in range(1, self.order + 1):
            expr = _build_temporal_window("lag", parent, feature, args=[str(k)])
            if expr is None:
                return None
            lags[k] = expr
        x = feature.name
        if self.order == 2:
            definition = f"({x}) - 2*({lags[1]}) + ({lags[2]})"
        elif self.order == 3:
            definition = f"({x}) - 3*({lags[1]}) + 3*({lags[2]}) - ({lags[3]})"
        else:
            raise ValueError(f"Unsupported difference order: {self.order}")
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=definition,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class CumProd:
    """Running product via the log-sum-exp identity (positive series).

    ``exp(sum(ln x) over (partition by id order by ts))``. Returns NULL once a
    non-positive value enters the running window — a documented limitation,
    since ``ln`` is undefined there. Backward-only.
    """

    def __init__(self, name="cumprod", input_types=["numeric"], output_type="numeric"):
        self.name = name
        self.input_types = input_types
        self.output_type = output_type

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        partition = parent.id.name if parent.id else None
        if partition is None:
            return None
        order_by = _temporal_ordering(feature)
        if order_by is None:
            return None
        x = feature.name
        window = f"over (partition by {partition} order by {order_by})"
        definition = (
            f"case when min({x}) {window} > 0 "
            f"then exp(sum(ln({x})) {window}) else null end"
        )
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=definition,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


diff2 = NthDiff(name="diff2", order=2)
diff3 = NthDiff(name="diff3", order=3)
cumprod = CumProd()


class DistributionTransformer(WindowFunctionTransformer):
    def __init__(
        self,
        name,
        function=None,
        arg_func=None,
        input_types=["numeric"],
        output_type="numeric",
        order_by=None,
        frame=None,
        stackable=True,
    ):
        #  Only window functions that are aggregates accept a FILTER clause.
        filter = False
        self.arg_func = arg_func
        super().__init__(
            name, function, input_types, output_type, order_by, filter, frame, stackable
        )

    def _build_window_function_call(self, parent, feature):
        partition = parent.id.name if parent.id else None
        if not partition:
            return None
        pieces = []
        if self.arg_func:
            pieces.append(f"{self.function}({self.arg_func})")
        else:
            pieces.append(f"{self.function}()")
        pieces.append(f" over (partition by {partition}")
        order_clause = (
            self._resolve_order_by(feature)
            if hasattr(self, "_resolve_order_by")
            else None
        )
        if order_clause:
            pieces.append(f" order by {order_clause}")
        if self.frame:
            start, end = self.frame
            pieces.append(f" rows between {start} and {end}")
        pieces.append(")")
        return " ".join(pieces)


cdf = DistributionTransformer(
    name="cdf", function="cum_dist", order_by=_temporal_ordering
)
## relative rank of the current row: (rank - 1) / (total partition rows - 1)
percent_rank = DistributionTransformer(name="percent_rank", order_by=_temporal_ordering)
ntile = DistributionTransformer(name="ntile", arg_func=5, order_by=_temporal_ordering)


class LagTransformer:
    """Access values from N periods ago.

    Creates a feature containing the value from N rows back, ordered by
    the entity's temporal index. Useful for comparing current values to
    historical values.

    Args:
        periods: Number of periods to look back.

    SQL: LAG(value, N) OVER (PARTITION BY id ORDER BY temporal_ix)

    Examples:
        - lag_1: Previous period's value
        - lag_7: Value from 7 periods ago (e.g., same day last week)
    """

    def __init__(self, periods: int):
        self.periods = periods
        self.name = f"lag_{periods}"
        self._input_types = ["numeric", "categorical", "date", "timestamp", "index"]

    def __call__(self, parent, feature):
        if feature.type == "key":
            return feature
        if feature.type not in self._input_types:
            return feature
        expression = _build_temporal_window(
            "lag", parent, feature, args=[str(self.periods)]
        )
        if expression is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=feature.type,
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class RollingStatisticTransformer:
    """Rolling window statistics (mean, std, etc.).

    Computes statistics over a sliding window of N rows preceding and
    including the current row. Useful for smoothing time series and
    detecting trends.

    Args:
        label: Base name for the transformer (e.g., 'rolling_mean').
        function: SQL aggregate function to apply (e.g., 'avg', 'stddev').
        window: Number of rows in the window (including current).

    SQL: FUNCTION(value) OVER (... ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW)

    Examples:
        - rolling_mean_7: 7-day moving average
        - rolling_std_14: 14-day rolling standard deviation
    """

    def __init__(self, label: str, function: str, window: int):
        self.label = label
        self.function = function
        self.window = window
        self.name = f"{label}_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        frame = _frame_for_window(self.window)
        expression = _build_temporal_window(self.function, parent, feature, frame=frame)
        if expression is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class RollingMedianTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"rolling_median_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        expression = _build_rolling_percentile(parent, feature, 0.5, self.window)
        if expression is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class RollingIQRTransformer:
    def __init__(self, window: int):
        self.window = window
        self.name = f"rolling_iqr_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        p75 = _build_rolling_percentile(parent, feature, 0.75, self.window)
        p25 = _build_rolling_percentile(parent, feature, 0.25, self.window)
        if not p75 or not p25:
            return None
        expression = f"({p75}) - ({p25})"
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class ExponentialMovingAverageTransformer:
    """Exponential Moving Average (EMA) with time-based weighting.

    EMA gives more weight to recent observations, with weights decaying
    exponentially based on time distance. More responsive to recent
    changes than simple moving averages.

    Args:
        window: Number of rows in the window.
        decay: Decay rate for exponential weighting (higher = faster decay).

    SQL Pattern:
        SUM(value * EXP(decay * time)) / SUM(EXP(decay * time)) OVER (...)

    Use cases:
        - Trend following in financial time series
        - Smoothing noisy signals while preserving responsiveness
    """

    def __init__(self, window: int, decay: float):
        self.window = window
        self.decay = decay
        self.name = f"ema_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
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
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class HoltWintersLevelTransformer:
    """Holt-Winters level component (smoothed average).

    Extracts the level (base value) component from a time series,
    implemented as a rolling average. Part of the Holt-Winters
    exponential smoothing method for forecasting.

    Args:
        window: Number of rows for smoothing.

    SQL: AVG(value) OVER (... ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW)
    """

    def __init__(self, window: int):
        self.window = window
        self.name = f"holt_winters_level_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        frame = _frame_for_window(self.window)
        expression = _build_temporal_window("avg", parent, feature, frame=frame)
        if expression is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class HoltWintersTrendTransformer:
    """Holt-Winters trend component (slope over time).

    Extracts the trend (direction) component from a time series using
    linear regression slope. Part of the Holt-Winters exponential
    smoothing method for forecasting.

    Args:
        window: Number of rows for trend calculation.

    SQL: REGR_SLOPE(value, time) OVER (... ROWS BETWEEN N-1 PRECEDING AND CURRENT ROW)

    Use cases:
        - Detecting upward/downward trends
        - Forecasting future values
    """

    def __init__(self, window: int):
        self.window = window
        self.name = f"holt_winters_trend_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        order_by = _temporal_ordering(feature)
        if order_by is None:
            return None
        frame = _frame_for_window(self.window)
        # regr_slope needs a numeric X axis; the temporal index is a date/
        # timestamp, so regress the value against epoch seconds.
        expression = _build_temporal_window(
            "regr_slope",
            parent,
            feature,
            args=[f"extract(epoch from {order_by})"],
            frame=frame,
        )
        if expression is None:
            return None
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class PercentageChangeTransformer:
    def __init__(self, periods: int):
        self.periods = periods
        self.name = f"pct_change_{periods}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        lag_expr = _build_temporal_window(
            "lag", parent, feature, args=[str(self.periods)]
        )
        if lag_expr is None:
            return None
        expression = f"""
        case
        when {lag_expr} is null or {lag_expr} = 0 then null
        else ({feature.name} - {lag_expr}) / {lag_expr}
        end
        """
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


class BinaryTransformer(Transformer):
    def __init__(
        self,
        name,
        operation,
        input_types=["numeric"],
        output_type="numeric",
        stackable=True,
    ):
        self.operation = operation
        self.transformer = None
        super().__init__(name, self.transformer, input_types, output_type, stackable)

    @staticmethod
    def _build_name(name, feature1, feature2):
        return _name_label(name, feature1, feature2)[0]

    @staticmethod
    def _build_label(name, feature1, feature2):
        return _name_label(name, feature1, feature2)[1]

    def _build_transformer_call(self, feature1, feature2):
        return f"{feature1.entity.alias}.{feature1.name} {self.operation}  {feature2.entity.alias}.{feature2.name}"

    def __call__(self, parent, feature1, feature2):
        if (
            feature1.type not in self.input_types
            or feature2.type not in self.input_types
        ):
            # Don't do anything
            trans_feature = None
        else:
            name, label = _name_label(self.name, feature1, feature2)
            trans_feature = Feature(
                name=name,
                type=self.output_type,
                definition=self._build_transformer_call(feature1, feature2),
                parents=[feature1, feature2],
                entity=parent,
                stack_depth=feature1.stack_depth + 1,
                label=label,
            )

        return trans_feature


add = BinaryTransformer(name="add", operation="+")
difference = BinaryTransformer(name="subs", operation="-")
multiply = BinaryTransformer(name="mul", operation="*")
ratio = BinaryTransformer(name="div", operation="/")
modulo = BinaryTransformer(name="mod", operation="%")
exponentiation = BinaryTransformer(name="exponentiation", operation="^")
bitwise_and = BinaryTransformer(name="bitwise_and", operation="&")
bitwise_or = BinaryTransformer(name="bitwise_or", operation="|")
bitwise_xor = BinaryTransformer(name="bitwise_xor", operation="#")
bitwise_shift_left = BinaryTransformer(name="bitwise_shift_left", operation="<<")
bitwise_shift_right = BinaryTransformer(name="bitwise_shift_right", operation=">>")

# Boolean
boolean_and = BinaryTransformer(
    name="and", operation="and", input_types=["boolean"], output_type="boolean"
)
boolean_or = BinaryTransformer(
    name="or", operation="or", input_types=["boolean"], output_type="boolean"
)

# Logical
eq = BinaryTransformer(name="eq", operation="=")
neq = BinaryTransformer(name="neq", operation="!=")
lt = BinaryTransformer(name="lt", operation="<")
gt = BinaryTransformer(name="gt", operation=">")
le = BinaryTransformer(name="le", operation="<=")
ge = BinaryTransformer(name="ge", operation=">=")
time_since = BinaryTransformer(
    name="time_since",
    operation="-",
    input_types=["date", "timestamp"],
    output_type="date",
)


class IsNull(Transformer):
    def __init__(self):
        name = "is_null"
        super().__init__(
            name,
            transformer=None,
            input_types=["numeric", "categorical", "date"],
            output_type="boolean",
            stackable=True,
        )

    def _build_transformer_call(self, feature):
        return f"({feature.name} is null)"


class IsInArray(Transformer):
    def __init__(self):
        name = "in_array"
        super().__init__(
            name,
            transformer=None,
            input_types=["numeric", "categorical", "date"],
            output_type="boolean",
            stackable=True,
        )

    def _build_transformer_call(self, feature, an_array):
        return f"({feature.name} = ANY (ARRAY {an_array})"

    def __call__(self, parent, feature, an_array):
        if feature.type not in self.input_types:
            return feature
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=self._build_transformer_call(feature, an_array),
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
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
}

for _name, _transformer in DEFAULT_TRANSFORMERS.items():
    register_transformer(_name, _transformer)

# ``in_array`` is intentionally NOT in DEFAULT_TRANSFORMERS: its ``__call__``
# requires a third ``an_array`` argument that the planner (which applies
# transformers as ``transformer(entity, feature)``) cannot supply, so it crashes
# when included in a wholesale default/wide transform set. It stays registered so
# it remains discoverable and usable by a caller that passes ``an_array`` directly.
register_transformer("in_array", inarray)

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


# ---------------------------------------------------------------------------
# 5a. PopulationWindowTransformer
# ---------------------------------------------------------------------------


class PopulationWindowTransformer:
    """Cross-entity window functions (no PARTITION BY)."""

    def __init__(
        self, name, expression_template, input_types=None, output_type="numeric"
    ):
        self.name = name
        self.expression_template = expression_template
        self.input_types = input_types or ["numeric"]
        self.output_type = output_type

    def __call__(self, parent, feature):
        if feature.type == "key" or feature.type not in self.input_types:
            return feature
        expression = self.expression_template.format(col=feature.name)
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type=self.output_type,
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


cross_entity_zscore = PopulationWindowTransformer(
    name="cross_entity_zscore",
    expression_template="({col} - AVG({col}) OVER ()) / NULLIF(STDDEV({col}) OVER (), 0)",
)
cross_entity_percentile = PopulationWindowTransformer(
    name="cross_entity_percentile",
    expression_template="PERCENT_RANK() OVER (ORDER BY {col})",
)


# ---------------------------------------------------------------------------
# 5b. MeanShiftRatioTransformer
# ---------------------------------------------------------------------------


class MeanShiftRatioTransformer:
    """Ratio of recent rolling mean to prior rolling mean (change-point detection)."""

    def __init__(self, window):
        self.window = window
        self.name = f"mean_shift_ratio_{window}"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        partition = parent.id.name if parent.id else None
        order_by = _temporal_ordering(feature)
        if not partition or not order_by:
            return None
        recent_start = self.window - 1
        prior_end = self.window
        prior_start = 2 * self.window - 1
        recent = f"AVG({feature.name}) OVER (partition by {partition} order by {order_by} rows between {recent_start} preceding and current row)"
        prior = f"AVG({feature.name}) OVER (partition by {partition} order by {order_by} rows between {prior_start} preceding and {prior_end} preceding)"
        expression = f"{recent} / NULLIF({prior}, 0)"
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


# ---------------------------------------------------------------------------
# 5c. CusumTransformer
# ---------------------------------------------------------------------------


class CusumTransformer:
    """CUSUM: cumulative sum of deviations from partition mean."""

    def __init__(self):
        self.name = "cusum"

    def __call__(self, parent, feature):
        if feature.type != "numeric":
            return feature
        partition = parent.id.name if parent.id else None
        order_by = _temporal_ordering(feature)
        if not partition or not order_by:
            return None
        cum_sum = (
            f"SUM({feature.name}) OVER (partition by {partition} order by {order_by})"
        )
        row_num = f"ROW_NUMBER() OVER (partition by {partition} order by {order_by})"
        part_avg = f"AVG({feature.name}) OVER (partition by {partition})"
        expression = f"{cum_sum} - {row_num} * {part_avg}"
        name, label = _name_label(self.name, feature)
        return Feature(
            name=name,
            type="numeric",
            definition=expression,
            parents=feature,
            entity=parent,
            stack_depth=feature.stack_depth + 1,
            label=label,
        )


# ---------------------------------------------------------------------------
# Registration (added to DEFAULT_TRANSFORMERS)
# ---------------------------------------------------------------------------

_cusum = CusumTransformer()
DEFAULT_TRANSFORMERS["cross_entity_zscore"] = cross_entity_zscore
DEFAULT_TRANSFORMERS["cross_entity_percentile"] = cross_entity_percentile
DEFAULT_TRANSFORMERS["cusum"] = _cusum
register_transformer("cross_entity_zscore", cross_entity_zscore)
register_transformer("cross_entity_percentile", cross_entity_percentile)
register_transformer("cusum", _cusum)

for _window in (7, 14):
    _msr = MeanShiftRatioTransformer(_window)
    DEFAULT_TRANSFORMERS[_msr.name] = _msr
    register_transformer(_msr.name, _msr)


# --- Higher-order differences and running product ---
for _t in (diff2, diff3, cumprod):
    DEFAULT_TRANSFORMERS[_t.name] = _t
    register_transformer(_t.name, _t)


# ---------------------------------------------------------------------------
# Text Path-1: per-document lexical features (pure SQL over a text column)
# ---------------------------------------------------------------------------
#
# Each transformer reduces one text column to one numeric score per row
# ("reduce" in the reduce->aggregate text path). Being numeric, the result is
# then reduced over the entity's documents by the ordinary aggregators
# (mean, max, ...). All expressions are plain PostgreSQL string/regex calls, so
# they stay inside the SQL-feasible tier — no NLP library required.


class TextTransformer(Transformer):
    """A per-row lexical feature over a ``text`` column.

    ``template`` is a SQL expression using ``{col}`` as the placeholder for the
    input column. The output is numeric so it composes with numeric aggregators.
    """

    def __init__(self, name: str, template: str) -> None:
        super().__init__(name=name, input_types=["text"], output_type="numeric")
        self._template = template

    def _build_transformer_call(self, feature):
        return f" {self._template.replace('{col}', feature.name)} "


# Non-empty whitespace-delimited tokens of the (NULL-safe) text column.
_WORD_TOKENS = r"regexp_split_to_table(coalesce({col}, ''), '\s+') as t(w)"
_NUM_WORDS = "(select count(*) from " + _WORD_TOKENS + " where t.w <> '')"

_LEXICAL_TEMPLATES = {
    # Counts
    "num_words": _NUM_WORDS,
    "num_sentences": r"length({col}) - length(regexp_replace(coalesce({col}, ''), '[.!?]', '', 'g'))",
    "exclamation_count": r"length(coalesce({col}, '')) - length(replace(coalesce({col}, ''), '!', ''))",
    "question_count": r"length(coalesce({col}, '')) - length(replace(coalesce({col}, ''), '?', ''))",
    # Averages / ratios
    "avg_word_length": r"length(regexp_replace(coalesce({col}, ''), '\s', '', 'g'))::numeric / nullif("
    + _NUM_WORDS
    + ", 0)",
    "unique_word_ratio": "(select count(distinct lower(t.w)) from "
    + _WORD_TOKENS
    + " where t.w <> '')::numeric / nullif("
    + _NUM_WORDS
    + ", 0)",
    "caps_ratio": r"length(regexp_replace(coalesce({col}, ''), '[^A-Z]', '', 'g'))::numeric / "
    r"nullif(length(regexp_replace(coalesce({col}, ''), '[^A-Za-z]', '', 'g')), 0)",
    "digit_ratio": r"length(regexp_replace(coalesce({col}, ''), '[^0-9]', '', 'g'))::numeric / "
    r"nullif(length({col}), 0)",
    "punct_ratio": r"length(regexp_replace(coalesce({col}, ''), '[A-Za-z0-9\s]', '', 'g'))::numeric / "
    r"nullif(length({col}), 0)",
}

for _name, _template in _LEXICAL_TEMPLATES.items():
    _text_transformer = TextTransformer(_name, _template)
    DEFAULT_TRANSFORMERS[_name] = _text_transformer
    register_transformer(_name, _text_transformer)


# percentage above avg
# percentage trues


# def _polynomial(self, target, x_1, x_2):
#     return f'(1 + { x_1 } + { x_2 } + { x_1 }*{ x_2 } + { x_1 }**2 + { x_2 }**2)'

# def _polynomial(self, target, x, coefs):
#     poly =  ' '.join(["{:+d}*{:s}**{:d}".format(a,x,n) for n, a in enumerate(coefs)][::-1])
#     return f'{poly} as "POLYNOMIAL({x})"'


# def num_words(self, target, text_var):
#     return f'''sum(array_length(regexp_split_to_array({ text_var }, '\s'),1)) as "NUM_WORDS({ text_var })"'''
