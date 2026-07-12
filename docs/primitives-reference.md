# Primitives Reference Guide

> **Canonical view:** the docs site's
> [primitives reference](https://ccd-ia.github.io/featurizer/reference/primitives/)
> is generated from the live registry at every deploy and cannot drift from the
> code. This hand-maintained file is kept for offline/in-repo reading.

This document provides a comprehensive reference for all aggregation and transformation primitives available in Featurizer.

## Quick Reference

Use the CLI to list primitives:

```bash
# List all primitives
python -m featurizer list-primitives

# List aggregations only with SQL examples
python -m featurizer list-primitives --type agg --show-sql

# List transformations grouped by category
python -m featurizer list-primitives --type transform --category
```

---

## Aggregation Primitives

Aggregation primitives are applied when traversing **backward relationships** (parent ← child). They reduce multiple rows to a single value.

### Basic Aggregations

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `sum` | Sum of all values | numeric | numeric | `SUM(amount)` |
| `min` | Minimum value | numeric | numeric | `MIN(amount)` |
| `max` | Maximum value | numeric | numeric | `MAX(amount)` |
| `mean` | Arithmetic mean (average) | numeric | numeric | `AVG(amount)` |
| `stddev` | Standard deviation | numeric | numeric | `STDDEV(amount)` |
| `variance` | Statistical variance | numeric | numeric | `VARIANCE(amount)` |
| `count` | Count of non-null values | categorical, index | numeric | `COUNT(status)` |
| `nunique` | Count of distinct values | categorical, index | numeric | `COUNT(DISTINCT status)` |

### Boolean Aggregations

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `all` | True if all values are true | boolean | boolean | `BOOL_AND(is_active)` |
| `any` | True if any value is true | boolean | boolean | `BOOL_OR(is_active)` |

### Ordered-Set Aggregations

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `median` | Median (50th percentile) | numeric | numeric | `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)` |
| `mode` | Most frequent value | categorical | categorical | `MODE() WITHIN GROUP (ORDER BY status)` |

### Statistical Aggregations

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `min_max_scale` | Min-max normalization (0-1) | numeric | numeric | `(val - MIN(val)) / (MAX(val) - MIN(val))` |
| `mean_deviation` | Average absolute deviation | numeric | numeric | `SUM(ABS(val - AVG(val))) / COUNT(val)` |
| `z_score` | Z-score (standard score) | numeric | numeric | `(val - AVG(val)) / STDDEV(val)` |
| `skewness` | Distribution asymmetry | numeric | numeric | `((val - AVG(val)) / STDDEV(val))^3` |
| `kurtosis` | Distribution tailedness | numeric | numeric | `((val - AVG(val)) / STDDEV(val))^4` |
| `harmonic_mean` | Harmonic mean (for rates) | numeric | numeric | `COUNT(val) / SUM(1.0/val)` |
| `geometric_mean` | Geometric mean (for growth) | numeric | numeric | `EXP(AVG(LOG(val)))` |

### Percentile Aggregations

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `p10` | 10th percentile | numeric | numeric | `PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY amount)` |
| `p25` | 25th percentile (first quartile) | numeric | numeric | `PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount)` |
| `p75` | 75th percentile (third quartile) | numeric | numeric | `PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount)` |
| `p90` | 90th percentile | numeric | numeric | `PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount)` |
| `p95` | 95th percentile | numeric | numeric | `PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount)` |
| `p99` | 99th percentile | numeric | numeric | `PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount)` |

### Distribution Metrics

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `iqr` | Interquartile range (P75 - P25) | numeric | numeric | `PERCENTILE_CONT(0.75) - PERCENTILE_CONT(0.25)` |
| `cv` | Coefficient of variation (STDDEV / MEAN) | numeric | numeric | `STDDEV(amount) / NULLIF(AVG(amount), 0)` |
| `range` | Range (MAX - MIN) | numeric | numeric | `MAX(amount) - MIN(amount)` |

### Temporal Metrics

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `event_rate` | Events per unit time | categorical, index | numeric | `COUNT(*) / EXTRACT(EPOCH FROM MAX(ts) - MIN(ts))` |
| `time_span` | Time span between first and last event | date, timestamp, index | numeric | `EXTRACT(EPOCH FROM MAX(ts) - MIN(ts))` |

### Inter-Event Gap Statistics

Compute statistics over the time gaps between consecutive events. Require `temporal_ix` on the child entity.

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `gap_mean` | Mean inter-event gap duration | date, timestamp, index | numeric | `AVG(ts - LAG(ts) OVER (ORDER BY ts))` |
| `gap_stddev` | Standard deviation of inter-event gaps | date, timestamp, index | numeric | `STDDEV(ts - LAG(ts) OVER (ORDER BY ts))` |
| `gap_min` | Minimum inter-event gap duration | date, timestamp, index | numeric | `MIN(ts - LAG(ts) OVER (ORDER BY ts))` |
| `gap_max` | Maximum inter-event gap duration | date, timestamp, index | numeric | `MAX(ts - LAG(ts) OVER (ORDER BY ts))` |
| `gap_cv` | Coefficient of variation of inter-event gaps | date, timestamp, index | numeric | `STDDEV(gap) / NULLIF(AVG(gap), 0)` |

### Temporal Patterns

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `burstiness` | Goh-Barabasi burstiness index (-1 to 1) | date, timestamp, index | numeric | `(STDDEV(gap) - AVG(gap)) / NULLIF(STDDEV(gap) + AVG(gap), 0)` |

A burstiness value of 1 indicates highly bursty behavior (events clustered together), 0 indicates random (Poisson) arrivals, and -1 indicates perfectly regular spacing.

### Categorical Distribution

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `entropy` | Shannon entropy of categorical distribution | categorical | numeric | `-SUM(p * LN(p))` where p = proportion |
| `hhi` | Herfindahl-Hirschman Index (concentration) | categorical | numeric | `SUM(p^2)` where p = proportion |

Higher entropy indicates more uniform distribution; lower indicates dominance by few categories. HHI ranges from 1/N (perfectly uniform) to 1 (single category dominates).

### Inequality

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `gini` | Gini coefficient (inequality measure, 0-1) | numeric | numeric | `2 * SUM(rank * val) / (n * SUM(val)) - (n+1)/n` |

A Gini of 0 means perfect equality; 1 means maximum inequality.

### Sequence Features

These primitives analyze sequential patterns in categorical event streams. They require `temporal_ix` on the child entity.

| Name | Description | Input Types | Output | SQL Example |
|------|-------------|-------------|--------|-------------|
| `ngram_2_freq` | Bigram frequency distribution | categorical | numeric | `COUNT(DISTINCT val->LEAD(val)) / COUNT(*)` |
| `ngram_3_freq` | Trigram frequency distribution | categorical | numeric | `COUNT(DISTINCT val->LEAD(val,1)->LEAD(val,2)) / COUNT(*)` |
| `sequence_entropy` | Transition entropy of categorical sequences | categorical | numeric | `-SUM(p_ij * LN(p_ij))` over transition matrix |
| `longest_streak` | Longest consecutive streak of same value | categorical, boolean | numeric | `MAX(streak_length)` using gaps-and-islands |

### Temporal Interval Support

Most basic aggregations support temporal intervals (e.g., `P7D`, `P30D`). When an interval is specified and the entity has a `temporal_ix`, the aggregation is filtered to include only records within that time window:

```sql
-- Example: SUM with P7D interval
SUM(amount) FILTER (WHERE daterange((aod.as_of_date - interval 'P7D')::date, aod.as_of_date::date, '[]') @> order_date)
```

---

## Transformation Primitives

Transformation primitives are applied to features **within an entity**. They transform individual values or compute window functions.

### Basic Transformers

| Name | Description | Input Types | Output | SQL |
|------|-------------|-------------|--------|-----|
| `identity` | Pass-through | numeric, categorical, text | same | `column_name` |

### Math Transformers

| Name | Description | Input Types | Output | SQL |
|------|-------------|-------------|--------|-----|
| `abs` | Absolute value | numeric | numeric | `ABS(value)` |
| `exp` | Exponential (e^x) | numeric | numeric | `EXP(value)` |
| `ln` | Natural logarithm | numeric | numeric | `LN(value)` |
| `log` | Base-10 logarithm | numeric | numeric | `LOG(value)` |
| `sqrt` | Square root | numeric | numeric | `SQRT(value)` |
| `cbrt` | Cube root | numeric | numeric | `CBRT(value)` |
| `sign` | Sign (-1, 0, or 1) | numeric | numeric | `SIGN(value)` |
| `ceil` | Round up | numeric | numeric | `CEIL(value)` |
| `floor` | Round down | numeric | numeric | `FLOOR(value)` |
| `trunc` | Truncate decimals | numeric | numeric | `TRUNC(value)` |

### Text Transformers

| Name | Description | Input Types | Output | SQL |
|------|-------------|-------------|--------|-----|
| `num_chars` | Character count | text | numeric | `CHAR_LENGTH(text)` |

### Date Part Extractors

| Name | Description | Input Types | Output | SQL |
|------|-------------|-------------|--------|-----|
| `day` | Day of month (1-31) | date, timestamp, index | categorical | `TO_CHAR(date, 'DD')` |
| `dow` | ISO day of week (1=Mon) | date, timestamp, index | categorical | `TO_CHAR(date, 'ID')` |
| `dom` | Day of month | date, timestamp, index | categorical | `TO_CHAR(date, 'DD')` |
| `doy` | Day of year (1-366) | date, timestamp, index | categorical | `TO_CHAR(date, 'DDD')` |
| `year` | Four-digit year | date, timestamp, index | categorical | `TO_CHAR(date, 'YYYY')` |
| `month` | Month (1-12) | date, timestamp, index | categorical | `TO_CHAR(date, 'MM')` |
| `hour` | Hour (0-23) | date, timestamp, index | categorical | `TO_CHAR(date, 'HH24')` |
| `quarter` | Quarter (1-4) | date, timestamp, index | categorical | `TO_CHAR(date, 'Q')` |
| `week` | Week of month | date, timestamp, index | categorical | `TO_CHAR(date, 'W')` |
| `week_of_year` | Week of year (1-53) | date, timestamp, index | categorical | `TO_CHAR(date, 'WW')` |
| `century` | Century number | date, timestamp, index | categorical | `TO_CHAR(date, 'CC')` |
| `tz` | Time zone | date, timestamp, index | categorical | `TO_CHAR(date, 'TZ')` |
| `tz_offset` | Time zone offset | date, timestamp, index | categorical | `TO_CHAR(date, 'OF')` |

### Binning Transformers

| Name | Description | Input Types | Output | Categories |
|------|-------------|-------------|--------|------------|
| `hourly_bin` | Time of day category | date, timestamp | categorical | night, early_morning, morning, midday, afternoon, evening |
| `daily_bin` | Weekday/weekend | date, timestamp | categorical | weekday, weekend |

### Cyclical Encoding

Cyclical encoding represents periodic features (hour, month, day of week) as sin/cos pairs, preserving the circular nature of time:

| Name | Description | Period | Output |
|------|-------------|--------|--------|
| `cyclic_hour` | Hour as sin/cos | 24 | Two features: `*_sin`, `*_cos` |
| `cyclic_month` | Month as sin/cos | 12 | Two features: `*_sin`, `*_cos` |
| `cyclic_day` | Day of week as sin/cos | 7 | Two features: `*_sin`, `*_cos` |

**Example**: For hour=6, `cyclic_hour_sin` = sin(6 × 2π/24) ≈ 0.707, `cyclic_hour_cos` = cos(6 × 2π/24) ≈ 0.707

### Cumulative Window Functions

These require `temporal_ix` on the entity and compute running aggregates:

| Name | Description | Input Types | SQL Pattern |
|------|-------------|-------------|-------------|
| `cum_sum` | Cumulative sum | numeric | `SUM(val) OVER (PARTITION BY id ORDER BY date)` |
| `cum_mean` | Cumulative mean | numeric | `AVG(val) OVER (PARTITION BY id ORDER BY date)` |
| `cum_max` | Cumulative maximum | numeric | `MAX(val) OVER (PARTITION BY id ORDER BY date)` |
| `cum_min` | Cumulative minimum | numeric | `MIN(val) OVER (PARTITION BY id ORDER BY date)` |
| `cum_count` | Cumulative count | categorical, index | `COUNT(val) OVER (PARTITION BY id ORDER BY date)` |

### Value Access (Window Functions)

| Name | Description | Input Types | SQL Pattern |
|------|-------------|-------------|-------------|
| `first` | First value in partition | all | `FIRST_VALUE(val) OVER (...)` |
| `last` | Last value in partition | all | `LAST_VALUE(val) OVER (...)` |
| `previous` | Previous row's value | numeric | `LAG(val) OVER (...)` |
| `diff` | Difference from previous | numeric | `val - LAG(val) OVER (...)` |
| `time_since_previous` | Time since last record | date, timestamp | `date - LAG(date) OVER (...)` |

### Lag Transformers

Access values from N periods ago:

| Name | Description | Periods |
|------|-------------|---------|
| `lag_1` | Value from 1 period ago | 1 |
| `lag_3` | Value from 3 periods ago | 3 |
| `lag_7` | Value from 7 periods ago | 7 |

### Rolling Statistics

Compute statistics over a sliding window of N rows:

| Name | Description | Window |
|------|-------------|--------|
| `rolling_mean_3` | 3-period rolling mean | 3 |
| `rolling_mean_7` | 7-period rolling mean | 7 |
| `rolling_mean_14` | 14-period rolling mean | 14 |
| `rolling_std_3` | 3-period rolling std dev | 3 |
| `rolling_std_7` | 7-period rolling std dev | 7 |
| `rolling_std_14` | 14-period rolling std dev | 14 |
| `rolling_median_5` | 5-period rolling median | 5 |
| `rolling_median_7` | 7-period rolling median | 7 |
| `rolling_iqr_7` | 7-period rolling IQR | 7 |
| `rolling_iqr_14` | 14-period rolling IQR | 14 |

### Exponential Moving Average

EMA gives more weight to recent values:

| Name | Description | Window | Decay |
|------|-------------|--------|-------|
| `ema_7` | 7-period EMA | 7 | 0.25 |
| `ema_14` | 14-period EMA | 14 | 0.15 |

### Holt-Winters Components

Decompose time series into level (smoothed average) and trend (slope):

| Name | Description | Window |
|------|-------------|--------|
| `holt_winters_level_7` | 7-period smoothed level | 7 |
| `holt_winters_level_14` | 14-period smoothed level | 14 |
| `holt_winters_trend_7` | 7-period trend (slope) | 7 |
| `holt_winters_trend_14` | 14-period trend (slope) | 14 |

### Percentage Change

Compute relative change from N periods ago:

| Name | Description | Periods |
|------|-------------|---------|
| `pct_change_1` | % change from 1 period ago | 1 |
| `pct_change_3` | % change from 3 periods ago | 3 |

### Distribution Functions

| Name | Description | SQL Pattern |
|------|-------------|-------------|
| `cdf` | Cumulative distribution value | `CUME_DIST() OVER (...)` |
| `percent_rank` | Relative rank (0-1) | `PERCENT_RANK() OVER (...)` |
| `ntile` | Divide into N groups (default: 5) | `NTILE(5) OVER (...)` |

### Boolean Checks

| Name | Description | Input Types | Output |
|------|-------------|-------------|--------|
| `is_null` | Check if null | numeric, categorical, date | boolean |
| `in_array` | Check membership | numeric, categorical, date | boolean |

### Population Window Transformers

These transformers normalize values across the entire population of entities rather than within a single entity's history. Useful for comparing an entity's feature values to the overall distribution.

| Name | Description | Input Types | Output | SQL Pattern |
|------|-------------|-------------|--------|-------------|
| `cross_entity_zscore` | Z-score across all entities | numeric | numeric | `(value - AVG(value) OVER ()) / NULLIF(STDDEV(value) OVER (), 0)` |
| `cross_entity_percentile` | Percentile rank across all entities | numeric | numeric | `PERCENT_RANK() OVER (ORDER BY value)` |

### Change-Point Detection

These transformers detect shifts in feature behavior over time. Useful for identifying regime changes, anomalies, and trend breaks.

| Name | Description | Input Types | Window | SQL Pattern |
|------|-------------|-------------|--------|-------------|
| `mean_shift_ratio_7` | Ratio of recent 7-period mean to overall mean | numeric | 7 | `AVG(val) OVER (... ROWS 6 PRECEDING) / NULLIF(AVG(val) OVER (), 0)` |
| `mean_shift_ratio_14` | Ratio of recent 14-period mean to overall mean | numeric | 14 | `AVG(val) OVER (... ROWS 13 PRECEDING) / NULLIF(AVG(val) OVER (), 0)` |
| `cusum` | Cumulative sum of deviations from target mean | numeric | all | `SUM(val - target_mean) OVER (PARTITION BY id ORDER BY date)` |

A `mean_shift_ratio` near 1.0 indicates stability; values significantly above or below 1.0 signal a change point. CUSUM accumulates deviations and is sensitive to sustained small shifts that rolling averages might miss.

---

## Adding Custom Primitives

### Custom Aggregation

```python
from featurizer.primitives.aggregations import Aggregator
from featurizer.primitives.utils import register_aggregation

class MedianAbsoluteDeviation(Aggregator):
    def __init__(self):
        super().__init__(name='mad')

    def _build_aggregate_expression(self, feature, interval):
        # Custom SQL expression
        return f"PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS({feature.name} - (SELECT AVG({feature.name}) FROM ...)))"

mad = MedianAbsoluteDeviation()
register_aggregation("mad", mad)
```

### Custom Transformation

```python
from featurizer.primitives.transformations import Transformer
from featurizer.primitives.abstractions import Feature
from featurizer.primitives.utils import register_transformer

class Normalize(Transformer):
    def __init__(self):
        super().__init__(name='normalize', input_types=['numeric'])

    def __call__(self, parent, feature):
        if feature.type not in self.input_types:
            return feature
        return Feature(
            name=f'"NORMALIZE({feature.entity.alias}.{feature.name})"',
            type='numeric',
            definition=f"({feature.name} - MIN({feature.name}) OVER ()) / NULLIF(MAX({feature.name}) OVER () - MIN({feature.name}) OVER (), 0)",
            entity=parent,
            stack_depth=feature.stack_depth + 1,
        )

normalize = Normalize()
register_transformer("normalize", normalize)
```

**Important**: Transformers must return **new** `Feature` instances (never mutate the input) to preserve hashing semantics.

---

## Default Primitives

The Featurizer uses these primitives by default:

### Default Aggregations
`count`, `mean`, `sum`, `stddev`

### Default Transformations
`identity`, `abs`, `cum_sum`, `day`, `dow`, `month`, `lag_1`, `lag_3`, `lag_7`, `rolling_mean_3`, `rolling_std_7`, `rolling_median_7`, `rolling_iqr_7`, `ema_7`, `holt_winters_level_7`, `holt_winters_trend_7`, `pct_change_1`

To use additional primitives, request them via `get_aggregations()` or `get_transformers()`:

```python
from featurizer.primitives.utils import get_aggregations, get_transformers

# Request specific aggregations
aggs = get_aggregations(["count", "mean", "median", "mode"])

# Get all available transformations
all_transforms = get_transformers()  # No argument = all registered
```
