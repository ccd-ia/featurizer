# Featurizer

**Automated feature engineering for temporal data using PostgreSQL.**

Featurizer implements [Deep Feature Synthesis](https://dai.lids.mit.edu/projects/deep-feature-synthesis/) (DFS) for relational databases with first-class support for temporal semantics. Given a schema of entities and relationships, it automatically synthesizes hundreds of meaningful features by traversing the entity graph, applying aggregations across relationships, and generating time-windowed statistics.


## Why Featurizer?

Feature engineering is often the most time-consuming part of building machine learning models on relational data. Featurizer automates this process by:

-   **Traversing relationships automatically**: Define your entity graph once; Featurizer handles the joins and aggregations.
-   **Respecting temporal semantics**: Point-in-time correct features via as-of joins prevent data leakage in time-series ML.
-   **Generating pure SQL**: No data movement required&#x2014;features are computed where your data lives.
-   **Providing extensible primitives**: 19 aggregations and 66 transformations out of the box, with a simple API for custom ones.


## Quick Example

```python
from featurizer import Featurizer

# Load configuration defining entities and relationships
f = Featurizer("config.yaml")

# Get the generated SQL query
print(f.query)

# Or execute directly and get a DataFrame
df = f.to_dataframe()
```

A simple configuration with customers and orders generates features like:

-   `SUM(orders.amount)` &#x2013; Total order value per customer
-   `COUNT(orders.status|interval=P7D)` &#x2013; Orders in the last 7 days
-   `ROLLING_MEAN_7(orders.amount)` &#x2013; 7-day moving average of order amounts
-   `HOLT_WINTERS_TREND_14(orders.amount)` &#x2013; Trend direction over 14 periods



## Supported Feature Primitives

-   **Aggregations:** count, mean, sum, stddev, median, mode, nunique, min/max, variance, harmonic/geometric means, deviation metrics, and more; interval windows are supported when entities declare temporal indexes.
-   **Scalar transforms:** identity, abs/exp/log/sqrt, ceil/floor/trunc, text length, boolean checks (is\_null, in\_array), binary arithmetic/logic operators, and categorical/date part extractors (day, dow, month, hour, etc.).
-   **Temporal windows:** cumulative sums/means/min/max/count, lag/lead-style `previous`, diff/time\_since\_previous, percentile-based distribution metrics (CDF, percent\_rank, ntile).
-   **Rolling statistics:** moving averages and standard deviations (3/7/14 rows), rolling medians and IQRs, exponential moving averages, Holt-Winters-inspired level/trend regressions, percentage change over configurable lags.
-   **Cyclical encoding:** sine/cosine projections for hours/months/days plus hourly/daily binning helpers.
-   **Temporal joins:** relationship-level `temporal` blocks enable as-of lateral joins with optional grace periods, letting target rows pull the latest parent record as of each timestamp.


## Feature Primitive Registries

-   Aggregation primitives register themselves via `register_aggregation` in `featurizer/primitives/aggregations.py`. New aggregators should call `register_aggregation("my_name", my_callable)` so they are discoverable without editing `featurizer/featurizer.py`.

-   Transformation primitives follow the same pattern using `register_transformer`. Avoid mutating the incoming `Feature`; always return a new instance (or list of instances) for deterministic hashing.

-   The runtime loads a default subset (`count`, `mean`, `sum`, `stddev`, `identity`, `abs`, `cum_sum`, `day`, `dow`, `month`, `lag_1`, `lag_3`, `lag_7`, `rolling_mean_3`, `rolling_std_7`, `rolling_median_7`, `rolling_iqr_7`, `ema_7`, `holt_winters_level_7`, `holt_winters_trend_7`, `pct_change_1`) and can be extended by requesting specific names via `get_aggregations` / `get_transformers`.

-   Example registration:

    ```python
      # featurizer/primitives/aggregations.py
      class SumSquares(Aggregator):
          def __init__(self):
              super().__init__(name="sum_squares")

          def _build_aggregate_expression(self, feature, interval):
              base = super()._build_aggregate_expression(feature, interval)
              return base.replace(feature.name, f"{feature.name} * {feature.name}")

      sum_squares = SumSquares()
      register_aggregation("sum_squares", sum_squares)
    ```

    After registration, request it from the planner with:

    ```python
      custom_aggs = get_aggregations(["sum_squares", "mean"])
    ```


## Debugging & Logging

-   The planner emits structured debug logs through [loguru](https://github.com/Delgan/loguru). Set `logger.remove()=/=logger.add(...)` in your entry point to adjust verbosity.
-   Set `FEATURIZER_DEBUG=1` (or pass `debug=True` to `Featurizer`) to mirror planner milestones via [icecream](https://github.com/gruns/icecream). The emitted payloads show traversal depth, aggregation counts, and transformation totals.
-   Planner/renderer/executor components live in `featurizer/planner.py`, `featurizer/sql.py`, and `featurizer/executor.py` respectively&#x2014;each can be imported independently for bespoke workflows.


## Temporal Joins

-   Declare `temporal` blocks on relationships (e.g., `mode: as_of`, optional `grace`) to pull the most recent parent record as of each target timestamp. The planner materializes these as `LEFT JOIN LATERAL` clauses so the nearest match&#x2014;and only that match&#x2014;feeds downstream transformations.
-   Source entities can override the timestamp column via `child_timestamp`; otherwise their declared `temporal_ix` is used. Temporal joins fall back to static key joins when either side lacks a temporal index.


## Testing

Run the test suite with:

```sh
uv run pytest -q

# With coverage report
uv run pytest --cov=featurizer --cov-report=term-missing
```



## Project Map

-   `featurizer/` &#x2013; Core modules (planner, sql, executor, validation)
-   `featurizer/primitives/` &#x2013; Aggregation and transformation primitives
-   `tests/` &#x2013; Test suite (120 tests, 93.40% coverage)
-   `examples/` &#x2013; Four self-contained examples with SQLite databases
-   `docs/` &#x2013; Session history and primitives reference
