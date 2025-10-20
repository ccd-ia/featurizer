# Featurizer

DFS implementation for PostgreSQL temporal datasources.

## Development Setup
1. Install [uv](https://docs.astral.sh/uv/) (version 0.8 or newer).
2. Sync dependencies and create the virtual environment:
   ```bash
   uv sync
   ```
3. Activate the environment for interactive use:
   ```bash
   source .venv/bin/activate
   ```
   Use `uv run <command>` for one-off executions without activating the shell.

## Quick Smoke Test
Render the demo SQL query defined in `featurizer/featurizer.yaml`:
```bash
uv run python -c 'from featurizer import Featurizer; print(Featurizer("featurizer/featurizer.yaml").query)'
```

## Supported Feature Primitives
- **Aggregations:** count, mean, sum, stddev, median, mode, nunique, min/max, variance, harmonic/geometric means, deviation metrics, and more; interval windows are supported when entities declare temporal indexes.
- **Scalar transforms:** identity, abs/exp/log/sqrt, ceil/floor/trunc, text length, boolean checks (is_null, in_array), binary arithmetic/logic operators, and categorical/date part extractors (day, dow, month, hour, etc.).
- **Temporal windows:** cumulative sums/means/min/max/count, lag/lead-style `previous`, diff/time_since_previous, percentile-based distribution metrics (CDF, percent_rank, ntile).
- **Rolling statistics:** moving averages and standard deviations (3/7/14 rows), rolling medians and IQRs, exponential moving averages, Holt-Winters-inspired level/trend regressions, percentage change over configurable lags.
- **Cyclical encoding:** sine/cosine projections for hours/months/days plus hourly/daily binning helpers.
- **Temporal joins:** relationship-level `temporal` blocks enable as-of lateral joins with optional grace periods, letting target rows pull the latest parent record as of each timestamp.

## Feature Primitive Registries
- Aggregation primitives register themselves via `register_aggregation` in `featurizer/primitives/aggregations.py`. New aggregators should call `register_aggregation("my_name", my_callable)` so they are discoverable without editing `featurizer/featurizer.py`.
- Transformation primitives follow the same pattern using `register_transformer`. Avoid mutating the incoming `Feature`; always return a new instance (or list of instances) for deterministic hashing.
- The runtime loads a default subset (`count`, `mean`, `sum`, `stddev`, `identity`, `abs`, `cum_sum`, `day`, `dow`, `month`, `lag_1`, `lag_3`, `lag_7`, `rolling_mean_3`, `rolling_std_7`, `rolling_median_7`, `rolling_iqr_7`, `ema_7`, `holt_winters_level_7`, `holt_winters_trend_7`, `pct_change_1`) and can be extended by requesting specific names via `get_aggregations` / `get_transformers`.
- Example registration:
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
- The planner emits structured debug logs through [loguru](https://github.com/Delgan/loguru). Set `logger.remove()`/`logger.add(...)` in your entry point to adjust verbosity.
- Set `FEATURIZER_DEBUG=1` (or pass `debug=True` to `Featurizer`) to mirror planner milestones via [icecream](https://github.com/gruns/icecream). The emitted payloads show traversal depth, aggregation counts, and transformation totals.
- Planner/renderer/executor components live in `featurizer/planner.py`, `featurizer/sql.py`, and `featurizer/executor.py` respectively—each can be imported independently for bespoke workflows.

## Temporal Joins
- Declare `temporal` blocks on relationships (e.g., `mode: as_of`, optional `grace`) to pull the most recent parent record as of each target timestamp. The planner materializes these as `LEFT JOIN LATERAL` clauses so the nearest match—and only that match—feeds downstream transformations.
- Source entities can override the timestamp column via `child_timestamp`; otherwise their declared `temporal_ix` is used. Temporal joins fall back to static key joins when either side lacks a temporal index.

## Dependencies
- Runtime dependencies (PyYAML, records, loguru, etc.) are declared in `pyproject.toml`; run `uv sync` after modifying them.
- `PyYAML` powers configuration parsing—ensure deployments vendor the matching version if you bundle the library.

## Testing
Run the test suite (once tests are added) with:
```bash
uv run pytest -q
```
