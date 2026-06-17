
# Table of Contents

1.  [Featurizer](#orgddcf3aa)
    1.  [Why Featurizer?](#org1e0062f)
    2.  [Quick Example](#orgd4b4e7d)
    3.  [Selecting primitives](#org9b66329)
    4.  [Examples](#orgf381421)
    5.  [Supported Feature Primitives](#orgde70c50)
    6.  [Feature Primitive Registries](#org7cae9fd)
    7.  [Debugging & Logging](#orga74b4b3)
    8.  [Temporal Joins](#orgc853b3f)
    9.  [Visualization](#org328866b)
    10. [Testing](#org3c58faa)
    11. [Project Map](#org1a0f823)



<a id="orgddcf3aa"></a>

# Featurizer

**Automated feature engineering for temporal data using PostgreSQL.**

[![CI](https://github.com/nanounanue/featurizer/actions/workflows/test.yml/badge.svg)](https://github.com/nanounanue/featurizer/actions/workflows/test.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue.svg)](https://www.python.org/downloads/)
[![Type checked: basedpyright strict](https://img.shields.io/badge/types-basedpyright%20strict-2a6db0.svg)](pyrightconfig.json)

Featurizer implements [Deep Feature Synthesis](https://dai.lids.mit.edu/projects/deep-feature-synthesis/) (DFS) for relational
databases with first-class support for temporal semantics. Given a
schema of entities and relationships, it automatically synthesizes
hundreds of meaningful features by traversing the entity graph,
applying aggregations across relationships, and generating
time-windowed statistics.

<p align="center"><img src="docs/images/architecture.svg" width="820" alt="Featurizer pipeline: config.yaml → Validator → Planner → SQL Renderer → Executor, with the φ-bridge precompute feeding the Planner and a point-in-time-correct feature matrix as output"/></p>


<a id="org1e0062f"></a>

## Why Featurizer?

Feature engineering is often the most time-consuming part of building
machine learning models on relational data. Featurizer automates this
process by:

-   **Traversing relationships automatically**: Define your entity graph
    once; Featurizer handles the joins and aggregations.
-   **Respecting temporal semantics**: Point-in-time correct features via
    as-of joins prevent data leakage in time-series ML.
-   **Generating pure SQL**: No data movement required&#x2014;features are
    computed where your data lives.
-   **Providing extensible primitives**: 69 aggregations and 83
    transformations out of the box, with a simple API for custom ones.


<a id="orgd4b4e7d"></a>

## Quick Example

    from featurizer import Featurizer
    
    # Load configuration defining entities and relationships
    f = Featurizer("config.yaml")
    
    # Get the generated SQL query
    print(f.query)
    
    # Or execute directly and get a DataFrame
    df = f.to_dataframe()

A simple configuration with customers and orders generates features like:

-   `SUM(orders.amount)` &#x2013; Total order value per customer
-   `COUNT(orders.status|interval=P7D)` &#x2013; Orders in the last 7 days
-   `ROLLING_MEAN_7(orders.amount)` &#x2013; 7-day moving average of order amounts
-   `HOLT_WINTERS_TREND_14(orders.amount)` &#x2013; Trend direction over 14 periods


<a id="org9b66329"></a>

## Selecting primitives

By default Featurizer applies a curated active set (`count`, `mean`, `sum`,
`stddev`, `min`, `max`, `median`, `nunique`, `recency`, `tenure`, plus the
default transformers). Override per-config with optional `aggregations:` and
`transformations:` lists &#x2014; any registered primitive name is valid (run
`python -m featurizer list-primitives` to see them all); an unknown name raises
a validation error with a "did you mean?" suggestion.

    target: customers
    max_depth: 2
    intervals: [P7D, P1M]
    aggregations: [sum, mean, recency, gap_cv, entropy]
    transformations: [identity, lag_1, rolling_mean_7]
    entities:
      # ...


<a id="orgf381421"></a>

## Examples

The `examples/` directory contains four self-contained tutorials with
SQLite databases:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Example</th>
<th scope="col" class="org-left">Scenario</th>
<th scope="col" class="org-left">Concepts</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><a href="examples/01-basic-aggregations/">01-basic-aggregations</a></td>
<td class="org-left">E-commerce (Customers → Orders)</td>
<td class="org-left">Parent-child relationships, time windows</td>
</tr>

<tr>
<td class="org-left"><a href="examples/02-temporal-joins/">02-temporal-joins</a></td>
<td class="org-left">Healthcare (Patients → Care Plans)</td>
<td class="org-left">As-of joins, grace periods</td>
</tr>

<tr>
<td class="org-left"><a href="examples/03-deep-nesting/">03-deep-nesting</a></td>
<td class="org-left">Retail (Stores → Orders → Products → Suppliers)</td>
<td class="org-left">Multi-level traversal (depth=3)</td>
</tr>

<tr>
<td class="org-left"><a href="examples/04-custom-primitives/">04-custom-primitives</a></td>
<td class="org-left">Finance (Accounts → Transactions)</td>
<td class="org-left">Custom aggregations and transformations</td>
</tr>
</tbody>
</table>

To run an example:

    cd examples/01-basic-aggregations/
    
    # Generate sample data
    python create_data.py
    
    # Run feature generation (shows summary)
    python run_example.py
    
    # View generated SQL
    python run_example.py --show-sql
    
    # Execute and save results
    python run_example.py --execute --output features.csv

Each example includes a Jupyter notebook (`tutorial.ipynb`) for
interactive exploration.


<a id="orgde70c50"></a>

## Supported Feature Primitives

-   **Aggregations:** count, mean, sum, stddev, median, mode, nunique,
    min/max, variance, harmonic/geometric means, deviation metrics, and
    more; interval windows are supported when entities declare temporal
    indexes.
-   **Percentiles:** p10, p25, p75, p90, p95, p99 via ordered-set
    aggregates.
-   **Distribution metrics:** interquartile range (iqr), coefficient of
    variation (cv), range (max minus min).
-   **Inter-event gap statistics:** gap<sub>mean</sub>, gap<sub>stddev</sub>, gap<sub>min</sub>,
    gap<sub>max</sub>, gap<sub>cv</sub> &#x2014; computed via `SubqueryAggregator` over consecutive
    event timestamps.
-   **Temporal patterns:** burstiness (Goh-Barabasi index, -1 to 1),
    event<sub>rate</sub>, time<sub>span</sub>.
-   **Categorical distribution:** entropy (Shannon), hhi
    (Herfindahl-Hirschman Index).
-   **Inequality:** gini coefficient (0 = equality, 1 = maximum
    inequality).
-   **Sequence features:** ngram<sub>2</sub><sub>freq</sub>, ngram<sub>3</sub><sub>freq</sub>, sequence<sub>entropy</sub>,
    longest<sub>streak</sub> &#x2014; analyze sequential patterns in categorical event
    streams.
-   **Scalar transforms:** identity, abs/exp/log/sqrt, ceil/floor/trunc,
    text length, boolean checks (is<sub>null</sub>, in<sub>array</sub>), binary
    arithmetic/logic operators, and categorical/date part extractors (day,
    dow, month, hour, etc.).
-   **Temporal windows:** cumulative sums/means/min/max/count,
    lag/lead-style `previous`, diff/time<sub>since</sub><sub>previous</sub>, percentile-based
    distribution metrics (CDF, percent<sub>rank</sub>, ntile).
-   **Rolling statistics:** moving averages and standard deviations (3/7/14
    rows), rolling medians and IQRs, exponential moving averages,
    Holt-Winters-inspired level/trend regressions, percentage change over
    configurable lags.
-   **Cyclical encoding:** sine/cosine projections for hours/months/days
    plus hourly/daily binning helpers.
-   **Population windows:** cross<sub>entity</sub><sub>zscore</sub>, cross<sub>entity</sub><sub>percentile</sub>
    for normalizing across all entities.
-   **Change-point detection:** mean<sub>shift</sub><sub>ratio</sub><sub>7</sub>/14, cusum for
    identifying regime changes.
-   **Temporal joins:** relationship-level `temporal` blocks enable as-of
    lateral joins with optional grace periods, letting target rows pull
    the latest parent record as of each timestamp.


<a id="org7cae9fd"></a>

## Feature Primitive Registries

-   Aggregation primitives register themselves via `register_aggregation`
    in `featurizer/primitives/aggregations.py`. New aggregators should
    call `register_aggregation("my_name", my_callable)` so they are
    discoverable without editing `featurizer/featurizer.py`.

-   Transformation primitives follow the same pattern using
    `register_transformer`. Avoid mutating the incoming `Feature`; always
    return a new instance (or list of instances) for deterministic
    hashing.

-   The runtime loads a default subset (`count`, `mean`, `sum`, `stddev`,
    `identity`, `abs`, `cum_sum`, `day`, `dow`, `month`, `lag_1`, `lag_3`,
    `lag_7`, `rolling_mean_3`, `rolling_std_7`, `rolling_median_7`,
    `rolling_iqr_7`, `ema_7`, `holt_winters_level_7`,
    `holt_winters_trend_7`, `pct_change_1`) and can be extended by
    requesting specific names via `get_aggregations` / `get_transformers`.
    In total there are 69 aggregations and 83 transformers (152
    registered primitives). Use `python -m featurizer list-primitives` to
    discover all registered primitives. Peer-group, spatial second-table,
    and φ-bridge features are produced by dedicated planner passes rather
    than the primitive registry.

-   Example registration:
    
        # featurizer/primitives/aggregations.py
        class SumSquares(Aggregator):
            def __init__(self):
                super().__init__(name="sum_squares")
        
            def _build_aggregate_expression(self, feature, interval):
                base = super()._build_aggregate_expression(feature, interval)
                return base.replace(feature.name, f"{feature.name} * {feature.name}")
        
        sum_squares = SumSquares()
        register_aggregation("sum_squares", sum_squares)
    
    After registration, request it from the planner with:
    
        custom_aggs = get_aggregations(["sum_squares", "mean"])


<a id="orga74b4b3"></a>

## Debugging & Logging

-   The planner emits structured debug logs through
    [loguru](https://github.com/Delgan/loguru). Set
    `logger.remove()=/=logger.add(...)` in your entry point to adjust
    verbosity.
-   Set `FEATURIZER_DEBUG=1` (or pass `debug=True` to `Featurizer`) to
    mirror planner milestones via
    [icecream](https://github.com/gruns/icecream). The emitted payloads
    show traversal depth, aggregation counts, and transformation totals.
-   Planner/renderer/executor components live in `featurizer/planner.py`,
    `featurizer/sql.py`, and `featurizer/executor.py` respectively&#x2014;each
    can be imported independently for bespoke workflows.


<a id="orgc853b3f"></a>

## Temporal Joins

-   Declare `temporal` blocks on relationships (e.g., `mode: as_of`,
    optional `grace`) to pull the most recent parent record as of each
    target timestamp. The planner materializes these as
    `LEFT JOIN LATERAL` clauses so the nearest match&#x2014;and only that
    match&#x2014;feeds downstream transformations.
-   Source entities can override the timestamp column via
    `child_timestamp`; otherwise their declared `temporal_ix` is used.
    Temporal joins fall back to static key joins when either side lacks a
    temporal index.


<a id="org328866b"></a>

## Visualization

After materializing the feature matrix, `FeaturizerViz` turns it into
diagnostic plots. It is an optional extra&#x2014;install with
`uv sync --extra viz` (matplotlib, seaborn, plotly, scikit-learn, scipy,
networkx, umap-learn).

    from featurizer import Featurizer, FeaturizerViz
    
    f = Featurizer("config.yaml")
    viz = FeaturizerViz.from_featurizer(f)          # resolves the entity id column
    
    # Distribution & data quality
    viz.feature_summary_table()                     # mean/std/skewness/% missing
    viz.plot_feature_distributions(kind="violin")
    viz.plot_missing_heatmap()
    
    # Redundancy & importance
    viz.plot_correlation_clustermap()
    viz.plot_feature_importance(target_col="label") # mutual_info | f_classif | f_regression
    viz.plot_feature_variance()
    
    # Entity structure (a single as-of slice)
    viz.plot_entity_embedding(method="umap")        # 'umap' | 'tsne' | 'pca'
    viz.plot_entity_dendrogram()
    
    # Per-entity time series
    viz.plot_feature_timeseries(entity_id=42, normalize=True)
    viz.plot_entity_feature_heatmap(entity_id=42)

`from_featurizer` reads the target entity's id column so the matrix's
`(as_of_date, <id>)` index is interpreted correctly; pass a plain DataFrame
to `FeaturizerViz(df, entity_col`&#x2026;)= if you materialized it elsewhere.
Methods that need a complete matrix (importance, embedding, dendrogram)
median-impute a **local copy**&#x2014;the stored matrix keeps its NULLs as signal.


<a id="org3c58faa"></a>

## Testing

Run the test suite with:

    uv run pytest -q
    
    # With coverage report
    uv run pytest --cov=featurizer --cov-report=term-missing


<a id="org1a0f823"></a>

## Project Map

-   `featurizer/` &#x2013; Core modules (planner, sql, executor, validation)
-   `featurizer/primitives/` &#x2013; Aggregation and transformation primitives
-   `tests/` &#x2013; Test suite (385 tests)
-   `examples/` &#x2013; Four self-contained examples with SQLite databases
-   `docs/` &#x2013; Session history and primitives reference

