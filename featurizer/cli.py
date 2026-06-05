# coding: utf-8

"""Command-line interface for Featurizer.

Provides commands for discovering primitives, validating configs, and more.

Usage:
    python -m featurizer list-primitives [--type agg|transform|all] [--show-sql]
    python -m featurizer validate <config.yaml>
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Optional

from .primitives.utils import (
    list_aggregations,
    list_transformations,
)
from .validation import validate_config

# Primitive metadata for documentation
AGGREGATION_DOCS: Dict[str, Dict[str, Any]] = {
    "sum": {
        "description": "Sum of all values",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(amount)",
        "temporal": True,
    },
    "min": {
        "description": "Minimum value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "MIN(amount)",
        "temporal": True,
    },
    "max": {
        "description": "Maximum value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "MAX(amount)",
        "temporal": True,
    },
    "mean": {
        "description": "Arithmetic mean (average)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(amount)",
        "temporal": True,
    },
    "stddev": {
        "description": "Standard deviation",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "STDDEV(amount)",
        "temporal": True,
    },
    "variance": {
        "description": "Statistical variance",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "VARIANCE(amount)",
        "temporal": True,
    },
    "count": {
        "description": "Count of non-null values",
        "input_types": ["categorical", "index"],
        "output_type": "numeric",
        "sql_example": "COUNT(status)",
        "temporal": True,
    },
    "nunique": {
        "description": "Count of distinct values",
        "input_types": ["categorical", "index"],
        "output_type": "numeric",
        "sql_example": "COUNT(DISTINCT status)",
        "temporal": True,
    },
    "all": {
        "description": "True if all values are true (boolean AND)",
        "input_types": ["boolean"],
        "output_type": "boolean",
        "sql_example": "BOOL_AND(is_active)",
        "temporal": True,
    },
    "any": {
        "description": "True if any value is true (boolean OR)",
        "input_types": ["boolean"],
        "output_type": "boolean",
        "sql_example": "BOOL_OR(is_active)",
        "temporal": True,
    },
    "median": {
        "description": "Median value (50th percentile)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "mode": {
        "description": "Most frequent value",
        "input_types": ["categorical"],
        "output_type": "categorical",
        "sql_example": "MODE() WITHIN GROUP (ORDER BY status)",
        "temporal": False,
    },
    "min_max_scale": {
        "description": "Min-max normalized value (0-1 scale)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "(value - MIN(value)) / (MAX(value) - MIN(value))",
        "temporal": False,
    },
    "mean_deviation": {
        "description": "Average absolute deviation from mean",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(ABS(value - AVG(value))) / COUNT(value)",
        "temporal": False,
    },
    "z_score": {
        "description": "Z-score (standard score)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "(value - AVG(value)) / STDDEV(value)",
        "temporal": False,
    },
    "skewness": {
        "description": "Measure of distribution asymmetry",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "((value - AVG(value)) / STDDEV(value))^3",
        "temporal": False,
    },
    "kurtosis": {
        "description": "Measure of distribution tailedness",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "((value - AVG(value)) / STDDEV(value))^4",
        "temporal": False,
    },
    "harmonic_mean": {
        "description": "Harmonic mean (for rates and ratios)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "COUNT(value) / SUM(1.0/value)",
        "temporal": False,
    },
    "geometric_mean": {
        "description": "Geometric mean (for growth rates)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "EXP(AVG(LOG(value)))",
        "temporal": False,
    },
    # Percentile aggregations
    "p10": {
        "description": "10th percentile",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.10) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "p25": {
        "description": "25th percentile (first quartile)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "p75": {
        "description": "75th percentile (third quartile)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "p90": {
        "description": "90th percentile",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "p95": {
        "description": "95th percentile",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "p99": {
        "description": "99th percentile",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    # Distribution metrics
    "iqr": {
        "description": "Interquartile range (P75 - P25)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount) - PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount)",
        "temporal": False,
    },
    "cv": {
        "description": "Coefficient of variation (STDDEV / MEAN)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "STDDEV(amount) / NULLIF(AVG(amount), 0)",
        "temporal": False,
    },
    "range": {
        "description": "Range (MAX - MIN)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "MAX(amount) - MIN(amount)",
        "temporal": True,
    },
    # Temporal metrics
    "event_rate": {
        "description": "Events per unit time",
        "input_types": ["categorical", "index"],
        "output_type": "numeric",
        "sql_example": "COUNT(*) / EXTRACT(EPOCH FROM MAX(ts) - MIN(ts))",
        "temporal": True,
    },
    "time_span": {
        "description": "Time span between first and last event",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "EXTRACT(EPOCH FROM MAX(ts) - MIN(ts))",
        "temporal": True,
    },
    # Inter-event gap statistics
    "gap_mean": {
        "description": "Mean inter-event gap duration",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "AVG(ts - LAG(ts) OVER (ORDER BY ts))",
        "temporal": True,
    },
    "gap_stddev": {
        "description": "Standard deviation of inter-event gaps",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "STDDEV(ts - LAG(ts) OVER (ORDER BY ts))",
        "temporal": True,
    },
    "gap_min": {
        "description": "Minimum inter-event gap duration",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "MIN(ts - LAG(ts) OVER (ORDER BY ts))",
        "temporal": True,
    },
    "gap_max": {
        "description": "Maximum inter-event gap duration",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "MAX(ts - LAG(ts) OVER (ORDER BY ts))",
        "temporal": True,
    },
    "gap_cv": {
        "description": "Coefficient of variation of inter-event gaps",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "STDDEV(gap) / NULLIF(AVG(gap), 0)",
        "temporal": True,
    },
    # Temporal patterns
    "burstiness": {
        "description": "Goh-Barabasi burstiness index (-1 to 1)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "(STDDEV(gap) - AVG(gap)) / NULLIF(STDDEV(gap) + AVG(gap), 0)",
        "temporal": True,
    },
    # Categorical distribution
    "entropy": {
        "description": "Shannon entropy of categorical distribution",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "-SUM(p * LN(p)) where p = COUNT(val) / SUM(COUNT(val))",
        "temporal": False,
    },
    "hhi": {
        "description": "Herfindahl-Hirschman Index (concentration measure)",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "SUM(p^2) where p = COUNT(val) / SUM(COUNT(val))",
        "temporal": False,
    },
    # Inequality
    "gini": {
        "description": "Gini coefficient (inequality measure, 0-1)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "2 * SUM(rank * val) / (n * SUM(val)) - (n+1)/n",
        "temporal": False,
    },
    # Sequence features
    "ngram_2_freq": {
        "description": "Bigram frequency distribution of categorical sequences",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "COUNT(DISTINCT val || '->' || LEAD(val)) / COUNT(*)",
        "temporal": True,
    },
    "ngram_3_freq": {
        "description": "Trigram frequency distribution of categorical sequences",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "COUNT(DISTINCT val || '->' || LEAD(val,1) || '->' || LEAD(val,2)) / COUNT(*)",
        "temporal": True,
    },
    "sequence_entropy": {
        "description": "Transition entropy of categorical sequences",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "-SUM(p_ij * LN(p_ij)) over transition matrix",
        "temporal": True,
    },
    "longest_streak": {
        "description": "Longest consecutive streak of same value",
        "input_types": ["categorical", "boolean"],
        "output_type": "numeric",
        "sql_example": "MAX(streak_length) using gaps-and-islands",
        "temporal": True,
    },
    # As-of state
    "recency": {
        "description": "Days since the most recent event (aod - max event ts)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "aod.as_of_date - max(event_ts)",
        "temporal": True,
    },
    "tenure": {
        "description": "Days since the first observed event (age in system)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "aod.as_of_date - min(event_ts)",
        "temporal": True,
    },
    "age_in_system": {
        "description": "Alias of tenure: days since the first observed event",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "aod.as_of_date - min(event_ts)",
        "temporal": True,
    },
    "inter_event_hazard_proxy": {
        "description": "Events per day over the observed lifespan (count / tenure)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "count(*) / (aod.as_of_date - min(event_ts))",
        "temporal": True,
    },
    # Distributional reductions
    "theil": {
        "description": "Theil-T inequality index over positive values",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG((x/mean) * LN(x/mean))",
        "temporal": True,
    },
    "trimmed_mean_10": {
        "description": "Mean of values within the 10th-90th percentile range",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(x) WHERE x BETWEEN p10 AND p90",
        "temporal": True,
    },
    "median_absolute_deviation": {
        "description": "Robust spread: median(|x - median(x)|)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "percentile_cont(0.5) WITHIN GROUP (ORDER BY abs(x - median))",
        "temporal": True,
    },
    # Sequence / process-mining reductions
    "state_volatility": {
        "description": "Count of categorical value changes over time",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "count(*) WHERE prev IS DISTINCT FROM curr",
        "temporal": True,
    },
    "transition_matrix_summary": {
        "description": "Number of distinct observed (prev -> curr) transitions",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "count(DISTINCT (prev, curr))",
        "temporal": True,
    },
    "rework_count": {
        "description": "Count of consecutive repeats (self-loops, prev == curr)",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "count(*) WHERE prev = curr",
        "temporal": True,
    },
    "time_in_current_state": {
        "description": "Days since the most recent change of a categorical attribute",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "aod.as_of_date - max(ts WHERE value changed)",
        "temporal": True,
    },
    # Numeric-stream reductions
    "acf_1": {
        "description": "Lag-1 autocorrelation: corr(x_t, x_{t-1})",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "corr(x, LAG(x,1) OVER (ORDER BY ts))",
        "temporal": True,
    },
    "variance_ratio": {
        "description": "Variance ratio: var(value) / var(first difference)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "var_samp(x) / NULLIF(var_samp(x - LAG(x)), 0)",
        "temporal": True,
    },
    "cosinor_amplitude_weekly": {
        "description": "Weekly cosinor amplitude (sin/cos regression approximation)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "sqrt(regr_slope(x,sin)^2 + regr_slope(x,cos)^2)",
        "temporal": True,
    },
    # Two-window distributional drift (interval-only)
    "kl_drift": {
        "description": "KL divergence: recent vs prior-window category distribution",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "SUM(p_recent * LN(p_recent / p_baseline)) over shared support",
        "temporal": True,
    },
    "wasserstein_drift": {
        "description": "Quantile L1 drift: recent vs prior-window numeric distribution",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "|q10_r - q10_b| + |q50_r - q50_b| + |q90_r - q90_b|",
        "temporal": True,
    },
    # Predicate-driven (needs `predicates` on the variable config)
    "right_censoring_indicator": {
        "description": "1 if the terminal event has not occurred by t0 (censored)",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "(count(*) FILTER (WHERE col = 'terminal') = 0)::int",
        "temporal": True,
    },
    "cross_type_latency": {
        "description": "Mean seconds from an A-typed event to the next B-typed event",
        "input_types": ["categorical"],
        "output_type": "numeric",
        "sql_example": "AVG(MIN(b.ts) - a.ts) for a=A-rows, b=next B-rows",
        "temporal": True,
    },
    # Spatial (needs spatial_ix {lat, lon} on the entity)
    "distance_travelled": {
        "description": "Total great-circle distance over consecutive events (m)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "SUM(haversine(lag(lat,lon), (lat,lon)))",
        "temporal": True,
    },
    "radius_of_gyration": {
        "description": "RMS great-circle distance of events from their centroid (m)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "sqrt(AVG(haversine(centroid, point)^2))",
        "temporal": True,
    },
    "spatial_std": {
        "description": "Degree-space dispersion: sqrt(var(lat) + var(lon))",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "sqrt(var_samp(lat) + var_samp(lon))",
        "temporal": True,
    },
    "bbox_area": {
        "description": "Approximate latitude-corrected bounding-box area (m^2)",
        "input_types": ["index"],
        "output_type": "numeric",
        "sql_example": "(max(lat)-min(lat))*(max(lon)-min(lon))*cos(avg(lat))*111320^2",
        "temporal": True,
    },
}

TRANSFORMATION_DOCS: Dict[str, Dict[str, Any]] = {
    # Basic transformers
    "identity": {
        "description": "Pass-through (no transformation)",
        "input_types": ["numeric", "categorical", "text"],
        "output_type": "same",
        "sql_example": "column_name",
        "category": "basic",
    },
    "abs": {
        "description": "Absolute value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "ABS(value)",
        "category": "math",
    },
    "exp": {
        "description": "Exponential (e^x)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "EXP(value)",
        "category": "math",
    },
    "ln": {
        "description": "Natural logarithm",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "LN(value)",
        "category": "math",
    },
    "log": {
        "description": "Base-10 logarithm",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "LOG(value)",
        "category": "math",
    },
    "sqrt": {
        "description": "Square root",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SQRT(value)",
        "category": "math",
    },
    "cbrt": {
        "description": "Cube root",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "CBRT(value)",
        "category": "math",
    },
    "sign": {
        "description": "Sign of value (-1, 0, or 1)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SIGN(value)",
        "category": "math",
    },
    "ceil": {
        "description": "Round up to nearest integer",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "CEIL(value)",
        "category": "math",
    },
    "floor": {
        "description": "Round down to nearest integer",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "FLOOR(value)",
        "category": "math",
    },
    "trunc": {
        "description": "Truncate decimal portion",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "TRUNC(value)",
        "category": "math",
    },
    "num_chars": {
        "description": "Character count in text",
        "input_types": ["text"],
        "output_type": "numeric",
        "sql_example": "CHAR_LENGTH(text)",
        "category": "text",
    },
    # Date transformers
    "day": {
        "description": "Day of month (1-31)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'DD')",
        "category": "date",
    },
    "dow": {
        "description": "ISO day of week (1=Monday to 7=Sunday)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'ID')",
        "category": "date",
    },
    "dom": {
        "description": "Day of month (DD format)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'DD')",
        "category": "date",
    },
    "doy": {
        "description": "Day of year (1-366)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'DDD')",
        "category": "date",
    },
    "year": {
        "description": "Four-digit year",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'YYYY')",
        "category": "date",
    },
    "month": {
        "description": "Month number (1-12)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'MM')",
        "category": "date",
    },
    "hour": {
        "description": "Hour (0-23)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'HH24')",
        "category": "date",
    },
    "quarter": {
        "description": "Quarter of year (1-4)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'Q')",
        "category": "date",
    },
    "week": {
        "description": "Week of month",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'W')",
        "category": "date",
    },
    "week_of_year": {
        "description": "Week of year (1-53)",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'WW')",
        "category": "date",
    },
    "century": {
        "description": "Century number",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'CC')",
        "category": "date",
    },
    "tz": {
        "description": "Time zone abbreviation",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'TZ')",
        "category": "date",
    },
    "tz_offset": {
        "description": "Time zone offset",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "categorical",
        "sql_example": "TO_CHAR(date, 'OF')",
        "category": "date",
    },
    # Binning transformers
    "hourly_bin": {
        "description": "Bin hours into time-of-day categories",
        "input_types": ["date", "timestamp"],
        "output_type": "categorical",
        "sql_example": "CASE WHEN hour < 5 THEN 'night' ... END",
        "category": "binning",
    },
    "daily_bin": {
        "description": "Bin days into weekday/weekend",
        "input_types": ["date", "timestamp"],
        "output_type": "categorical",
        "sql_example": "CASE WHEN dow < 6 THEN 'weekday' ELSE 'weekend' END",
        "category": "binning",
    },
    # Cyclical transformers
    "cyclic_hour": {
        "description": "Hour as sin/cos pair for cyclical encoding",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "SIN(hour * 2*PI/24), COS(hour * 2*PI/24)",
        "category": "cyclical",
    },
    "cyclic_month": {
        "description": "Month as sin/cos pair for cyclical encoding",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "SIN((month-1) * 2*PI/12), COS((month-1) * 2*PI/12)",
        "category": "cyclical",
    },
    "cyclic_day": {
        "description": "Day of week as sin/cos pair for cyclical encoding",
        "input_types": ["date", "timestamp", "index"],
        "output_type": "numeric",
        "sql_example": "SIN(dow * 2*PI/7), COS(dow * 2*PI/7)",
        "category": "cyclical",
    },
    # Cumulative window functions
    "cum_sum": {
        "description": "Cumulative sum over time",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "cumulative",
        "requires_temporal": True,
    },
    "cum_mean": {
        "description": "Cumulative mean over time",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "cumulative",
        "requires_temporal": True,
    },
    "cum_max": {
        "description": "Cumulative maximum over time",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "MAX(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "cumulative",
        "requires_temporal": True,
    },
    "cum_min": {
        "description": "Cumulative minimum over time",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "MIN(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "cumulative",
        "requires_temporal": True,
    },
    "cum_count": {
        "description": "Cumulative count over time",
        "input_types": ["categorical", "index"],
        "output_type": "numeric",
        "sql_example": "COUNT(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "cumulative",
        "requires_temporal": True,
    },
    # Value access
    "first": {
        "description": "First value in partition",
        "input_types": ["categorical", "index", "numeric", "date"],
        "output_type": "same",
        "sql_example": "FIRST_VALUE(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "window",
        "requires_temporal": True,
    },
    "last": {
        "description": "Last value in partition",
        "input_types": ["categorical", "index", "numeric"],
        "output_type": "same",
        "sql_example": "LAST_VALUE(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "window",
        "requires_temporal": True,
    },
    "previous": {
        "description": "Previous row's value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "LAG(value) OVER (PARTITION BY id ORDER BY date)",
        "category": "window",
        "requires_temporal": True,
    },
    "diff": {
        "description": "Difference from previous value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "value - LAG(value) OVER (...)",
        "category": "window",
        "requires_temporal": True,
    },
    "time_since_previous": {
        "description": "Time elapsed since previous record",
        "input_types": ["date", "timestamp"],
        "output_type": "date",
        "sql_example": "date - LAG(date) OVER (...)",
        "category": "window",
        "requires_temporal": True,
    },
    # Distribution functions
    "cdf": {
        "description": "Cumulative distribution function value",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "CUME_DIST() OVER (PARTITION BY id ORDER BY value)",
        "category": "distribution",
        "requires_temporal": True,
    },
    "percent_rank": {
        "description": "Relative rank as percentage (0-1)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENT_RANK() OVER (PARTITION BY id ORDER BY value)",
        "category": "distribution",
        "requires_temporal": True,
    },
    "ntile": {
        "description": "Divide into N equal groups (default: 5)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "NTILE(5) OVER (PARTITION BY id ORDER BY value)",
        "category": "distribution",
        "requires_temporal": True,
    },
    # Lag transformers
    "lag_1": {
        "description": "Value from 1 period ago",
        "input_types": ["numeric", "categorical", "date", "timestamp", "index"],
        "output_type": "same",
        "sql_example": "LAG(value, 1) OVER (PARTITION BY id ORDER BY date)",
        "category": "lag",
        "requires_temporal": True,
    },
    "lag_3": {
        "description": "Value from 3 periods ago",
        "input_types": ["numeric", "categorical", "date", "timestamp", "index"],
        "output_type": "same",
        "sql_example": "LAG(value, 3) OVER (PARTITION BY id ORDER BY date)",
        "category": "lag",
        "requires_temporal": True,
    },
    "lag_7": {
        "description": "Value from 7 periods ago",
        "input_types": ["numeric", "categorical", "date", "timestamp", "index"],
        "output_type": "same",
        "sql_example": "LAG(value, 7) OVER (PARTITION BY id ORDER BY date)",
        "category": "lag",
        "requires_temporal": True,
    },
    # Rolling statistics
    "rolling_mean_3": {
        "description": "3-period rolling mean",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_mean_7": {
        "description": "7-period rolling mean",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_mean_14": {
        "description": "14-period rolling mean",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_std_3": {
        "description": "3-period rolling standard deviation",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "STDDEV(value) OVER (... ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_std_7": {
        "description": "7-period rolling standard deviation",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "STDDEV(value) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_std_14": {
        "description": "14-period rolling standard deviation",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "STDDEV(value) OVER (... ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_median_5": {
        "description": "5-period rolling median",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) OVER (...)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_median_7": {
        "description": "7-period rolling median",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) OVER (...)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_iqr_7": {
        "description": "7-period rolling interquartile range (P75 - P25)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.75) - PERCENTILE_CONT(0.25) OVER (...)",
        "category": "rolling",
        "requires_temporal": True,
    },
    "rolling_iqr_14": {
        "description": "14-period rolling interquartile range",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENTILE_CONT(0.75) - PERCENTILE_CONT(0.25) OVER (...)",
        "category": "rolling",
        "requires_temporal": True,
    },
    # Exponential moving average
    "ema_7": {
        "description": "7-period exponential moving average",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(value * EXP(decay * t)) / SUM(EXP(decay * t)) OVER (...)",
        "category": "ema",
        "requires_temporal": True,
    },
    "ema_14": {
        "description": "14-period exponential moving average",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(value * EXP(decay * t)) / SUM(EXP(decay * t)) OVER (...)",
        "category": "ema",
        "requires_temporal": True,
    },
    # Holt-Winters
    "holt_winters_level_7": {
        "description": "7-period Holt-Winters level (smoothed average)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)",
        "category": "holt_winters",
        "requires_temporal": True,
    },
    "holt_winters_level_14": {
        "description": "14-period Holt-Winters level",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)",
        "category": "holt_winters",
        "requires_temporal": True,
    },
    "holt_winters_trend_7": {
        "description": "7-period Holt-Winters trend (slope)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "REGR_SLOPE(value, time) OVER (...)",
        "category": "holt_winters",
        "requires_temporal": True,
    },
    "holt_winters_trend_14": {
        "description": "14-period Holt-Winters trend",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "REGR_SLOPE(value, time) OVER (...)",
        "category": "holt_winters",
        "requires_temporal": True,
    },
    # Percentage change
    "pct_change_1": {
        "description": "Percentage change from 1 period ago",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "(value - LAG(value, 1)) / LAG(value, 1)",
        "category": "pct_change",
        "requires_temporal": True,
    },
    "pct_change_3": {
        "description": "Percentage change from 3 periods ago",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "(value - LAG(value, 3)) / LAG(value, 3)",
        "category": "pct_change",
        "requires_temporal": True,
    },
    # Boolean checks
    "is_null": {
        "description": "Check if value is null",
        "input_types": ["numeric", "categorical", "date"],
        "output_type": "boolean",
        "sql_example": "(value IS NULL)",
        "category": "boolean",
    },
    "in_array": {
        "description": "Check if value is in array",
        "input_types": ["numeric", "categorical", "date"],
        "output_type": "boolean",
        "sql_example": "value = ANY(ARRAY[...])",
        "category": "boolean",
    },
    # Population window transformers
    "cross_entity_zscore": {
        "description": "Z-score normalized across all entities in the population",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "(value - AVG(value) OVER ()) / NULLIF(STDDEV(value) OVER (), 0)",
        "category": "population_window",
        "requires_temporal": False,
    },
    "cross_entity_percentile": {
        "description": "Percentile rank across all entities in the population",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "PERCENT_RANK() OVER (ORDER BY value)",
        "category": "population_window",
        "requires_temporal": False,
    },
    # Change-point detection
    "mean_shift_ratio_7": {
        "description": "Ratio of recent 7-period mean to overall mean (change-point detection)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS 6 PRECEDING) / NULLIF(AVG(value) OVER (), 0)",
        "category": "change_point",
        "requires_temporal": True,
    },
    "mean_shift_ratio_14": {
        "description": "Ratio of recent 14-period mean to overall mean (change-point detection)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "AVG(value) OVER (... ROWS 13 PRECEDING) / NULLIF(AVG(value) OVER (), 0)",
        "category": "change_point",
        "requires_temporal": True,
    },
    "cusum": {
        "description": "CUSUM: cumulative sum of deviations from target mean",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "SUM(value - target_mean) OVER (PARTITION BY id ORDER BY date)",
        "category": "change_point",
        "requires_temporal": True,
    },
    # Higher-order differences and running product
    "diff2": {
        "description": "Second difference (acceleration): x - 2*lag1 + lag2",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "value - 2*LAG(value,1) OVER w + LAG(value,2) OVER w",
        "category": "window",
        "requires_temporal": True,
    },
    "diff3": {
        "description": "Third difference (jerk): x - 3*lag1 + 3*lag2 - lag3",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "value - 3*LAG(value,1) + 3*LAG(value,2) - LAG(value,3) OVER w",
        "category": "window",
        "requires_temporal": True,
    },
    "cumprod": {
        "description": "Running product via log-sum-exp (positive series only)",
        "input_types": ["numeric"],
        "output_type": "numeric",
        "sql_example": "exp(sum(ln(value)) OVER (PARTITION BY id ORDER BY date))",
        "category": "cumulative",
        "requires_temporal": True,
    },
}


def list_primitives_command(args: argparse.Namespace) -> int:
    """List available primitives."""
    # Ensure primitives are loaded
    from . import primitives  # noqa: F401

    show_all = args.type == "all"
    show_agg = show_all or args.type == "agg"
    show_transform = show_all or args.type == "transform"

    if show_agg:
        aggs = list(list_aggregations())
        print(f"\n{'=' * 60}")
        print(f"AGGREGATION PRIMITIVES ({len(aggs)} available)")
        print(f"{'=' * 60}")

        if args.category:
            # Group by whether they support temporal intervals
            temporal = [
                a for a in aggs if AGGREGATION_DOCS.get(a, {}).get("temporal", False)
            ]
            non_temporal = [
                a
                for a in aggs
                if not AGGREGATION_DOCS.get(a, {}).get("temporal", False)
            ]

            if temporal:
                print("\n  Temporal (support interval windows):")
                for name in sorted(temporal):
                    _print_primitive(
                        name, AGGREGATION_DOCS.get(name, {}), args.show_sql
                    )
            if non_temporal:
                print("\n  Non-temporal:")
                for name in sorted(non_temporal):
                    _print_primitive(
                        name, AGGREGATION_DOCS.get(name, {}), args.show_sql
                    )
        else:
            for name in aggs:
                _print_primitive(name, AGGREGATION_DOCS.get(name, {}), args.show_sql)

    if show_transform:
        transforms = list(list_transformations())
        print(f"\n{'=' * 60}")
        print(f"TRANSFORMATION PRIMITIVES ({len(transforms)} available)")
        print(f"{'=' * 60}")

        if args.category:
            # Group by category
            categories: Dict[str, List[str]] = {}
            for name in transforms:
                cat = TRANSFORMATION_DOCS.get(name, {}).get("category", "other")
                categories.setdefault(cat, []).append(name)

            category_order = [
                "basic",
                "math",
                "text",
                "date",
                "binning",
                "cyclical",
                "cumulative",
                "window",
                "distribution",
                "lag",
                "rolling",
                "ema",
                "holt_winters",
                "pct_change",
                "boolean",
                "population_window",
                "change_point",
                "other",
            ]
            for cat in category_order:
                if cat in categories:
                    print(f"\n  {cat.upper().replace('_', ' ')}:")
                    for name in sorted(categories[cat]):
                        _print_primitive(
                            name, TRANSFORMATION_DOCS.get(name, {}), args.show_sql
                        )
        else:
            for name in transforms:
                _print_primitive(name, TRANSFORMATION_DOCS.get(name, {}), args.show_sql)

    print()
    return 0


def _print_primitive(name: str, doc: Dict[str, Any], show_sql: bool) -> None:
    """Print a single primitive's information."""
    desc = doc.get("description", "No description available")
    inputs = doc.get("input_types", ["unknown"])
    requires_temporal = doc.get("requires_temporal", False)

    temporal_marker = " [T]" if requires_temporal else ""
    input_str = ", ".join(inputs)

    print(f"    {name:<25} {desc}{temporal_marker}")
    print(f"      {'':25} inputs: {input_str}")

    if show_sql:
        sql = doc.get("sql_example", "N/A")
        print(f"      {'':25} SQL: {sql}")


def validate_command(args: argparse.Namespace) -> int:
    """Validate a configuration file."""
    try:
        result = validate_config(args.config)

        if result.is_valid:
            print(f"Configuration is valid: {args.config}")
            if result.warnings:
                print(f"\nWarnings ({len(result.warnings)}):")
                for w in result.warnings:
                    loc = f"[{w.location}] " if w.location else ""
                    print(f"  {loc}{w.message}")
            return 0
        else:
            print(result.format_errors())
            return 1
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="featurizer",
        description="Featurizer - Deep Feature Synthesis for PostgreSQL",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # list-primitives command
    list_parser = subparsers.add_parser(
        "list-primitives",
        aliases=["lp"],
        help="List available aggregation and transformation primitives",
    )
    list_parser.add_argument(
        "--type",
        "-t",
        choices=["agg", "transform", "all"],
        default="all",
        help="Type of primitives to list (default: all)",
    )
    list_parser.add_argument(
        "--show-sql",
        "-s",
        action="store_true",
        help="Show example SQL for each primitive",
    )
    list_parser.add_argument(
        "--category",
        "-c",
        action="store_true",
        help="Group primitives by category",
    )
    list_parser.set_defaults(func=list_primitives_command)

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a configuration file",
    )
    validate_parser.add_argument(
        "config",
        help="Path to YAML configuration file",
    )
    validate_parser.set_defaults(func=validate_command)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
