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

from .primitives.utils import list_aggregations, list_transformations, _AGGREGATIONS, _TRANSFORMATIONS
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
        print(f"\n{'='*60}")
        print(f"AGGREGATION PRIMITIVES ({len(aggs)} available)")
        print(f"{'='*60}")

        if args.category:
            # Group by whether they support temporal intervals
            temporal = [a for a in aggs if AGGREGATION_DOCS.get(a, {}).get("temporal", False)]
            non_temporal = [a for a in aggs if not AGGREGATION_DOCS.get(a, {}).get("temporal", False)]

            if temporal:
                print(f"\n  Temporal (support interval windows):")
                for name in sorted(temporal):
                    _print_primitive(name, AGGREGATION_DOCS.get(name, {}), args.show_sql)
            if non_temporal:
                print(f"\n  Non-temporal:")
                for name in sorted(non_temporal):
                    _print_primitive(name, AGGREGATION_DOCS.get(name, {}), args.show_sql)
        else:
            for name in aggs:
                _print_primitive(name, AGGREGATION_DOCS.get(name, {}), args.show_sql)

    if show_transform:
        transforms = list(list_transformations())
        print(f"\n{'='*60}")
        print(f"TRANSFORMATION PRIMITIVES ({len(transforms)} available)")
        print(f"{'='*60}")

        if args.category:
            # Group by category
            categories: Dict[str, List[str]] = {}
            for name in transforms:
                cat = TRANSFORMATION_DOCS.get(name, {}).get("category", "other")
                categories.setdefault(cat, []).append(name)

            category_order = [
                "basic", "math", "text", "date", "binning", "cyclical",
                "cumulative", "window", "distribution", "lag", "rolling",
                "ema", "holt_winters", "pct_change", "boolean", "other"
            ]
            for cat in category_order:
                if cat in categories:
                    print(f"\n  {cat.upper().replace('_', ' ')}:")
                    for name in sorted(categories[cat]):
                        _print_primitive(name, TRANSFORMATION_DOCS.get(name, {}), args.show_sql)
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
        "--type", "-t",
        choices=["agg", "transform", "all"],
        default="all",
        help="Type of primitives to list (default: all)",
    )
    list_parser.add_argument(
        "--show-sql", "-s",
        action="store_true",
        help="Show example SQL for each primitive",
    )
    list_parser.add_argument(
        "--category", "-c",
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
