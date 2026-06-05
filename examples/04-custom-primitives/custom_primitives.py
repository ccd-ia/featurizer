"""Custom aggregation and transformation primitives for financial analytics."""

from featurizer.primitives.abstractions import Feature
from featurizer.primitives.aggregations import Aggregator
from featurizer.primitives.transformations import Transformer
from featurizer.primitives.utils import register_aggregation, register_transformer

# ============================================================================
# Custom Aggregations
# ============================================================================


class Median(Aggregator):
    """Calculate median value."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for median calculation.

        Note: This uses percentile_cont which is PostgreSQL-specific.
        For SQLite, you would need a different approach.
        """
        return f"percentile_cont(0.5) WITHIN GROUP (ORDER BY {feature.name}) AS {alias}"


class Percentile95(Aggregator):
    """Calculate 95th percentile."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for 95th percentile calculation."""
        return (
            f"percentile_cont(0.95) WITHIN GROUP (ORDER BY {feature.name}) AS {alias}"
        )


class Range(Aggregator):
    """Calculate range (max - min)."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for range calculation."""
        return f"(MAX({feature.name}) - MIN({feature.name})) AS {alias}"


# ============================================================================
# Custom Transformations
# ============================================================================


class Log(Transformer):
    """Natural logarithm transformation."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for log transformation."""
        # Add small constant to avoid log(0)
        return f"LN({feature.name} + 1) AS {alias}"


class ZScore(Transformer):
    """Z-score standardization transformation."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for z-score transformation.

        Note: This is a window function that requires appropriate partitioning.
        For simplicity, this calculates z-score across all rows.
        """
        return f"""
            (({feature.name} - AVG({feature.name}) OVER ())
             / NULLIF(STDDEV({feature.name}) OVER (), 0)) AS {alias}
        """.strip()


class BinCount(Transformer):
    """Discretize continuous values into 5 bins."""

    def to_sql(self, feature: Feature, alias: str) -> str:
        """Generate SQL for binning into 5 equal-width bins."""
        return f"""
            CASE
                WHEN {feature.name} IS NULL THEN NULL
                ELSE CAST(
                    FLOOR(
                        5 * ({feature.name} - MIN({feature.name}) OVER ())
                        / NULLIF(MAX({feature.name}) OVER () - MIN({feature.name}) OVER (), 0)
                    ) AS INTEGER
                )
            END AS {alias}
        """.strip()


# ============================================================================
# Registration
# ============================================================================


def register_all_custom_primitives():
    """Register all custom primitives with the feature system."""
    # Register aggregations
    register_aggregation("median", Median)
    register_aggregation("p95", Percentile95)
    register_aggregation("range", Range)

    # Register transformations
    register_transformer("log", Log)
    register_transformer("zscore", ZScore)
    register_transformer("bin", BinCount)

    print("✓ Registered custom primitives:")
    print("  Aggregations: median, p95, range")
    print("  Transformations: log, zscore, bin")
