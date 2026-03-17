"""Tests for new aggregation and transformer primitives."""

import pytest

from featurizer.primitives.abstractions import Entity, Feature, Key, Relationship
from featurizer.primitives.utils import (
    get_aggregations,
    get_transformers,
    list_aggregations,
    list_transformations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_parent_entity():
    return Entity(alias="customers", table="analytics.customers", id="customer_id")


def _make_child_entity():
    return Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        temporal_ix="ordered_at",
        variables={
            "amount": {"type": "numeric"},
            "category": {"type": "categorical"},
        },
    )


def _make_child_entity_no_temporal():
    """Child entity without temporal_ix."""
    return Entity(
        alias="orders",
        table="analytics.orders",
        id="order_id",
        variables={
            "amount": {"type": "numeric"},
            "category": {"type": "categorical"},
        },
    )


def _make_relationship(parent, child):
    return Relationship(
        parent=parent,
        child=child,
        parent_key="customer_id",
        child_key="customer_id",
    )


def _get_feature(entity, name):
    return next(ft for ft in entity.features if ft.name == name)


# =========================================================================
# 1. Registry tests
# =========================================================================


class TestRegistryPresence:
    """All new primitives must be discoverable via the registry."""

    NEW_AGGREGATION_NAMES = {
        "p10",
        "p25",
        "p75",
        "p90",
        "p95",
        "p99",
        "iqr",
        "cv",
        "range",
        "event_rate",
        "time_span",
        "gap_mean",
        "gap_stddev",
        "gap_min",
        "gap_max",
        "gap_cv",
        "burstiness",
        "entropy",
        "hhi",
        "gini",
        "ngram_2_freq",
        "ngram_3_freq",
        "sequence_entropy",
        "longest_streak",
    }

    NEW_TRANSFORMER_NAMES = {
        "cross_entity_zscore",
        "cross_entity_percentile",
        "cusum",
        "mean_shift_ratio_7",
        "mean_shift_ratio_14",
    }

    def test_new_aggregations_registered(self):
        registered = set(list_aggregations())
        assert (
            self.NEW_AGGREGATION_NAMES <= registered
        ), f"Missing: {self.NEW_AGGREGATION_NAMES - registered}"

    def test_new_transformers_registered(self):
        registered = set(list_transformations())
        assert (
            self.NEW_TRANSFORMER_NAMES <= registered
        ), f"Missing: {self.NEW_TRANSFORMER_NAMES - registered}"

    def test_new_aggregations_in_DEFAULT_AGGREGATIONS(self):
        from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS

        assert self.NEW_AGGREGATION_NAMES <= set(DEFAULT_AGGREGATIONS.keys())

    def test_new_transformers_in_DEFAULT_TRANSFORMERS(self):
        from featurizer.primitives.transformations import DEFAULT_TRANSFORMERS

        assert self.NEW_TRANSFORMER_NAMES <= set(DEFAULT_TRANSFORMERS.keys())

    def test_get_aggregations_returns_all_new(self):
        aggs = get_aggregations(list(self.NEW_AGGREGATION_NAMES))
        assert set(aggs.keys()) == self.NEW_AGGREGATION_NAMES

    def test_get_transformers_returns_all_new(self):
        txs = get_transformers(list(self.NEW_TRANSFORMER_NAMES))
        assert set(txs.keys()) == self.NEW_TRANSFORMER_NAMES


# =========================================================================
# 2. Tier-1 aggregation tests: percentiles, IQR, CV, Range
# =========================================================================


class TestPercentileAggregations:
    """Ordered-set percentile aggregations (p25 as representative)."""

    def test_p25_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_p25_returns_none_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "category")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature)
        assert result is None

    def test_p25_result_is_new_instance(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature)
        assert result is not feature

    def test_p25_sql_contains_percentile_cont(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature)
        assert "percentile_cont(0.25)" in result.definition

    def test_p25_with_interval_produces_filter(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature, interval="P1W")
        assert "filter" in result.definition.lower()
        assert "P1W" in result.definition

    def test_p25_interval_name_annotation(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["p25"])["p25"]
        result = agg(parent, child, feature, interval="P1W")
        assert "interval=P1W" in result.name

    @pytest.mark.parametrize(
        "name,frac",
        [
            ("p10", "0.1"),
            ("p75", "0.75"),
            ("p90", "0.9"),
            ("p95", "0.95"),
            ("p99", "0.99"),
        ],
    )
    def test_other_percentiles_sql(self, name, frac):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations([name])[name]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)
        assert f"percentile_cont({frac})" in result.definition


class TestIQR:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["iqr"])["iqr"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_returns_none_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "category")
        agg = get_aggregations(["iqr"])["iqr"]
        result = agg(parent, child, feature)
        assert result is None

    def test_result_is_new_instance(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["iqr"])["iqr"]
        result = agg(parent, child, feature)
        assert result is not feature

    def test_sql_contains_both_percentiles(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["iqr"])["iqr"]
        result = agg(parent, child, feature)
        assert "percentile_cont(0.75)" in result.definition
        assert "percentile_cont(0.25)" in result.definition

    def test_with_interval_produces_filter(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["iqr"])["iqr"]
        result = agg(parent, child, feature, interval="P1W")
        assert "filter" in result.definition.lower()
        assert "P1W" in result.definition


class TestCoefficientOfVariation:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["cv"])["cv"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_returns_none_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "category")
        agg = get_aggregations(["cv"])["cv"]
        result = agg(parent, child, feature)
        assert result is None

    def test_result_is_new_instance(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["cv"])["cv"]
        result = agg(parent, child, feature)
        assert result is not feature

    def test_sql_contains_stddev_and_avg(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["cv"])["cv"]
        result = agg(parent, child, feature)
        assert "stddev(" in result.definition
        assert "avg(" in result.definition
        assert "NULLIF" in result.definition

    def test_with_interval_produces_filter(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["cv"])["cv"]
        result = agg(parent, child, feature, interval="P1W")
        assert "filter" in result.definition.lower()


class TestRangeAgg:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["range"])["range"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_returns_none_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "category")
        agg = get_aggregations(["range"])["range"]
        result = agg(parent, child, feature)
        assert result is None

    def test_result_is_new_instance(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["range"])["range"]
        result = agg(parent, child, feature)
        assert result is not feature

    def test_sql_contains_max_minus_min(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["range"])["range"]
        result = agg(parent, child, feature)
        assert "max(" in result.definition
        assert "min(" in result.definition

    def test_with_interval_produces_filter(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["range"])["range"]
        result = agg(parent, child, feature, interval="P1W")
        assert "filter" in result.definition.lower()


# =========================================================================
# 3. EventRate and TimeSpan tests
# =========================================================================


class TestEventRate:
    def test_returns_feature_for_temporal_ix(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix  # ordered_at is the temporal_ix
        agg = get_aggregations(["event_rate"])["event_rate"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_returns_none_for_numeric_feature(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["event_rate"])["event_rate"]
        result = agg(parent, child, feature)
        assert result is None

    def test_returns_none_for_non_temporal_index(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.id  # order_id is an index but not temporal_ix
        agg = get_aggregations(["event_rate"])["event_rate"]
        result = agg(parent, child, feature)
        assert result is None

    def test_returns_none_when_entity_lacks_temporal_ix(self):
        parent = _make_parent_entity()
        child = _make_child_entity_no_temporal()
        feature = child.id
        agg = get_aggregations(["event_rate"])["event_rate"]
        result = agg(parent, child, feature)
        assert result is None

    def test_sql_contains_extract_epoch(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix
        agg = get_aggregations(["event_rate"])["event_rate"]
        result = agg(parent, child, feature)
        assert "EXTRACT(EPOCH FROM" in result.definition


class TestTimeSpan:
    def test_returns_feature_for_temporal_ix(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix
        agg = get_aggregations(["time_span"])["time_span"]
        result = agg(parent, child, feature)
        assert isinstance(result, Feature)

    def test_returns_none_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["time_span"])["time_span"]
        result = agg(parent, child, feature)
        assert result is None

    def test_returns_none_for_non_temporal_index(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.id
        agg = get_aggregations(["time_span"])["time_span"]
        result = agg(parent, child, feature)
        assert result is None

    def test_returns_none_when_entity_lacks_temporal_ix(self):
        parent = _make_parent_entity()
        child = _make_child_entity_no_temporal()
        feature = child.id
        agg = get_aggregations(["time_span"])["time_span"]
        result = agg(parent, child, feature)
        assert result is None

    def test_sql_contains_extract_epoch(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix
        agg = get_aggregations(["time_span"])["time_span"]
        result = agg(parent, child, feature)
        assert "EXTRACT(EPOCH FROM" in result.definition


# =========================================================================
# 4. SubqueryAggregator tests: gap stats, burstiness
# =========================================================================


class TestGapMean:
    def test_returns_none_when_relationship_is_none(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=None)
        assert result is None

    def test_returns_none_when_feature_is_not_temporal_ix(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_returns_feature_with_proper_inputs(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_definition_contains_child_table_alias(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=rel)
        assert "orders_transform" in result.definition

    def test_definition_contains_child_key(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=rel)
        assert "customer_id" in result.definition

    def test_definition_contains_lag_and_avg(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, relationship=rel)
        assert "LAG" in result.definition
        assert "AVG" in result.definition

    def test_with_interval_definition_contains_daterange(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_mean"])["gap_mean"]
        result = agg(parent, child, feature, interval="P1W", relationship=rel)
        assert isinstance(result, Feature)
        assert "daterange" in result.definition
        assert "P1W" in result.definition


class TestGapStddev:
    def test_returns_feature_with_proper_inputs(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_stddev"])["gap_stddev"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)
        assert "STDDEV" in result.definition


class TestGapMinMax:
    @pytest.mark.parametrize("name,sql_func", [("gap_min", "MIN"), ("gap_max", "MAX")])
    def test_returns_feature_with_correct_aggregate(self, name, sql_func):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations([name])[name]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)
        assert sql_func in result.definition


class TestGapCV:
    def test_returns_feature_with_proper_inputs(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_cv"])["gap_cv"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_definition_contains_stddev_and_avg(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["gap_cv"])["gap_cv"]
        result = agg(parent, child, feature, relationship=rel)
        assert "STDDEV" in result.definition
        assert "AVG" in result.definition

    def test_only_fires_on_temporal_ix(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gap_cv"])["gap_cv"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None


class TestBurstiness:
    def test_returns_feature_with_stddev_avg_formula(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = child.temporal_ix
        agg = get_aggregations(["burstiness"])["burstiness"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)
        assert "STDDEV" in result.definition
        assert "AVG" in result.definition

    def test_only_fires_on_temporal_ix(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["burstiness"])["burstiness"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_returns_none_without_relationship(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = child.temporal_ix
        agg = get_aggregations(["burstiness"])["burstiness"]
        result = agg(parent, child, feature, relationship=None)
        assert result is None


# =========================================================================
# 5. Entropy and HHI tests
# =========================================================================


class TestEntropy:
    def test_returns_feature_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_returns_none_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_definition_contains_group_by(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert "GROUP BY sub.category" in result.definition

    def test_definition_contains_ln(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert "LN" in result.definition

    def test_with_relationship_references_child_table(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert "orders_transform" in result.definition

    def test_returns_none_without_relationship(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "category")
        agg = get_aggregations(["entropy"])["entropy"]
        result = agg(parent, child, feature, relationship=None)
        assert result is None


class TestHHI:
    def test_returns_feature_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["hhi"])["hhi"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_returns_none_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["hhi"])["hhi"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_definition_contains_group_by(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["hhi"])["hhi"]
        result = agg(parent, child, feature, relationship=rel)
        assert "GROUP BY sub.category" in result.definition

    def test_definition_contains_power(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["hhi"])["hhi"]
        result = agg(parent, child, feature, relationship=rel)
        assert "POWER" in result.definition

    def test_with_relationship_references_child_table(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["hhi"])["hhi"]
        result = agg(parent, child, feature, relationship=rel)
        assert "orders_transform" in result.definition


# =========================================================================
# 6. Gini test
# =========================================================================


class TestGini:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gini"])["gini"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_returns_none_for_categorical(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["gini"])["gini"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_definition_contains_row_number(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gini"])["gini"]
        result = agg(parent, child, feature, relationship=rel)
        assert "ROW_NUMBER" in result.definition

    def test_definition_contains_sum_rn_val(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gini"])["gini"]
        result = agg(parent, child, feature, relationship=rel)
        assert "SUM(rn * val)" in result.definition

    def test_returns_none_without_relationship(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["gini"])["gini"]
        result = agg(parent, child, feature, relationship=None)
        assert result is None


# =========================================================================
# 7. Sequence feature tests: ngram, sequence_entropy, longest_streak
# =========================================================================


class TestNgramFrequency:
    def test_returns_none_when_temporal_ix_missing(self):
        parent = _make_parent_entity()
        child = _make_child_entity_no_temporal()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["ngram_2_freq"])["ngram_2_freq"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_returns_feature_for_categorical_with_temporal_and_rel(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["ngram_2_freq"])["ngram_2_freq"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_ngram_2_definition_contains_lag(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["ngram_2_freq"])["ngram_2_freq"]
        result = agg(parent, child, feature, relationship=rel)
        assert "LAG" in result.definition

    def test_ngram_3_returns_feature(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["ngram_3_freq"])["ngram_3_freq"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_returns_none_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["ngram_2_freq"])["ngram_2_freq"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None


class TestSequenceEntropy:
    def test_returns_none_when_temporal_ix_missing(self):
        parent = _make_parent_entity()
        child = _make_child_entity_no_temporal()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["sequence_entropy"])["sequence_entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_returns_feature_for_categorical_with_temporal_and_rel(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["sequence_entropy"])["sequence_entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_definition_contains_transitions(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["sequence_entropy"])["sequence_entropy"]
        result = agg(parent, child, feature, relationship=rel)
        assert "transitions" in result.definition


class TestLongestStreak:
    def test_returns_none_when_temporal_ix_missing(self):
        parent = _make_parent_entity()
        child = _make_child_entity_no_temporal()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["longest_streak"])["longest_streak"]
        result = agg(parent, child, feature, relationship=rel)
        assert result is None

    def test_returns_feature_for_categorical_with_temporal_and_rel(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["longest_streak"])["longest_streak"]
        result = agg(parent, child, feature, relationship=rel)
        assert isinstance(result, Feature)

    def test_definition_contains_streak(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        rel = _make_relationship(parent, child)
        feature = _get_feature(child, "category")
        agg = get_aggregations(["longest_streak"])["longest_streak"]
        result = agg(parent, child, feature, relationship=rel)
        assert "streak" in result.definition.lower()


# =========================================================================
# 8. Transformer tests
# =========================================================================


class TestCrossEntityZscore:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cross_entity_zscore"])["cross_entity_zscore"]
        result = tx(child, feature)
        assert isinstance(result, Feature)
        assert result is not feature

    def test_returns_feature_unchanged_for_key(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        # Manually add a key (normally ERGraph does this)
        key_feature = Key(name="customer_id", entity=child)
        child.add_key(key_feature)
        tx = get_transformers(["cross_entity_zscore"])["cross_entity_zscore"]
        result = tx(child, key_feature)
        assert result is key_feature

    def test_definition_contains_avg_stddev_over(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cross_entity_zscore"])["cross_entity_zscore"]
        result = tx(child, feature)
        assert "AVG" in result.definition
        assert "STDDEV" in result.definition
        assert "OVER ()" in result.definition


class TestCrossEntityPercentile:
    def test_returns_feature_for_numeric(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cross_entity_percentile"])["cross_entity_percentile"]
        result = tx(child, feature)
        assert isinstance(result, Feature)
        assert result is not feature

    def test_definition_contains_percent_rank(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cross_entity_percentile"])["cross_entity_percentile"]
        result = tx(child, feature)
        assert "PERCENT_RANK" in result.definition


class TestMeanShiftRatio:
    def test_returns_feature_for_numeric_with_temporal(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["mean_shift_ratio_7"])["mean_shift_ratio_7"]
        result = tx(child, feature)
        assert isinstance(result, Feature)
        assert result is not feature

    def test_returns_none_without_temporal_ix(self):
        child = _make_child_entity_no_temporal()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["mean_shift_ratio_7"])["mean_shift_ratio_7"]
        result = tx(child, feature)
        assert result is None

    def test_definition_contains_two_avg_windows(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["mean_shift_ratio_7"])["mean_shift_ratio_7"]
        result = tx(child, feature)
        # Should contain two AVG OVER windows (recent and prior)
        assert result.definition.count("AVG(") >= 2

    def test_mean_shift_ratio_14_also_works(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["mean_shift_ratio_14"])["mean_shift_ratio_14"]
        result = tx(child, feature)
        assert isinstance(result, Feature)


class TestCusum:
    def test_returns_feature_for_numeric_with_temporal(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cusum"])["cusum"]
        result = tx(child, feature)
        assert isinstance(result, Feature)
        assert result is not feature

    def test_returns_none_without_temporal_ix(self):
        child = _make_child_entity_no_temporal()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cusum"])["cusum"]
        result = tx(child, feature)
        assert result is None

    def test_definition_contains_sum_row_number_avg(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        tx = get_transformers(["cusum"])["cusum"]
        result = tx(child, feature)
        assert "SUM(" in result.definition
        assert "ROW_NUMBER()" in result.definition
        assert "AVG(" in result.definition


# =========================================================================
# 9. Interval bug-fix regression tests
# =========================================================================


class TestIntervalBugFixes:
    """Aggregators that previously crashed on interval parameter now work."""

    def test_zscore_with_interval_no_crash(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["z_score"])["z_score"]
        # This used to raise TypeError because Zscore._build_aggregate_expression
        # did not accept interval in older code. Should not raise now.
        result = agg(parent, child, feature, interval="P1W")
        assert isinstance(result, Feature)

    def test_median_with_interval_produces_filter(self):
        parent, child = _make_parent_entity(), _make_child_entity()
        feature = _get_feature(child, "amount")
        agg = get_aggregations(["median"])["median"]
        result = agg(parent, child, feature, interval="P1M")
        assert isinstance(result, Feature)
        assert "filter" in result.definition.lower()
        assert "P1M" in result.definition
