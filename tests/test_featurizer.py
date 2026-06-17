"""Featurizer integration tests covering planner hashability.

The sample config mirrors the production schema shape while remaining self-contained.
"""

from featurizer import Featurizer


def test_featurizer_features_remain_hashable(sample_config_path):
    featurizer = Featurizer(sample_config_path)

    target_alias = featurizer.target.alias
    features = featurizer.features[target_alias]

    # Ensure numeric aggregations landed in the feature set.
    feature_names = {feature.name for feature in features}
    assert any("MEAN(" in name for name in feature_names)
    assert any("SUM(" in name for name in feature_names)
    assert any("STDDEV(" in name for name in feature_names)
    assert any("ABS(" in name for name in feature_names)
    assert any("CUM_SUM(" in name for name in feature_names)
    assert any("ROLLING_MEDIAN_7(" in name for name in feature_names)
    assert any("ROLLING_IQR_7(" in name for name in feature_names)
    assert any("EMA_7(" in name for name in feature_names)
    assert any("HOLT_WINTERS_TREND_7(" in name for name in feature_names)

    # Hashability regression guard: re-adding the set should not change cardinality.
    feature_set = set(features)
    feature_set.update(features)
    assert len(feature_set) == len(features)

    # This config is wide enough that the single `<target>_transform` CTE tuple
    # exceeds PostgreSQL's 1664-entry limit, so `.query` refuses (pointing at the
    # sharded API) rather than emitting SQL Postgres would reject (issue #7).
    import pytest

    with pytest.raises(ValueError, match="too wide"):
        _ = featurizer.query

    # The sharded queries still carry the lateral join + as-of CTE; assert
    # against the union of all column groups.
    all_group_sql = "\n".join(featurizer.query_groups.values()).lower()
    assert "lateral (" in all_group_sql
    assert "care_plans_asof_for_patients" in all_group_sql
