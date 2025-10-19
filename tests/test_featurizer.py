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

    # Hashability regression guard: re-adding the set should not change cardinality.
    feature_set = set(features)
    feature_set.update(features)
    assert len(feature_set) == len(features)
