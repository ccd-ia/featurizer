from featurizer import Featurizer

from tests.utils.sql import load_snapshot, normalize_sql


def test_sample_query_snapshot(sample_config_path):
    featurizer = Featurizer(sample_config_path)
    actual = normalize_sql(featurizer.query)
    expected = normalize_sql(load_snapshot("sample_featurizer.sql"))
    assert actual == expected
