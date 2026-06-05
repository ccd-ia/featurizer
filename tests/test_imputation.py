"""Tests for the opt-in imputation helper (featurizer.imputation)."""

from __future__ import annotations

import pandas as pd

from featurizer import impute_features
from featurizer.imputation import DEFAULT_COUNT_LIKE_PREFIXES, _is_count_like


def _matrix() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "COUNT(orders.order_id|interval=P1W)": [1.0, None, 3.0],
            "SUM(orders.amount)": [10.0, None, 30.0],
            "AVG(orders.amount)": [5.0, None, 15.0],
            "RECENCY(orders.ts)": [2.0, None, 4.0],
        }
    )


def test_count_like_detection():
    assert _is_count_like("COUNT(x)", DEFAULT_COUNT_LIKE_PREFIXES)
    assert _is_count_like("SUM(x)", DEFAULT_COUNT_LIKE_PREFIXES)
    assert _is_count_like("N_FAILED(x)", DEFAULT_COUNT_LIKE_PREFIXES)
    # measures are not count-like
    assert not _is_count_like("AVG(x)", DEFAULT_COUNT_LIKE_PREFIXES)
    assert not _is_count_like("RECENCY(x)", DEFAULT_COUNT_LIKE_PREFIXES)
    # CUM_SUM must not be mistaken for SUM
    assert not _is_count_like("CUM_SUM(x)", DEFAULT_COUNT_LIKE_PREFIXES)


def test_count_filled_zero_measures_stay_null():
    out = impute_features(_matrix())
    # count-like → 0
    assert out["COUNT(orders.order_id|interval=P1W)"].iloc[1] == 0
    assert out["SUM(orders.amount)"].iloc[1] == 0
    # measures left NULL under the default strategy
    assert pd.isna(out["AVG(orders.amount)"].iloc[1])
    assert pd.isna(out["RECENCY(orders.ts)"].iloc[1])


def test_missing_indicators_emitted_before_fill():
    out = impute_features(_matrix())
    for col in (
        "COUNT(orders.order_id|interval=P1W)",
        "SUM(orders.amount)",
        "AVG(orders.amount)",
        "RECENCY(orders.ts)",
    ):
        ind = f"{col}__missing"
        assert ind in out.columns
        assert out[ind].tolist() == [0, 1, 0]


def test_measure_strategy_median():
    out = impute_features(_matrix(), measure_strategy="median")
    # median of [5, 15] = 10 ; [2, 4] = 3
    assert out["AVG(orders.amount)"].iloc[1] == 10.0
    assert out["RECENCY(orders.ts)"].iloc[1] == 3.0


def test_count_fill_zero_can_be_disabled():
    out = impute_features(
        _matrix(), count_fill_zero=False, add_missing_indicators=False
    )
    assert pd.isna(out["COUNT(orders.order_id|interval=P1W)"].iloc[1])


def test_input_not_mutated():
    df = _matrix()
    impute_features(df)
    assert pd.isna(df["COUNT(orders.order_id|interval=P1W)"].iloc[1])
    assert "COUNT(orders.order_id|interval=P1W)__missing" not in df.columns


def test_idempotent_under_median():
    once = impute_features(_matrix(), measure_strategy="median")
    twice = impute_features(once, measure_strategy="median")
    pd.testing.assert_frame_equal(once, twice)


def test_invalid_measure_strategy_raises():
    import pytest

    with pytest.raises(ValueError, match="measure_strategy"):
        impute_features(_matrix(), measure_strategy="bogus")
