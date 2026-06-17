"""Unit tests for the Arrow-native imputation pass and the leakage gate.

These do not touch a database; they exercise ``impute_arrow`` and
``guard_full_matrix_fit`` on hand-built pyarrow tables. The COPY -> Arrow export
itself is covered by ``tests/integration/test_arrow_export.py``.
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")

from featurizer.imputation import (  # noqa: E402
    MISSING_INDICATOR_SUFFIX,
    guard_full_matrix_fit,
    impute_arrow,
)


def _matrix() -> "pa.Table":
    """A small matrix with one all-present row and one all-null feature row."""
    return pa.table(
        {
            "as_of_date": pa.array(["2024-01-01", "2024-01-01", "2024-01-01"]),
            "cust_id": pa.array([1, 2, 3], type=pa.int64()),
            "COUNT(orders.order_id|interval=P1W)": pa.array(
                [1, None, 3], type=pa.int64()
            ),
            "SUM(orders.amount)": pa.array([10.0, None, 30.0], type=pa.float64()),
            "AVG(orders.amount)": pa.array([5.0, None, 15.0], type=pa.float64()),
            "RECENCY(orders.ts)": pa.array([2.0, None, 4.0], type=pa.float64()),
        }
    )


_KEYS = ("as_of_date", "cust_id")


def test_count_filled_zero_measures_stay_null():
    out = impute_arrow(_matrix(), key_columns=_KEYS)
    d = out.to_pydict()
    # count-like -> 0 (structural zero), measures left null
    assert d["COUNT(orders.order_id|interval=P1W)"][1] == 0
    assert d["SUM(orders.amount)"][1] == 0
    assert d["AVG(orders.amount)"][1] is None
    assert d["RECENCY(orders.ts)"][1] is None


def test_missing_indicators_recorded_before_fill():
    out = impute_arrow(_matrix(), key_columns=_KEYS)
    d = out.to_pydict()
    for col in (
        "COUNT(orders.order_id|interval=P1W)",
        "SUM(orders.amount)",
        "AVG(orders.amount)",
        "RECENCY(orders.ts)",
    ):
        ind = f"{col}{MISSING_INDICATOR_SUFFIX}"
        assert ind in out.column_names
        assert d[ind] == [0, 1, 0]


def test_key_columns_are_never_features():
    out = impute_arrow(_matrix(), key_columns=_KEYS)
    # keys are passed through untouched and get no missing indicators
    assert f"cust_id{MISSING_INDICATOR_SUFFIX}" not in out.column_names
    assert f"as_of_date{MISSING_INDICATOR_SUFFIX}" not in out.column_names
    assert out.to_pydict()["cust_id"] == [1, 2, 3]


def test_nulls_preserved_as_arrow_nulls_not_nan():
    out = impute_arrow(_matrix(), key_columns=_KEYS)
    # measures that stay null must remain Arrow nulls, not NaN floats
    assert out.column("AVG(orders.amount)").null_count == 1


def test_measure_strategy_mean_fills_measures():
    out = impute_arrow(_matrix(), key_columns=_KEYS, measure_strategy="mean")
    d = out.to_pydict()
    # mean of [5, 15] = 10 ; mean of [2, 4] = 3
    assert d["AVG(orders.amount)"][1] == 10.0
    assert d["RECENCY(orders.ts)"][1] == 3.0
    # count-like still filled with 0, not the mean
    assert d["SUM(orders.amount)"][1] == 0


def test_count_fill_zero_can_be_disabled():
    out = impute_arrow(_matrix(), key_columns=_KEYS, count_fill_zero=False)
    assert out.to_pydict()["SUM(orders.amount)"][1] is None


def test_input_not_mutated():
    table = _matrix()
    impute_arrow(table, key_columns=_KEYS)
    assert table.column("SUM(orders.amount)").null_count == 1
    assert f"SUM(orders.amount){MISSING_INDICATOR_SUFFIX}" not in table.column_names


def test_invalid_measure_strategy_raises():
    with pytest.raises(ValueError, match="measure_strategy"):
        impute_arrow(_matrix(), key_columns=_KEYS, measure_strategy="bogus")


# --------------------------------------------------------------------------- #
# Leakage gate (guard_full_matrix_fit) — shared by to_dataframe/to_arrow/to_parquet
# --------------------------------------------------------------------------- #


def test_gate_allows_none_strategy_silently():
    # no warning, no error for the default (non-fitting) path
    guard_full_matrix_fit("none", allow_full_matrix_fit=False, caller="to_arrow")


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_gate_refuses_fit_without_optin(strategy):
    with pytest.raises(ValueError, match="temporal leakage"):
        guard_full_matrix_fit(strategy, allow_full_matrix_fit=False, caller="to_arrow")


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_gate_warns_when_optin(strategy):
    with pytest.warns(UserWarning, match="temporal leakage"):
        guard_full_matrix_fit(strategy, allow_full_matrix_fit=True, caller="to_arrow")


def test_gate_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="measure_strategy"):
        guard_full_matrix_fit("bogus", allow_full_matrix_fit=True, caller="to_arrow")
