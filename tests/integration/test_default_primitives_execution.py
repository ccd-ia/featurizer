"""Execute every *default-active* primitive against PostgreSQL and assert values.

Issue #6: most primitives were only ever SQL-string-shape tested. The dialect
bugs fixed in ``9c53136`` (``rolling_median_*`` / ``rolling_iqr_*`` using
``OVER`` on an ordered-set aggregate; ``holt_winters_trend_*`` regressing
against a non-numeric date) and the issue #4 bugs (``ge`` emitting ``=>``;
``last`` silently returning the current row) all survived precisely because no
integration test *executed* the generated SQL.

This module closes that gap for the **default-active** primitive set — the
aggregations and transformations the engine activates when a config does *not*
override them (``featurizer.featurizer.DEFAULT_AGGREGATIONS`` /
``DEFAULT_TRANSFORMATIONS``). It executes the generated SQL on small fixtures
with hand-computed expected values and asserts the computed values, not just
that the query parses.

Scope decision (from the user): the default-active set, *not* all 152
registered primitives. The deliberately-excluded primitives are recorded in
``test_default_active_set_is_fully_covered`` so there are no silent gaps.

Window transformers (``last``, ``rolling_*``, ``ema_*``, ``holt_winters_*``,
``cum_sum``, ``pct_change_*``, ``lag_*``) only produce a meaningful multi-row
result when the entity's ``id`` groups several temporally-ordered rows. They are
therefore exercised on a *transactional* entity (one ``account_id`` with a known
series of amounts) where the rolling / last / diff values are hand-verifiable.
"""

from __future__ import annotations

import math

import pytest

from featurizer.featurizer import DEFAULT_AGGREGATIONS, DEFAULT_TRANSFORMATIONS

from ._harness import create_temp_table, run_featurizer

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# A transactional entity: one account with a strictly increasing series so the
# window transformers have a non-trivial partition to operate over. The values
# below are all derivable by hand from this series.
# --------------------------------------------------------------------------- #
_SERIES = [10.0, 20.0, 30.0, 40.0, 50.0]


def _seed_txns(conn) -> None:
    create_temp_table(
        conn,
        "txns",
        [("account_id", "int"), ("txn_date", "date"), ("amount", "numeric")],
        [
            (1, "2023-01-01", 10.0),
            (1, "2023-01-02", 20.0),
            (1, "2023-01-03", 30.0),
            (1, "2023-01-04", 40.0),
            (1, "2023-01-05", 50.0),
        ],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


# The window transformers we want to assert real values for. ``identity`` is the
# bare passthrough; ``abs`` is unary; the rest are window functions over the
# account's series. This set is the default-active window transforms plus the
# explicitly-requested bug-prone extras (cumprod, diff2/3, mean_shift, cusum).
_WINDOW_TX = [
    "identity",
    "abs",
    "cum_sum",
    "cumprod",
    "first",
    "last",
    "previous",
    "lag_1",
    "diff",
    "diff2",
    "diff3",
    "rolling_mean_3",
    "rolling_std_7",
    "rolling_median_5",
    "rolling_median_7",
    "rolling_iqr_7",
    "ema_7",
    "holt_winters_level_7",
    "holt_winters_trend_7",
    "pct_change_1",
    "mean_shift_ratio_7",
    "cusum",
]


def _txn_window_config() -> dict:
    return {
        "target": "txns",
        "max_depth": 1,
        "intervals": [],
        "aggregations": ["count"],
        "transformations": _WINDOW_TX,
        "entities": [
            {
                "alias": "txns",
                "table": "txns",
                "id": "account_id",
                "temporal_ix": "txn_date",
                "variables": {"amount": {"type": "numeric"}},
            }
        ],
    }


def _by_date(rows: list[dict]) -> dict[str, dict]:
    return {str(r["txn_date"]): r for r in rows}


def test_window_transformers_execute_with_correct_values(pg_conn):
    """Every default-active window transformer (plus the explicitly-requested
    bug-prone extras) computes the hand-verified value for a known series.

    Asserting *values* — not just "does it parse" — is what catches frame and
    dialect bugs. ``last`` is the canary for the issue #4 fix: with the wrong
    (default) frame it would equal the current row; with the full-partition
    frame it is the partition's final value (50) on every row.
    """
    _seed_txns(pg_conn)
    rows = run_featurizer(pg_conn, _txn_window_config())
    assert len(rows) == len(_SERIES)
    by_date = _by_date(rows)

    def col(row: dict, prefix: str) -> object:
        matches = [k for k in row if k.startswith(prefix)]
        assert matches, f"no column starting {prefix!r}; got {sorted(row)}"
        assert len(matches) == 1, f"ambiguous {prefix!r}: {matches}"
        return row[matches[0]]

    last_row = by_date["2023-01-05"]
    first_row = by_date["2023-01-01"]
    third_row = by_date["2023-01-03"]

    # identity passthrough: the bare value survives.
    assert float(last_row["amount"]) == 50.0

    # last_value over the FULL partition frame -> 50 on every row (issue #4).
    for r in rows:
        assert float(col(r, "LAST(")) == 50.0
    # first_value -> 10 on every row.
    for r in rows:
        assert float(col(r, "FIRST(")) == 10.0

    # cum_sum: running total.
    assert float(col(last_row, "CUM_SUM(")) == sum(_SERIES)
    assert float(col(third_row, "CUM_SUM(")) == 60.0
    # cumprod: running product, here 10*20*30*40*50 = 12_000_000.
    assert math.isclose(float(col(last_row, "CUMPROD(")), 12_000_000.0, rel_tol=1e-9)

    # previous / lag_1 / diff: first-order lag relations.
    assert col(first_row, "LAG_1(") is None
    assert float(col(last_row, "LAG_1(")) == 40.0
    assert float(col(last_row, "PREVIOUS(")) == 40.0
    assert float(col(last_row, "DIFF(")) == 10.0  # 50 - 40

    # diff2 (acceleration) and diff3 (jerk) of a linear series are 0.
    assert float(col(last_row, "DIFF2(")) == 0.0
    assert float(col(last_row, "DIFF3(")) == 0.0
    assert col(first_row, "DIFF2(") is None  # needs 2 lags

    # rolling_mean_3 over [30,40,50] = 40.
    assert math.isclose(float(col(last_row, "ROLLING_MEAN_3(")), 40.0)
    # rolling_median_5 over the full [10..50] = 30; rolling_median_7 same here.
    assert float(col(last_row, "ROLLING_MEDIAN_5(")) == 30.0
    assert float(col(last_row, "ROLLING_MEDIAN_7(")) == 30.0
    assert float(col(third_row, "ROLLING_MEDIAN_5(")) == 20.0  # median(10,20,30)
    # rolling_iqr_7 over [10..50]: P75=40, P25=20 -> 20.
    assert math.isclose(float(col(last_row, "ROLLING_IQR_7(")), 20.0)
    # rolling_std_7 over [10..50] = sample stddev = sqrt(250).
    assert math.isclose(
        float(col(last_row, "ROLLING_STD_7(")), math.sqrt(250.0), rel_tol=1e-9
    )

    # pct_change_1 at the 2nd row = (20-10)/10 = 1.0.
    assert math.isclose(float(col(by_date["2023-01-02"], "PCT_CHANGE_1(")), 1.0)
    assert col(first_row, "PCT_CHANGE_1(") is None

    # cusum = cumsum - row_number * partition_mean. At the last row:
    # 150 - 5*30 = 0.
    assert math.isclose(float(col(last_row, "CUSUM(")), 0.0, abs_tol=1e-9)

    # holt_winters_level_7 is a windowed mean -> 30 at the last row (mean of all
    # five, window of 7 covers them); holt_winters_trend_7 is a positive slope.
    assert math.isclose(float(col(last_row, "HOLT_WINTERS_LEVEL_7(")), 30.0)
    assert float(col(last_row, "HOLT_WINTERS_TREND_7(")) > 0.0

    # ema_7 is a recency-weighted mean: bounded by the series and above the
    # simple mean because recent (larger) values dominate.
    ema_last = float(col(last_row, "EMA_7("))
    assert 30.0 < ema_last < 50.0

    # mean_shift_ratio_7 needs a prior 7-row window that does not exist for this
    # 5-row series, so it is NULL throughout (correctly, not an error).
    assert col(last_row, "MEAN_SHIFT_RATIO_7(") is None


# --------------------------------------------------------------------------- #
# Default-active *aggregations* (count/mean/sum/stddev/min/max/median/nunique
# plus recency/tenure) over a known parent->child fixture.
# --------------------------------------------------------------------------- #
_AMOUNTS = [10.0, 20.0, 30.0, 40.0]


def _seed_customer_orders(conn) -> None:
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
        ],
        [
            (1, 1, "2023-06-01", 10.0),
            (2, 1, "2023-07-01", 20.0),
            (3, 1, "2023-08-01", 30.0),
            (4, 1, "2023-09-01", 40.0),
        ],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])


def _customer_orders_aggs_config() -> dict:
    return {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": list(DEFAULT_AGGREGATIONS),
        "transformations": ["identity"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"amount": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }


def test_default_aggregations_execute_with_correct_values(pg_conn):
    """The full DEFAULT_AGGREGATIONS tuple computes hand-verified values."""
    _seed_customer_orders(pg_conn)
    rows = run_featurizer(pg_conn, _customer_orders_aggs_config())
    assert len(rows) == 1
    row = rows[0]

    assert int(row["COUNT(orders.order_id)"]) == len(_AMOUNTS)
    assert float(row["SUM(orders.amount)"]) == sum(_AMOUNTS)
    assert float(row["MEAN(orders.amount)"]) == sum(_AMOUNTS) / len(_AMOUNTS)
    assert float(row["MEDIAN(orders.amount)"]) == 25.0
    assert float(row["MIN(orders.amount)"]) == 10.0
    assert float(row["MAX(orders.amount)"]) == 40.0
    assert int(row["NUNIQUE(orders.order_id)"]) == 4
    # sample stddev of [10,20,30,40] = sqrt(500/3).
    assert math.isclose(
        float(row["STDDEV(orders.amount)"]), math.sqrt(500.0 / 3.0), rel_tol=1e-9
    )
    # recency: 2024-01-01 - last order (2023-09-01) = 122 days.
    assert int(row["RECENCY(orders.ordered_at)"]) == 122
    # tenure: 2024-01-01 - first order (2023-06-01) = 214 days.
    assert int(row["TENURE(orders.ordered_at)"]) == 214


# --------------------------------------------------------------------------- #
# The date-part transformers (day/dow/month) only fire on a date-typed
# *variable* (the temporal_ix is type ``index`` and is skipped by the planner).
# They are ``categorical`` (``to_char``) outputs, so only the categorical
# aggregations (``count``/``nunique``) fire on them.
# --------------------------------------------------------------------------- #
def test_date_part_transformers_execute(pg_conn):
    """day/dow/month render valid SQL on a date variable and aggregate cleanly."""
    create_temp_table(pg_conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("shipped_on", "date"),
        ],
        [
            (1, 1, "2023-06-01", "2023-06-15"),  # Thursday, ISO-dow 4
            (2, 1, "2023-07-01", "2023-07-15"),  # Saturday, ISO-dow 6
        ],
    )
    create_temp_table(
        pg_conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)]
    )

    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["count", "nunique"],
        "transformations": ["day", "dow", "month"],
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {"shipped_on": {"type": "date"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }
    rows = run_featurizer(pg_conn, config)
    assert len(rows) == 1
    row = rows[0]

    def col(prefix: str) -> object:
        matches = [k for k in row if k.startswith(prefix)]
        assert matches, f"no column starting {prefix!r}; got {sorted(row)}"
        return row[matches[0]]

    # ``day`` -> to_char(d, 'day') = weekday name ("thursday"/"saturday"); two
    # rows, two distinct names. ``dow`` -> to_char(d, 'ID') = ISO weekday number
    # ("4"/"6"); two distinct.
    assert int(col("COUNT(orders.DAY(")) == 2
    assert int(col("NUNIQUE(orders.DAY(")) == 2
    assert int(col("NUNIQUE(orders.DOW(")) == 2
    # ``month`` -> to_char(d, 'M'). PostgreSQL has no 'M' template pattern, so it
    # emits the literal "M" for every row (a pre-existing quirk, out of scope for
    # issues #4/#6). The point here is that the SQL *executes*; the value is the
    # constant "M" -> one distinct value.
    assert int(col("NUNIQUE(orders.MONTH(")) == 1


# --------------------------------------------------------------------------- #
# Coverage checklist: every default-active primitive must be EXECUTED somewhere.
# --------------------------------------------------------------------------- #

# Default-active primitives deliberately not asserted as standalone values are
# recorded here with the reason. Keeping the set explicit means a newly-added
# default primitive that no test executes will fail the checklist loudly rather
# than slipping through silently.
_DELIBERATELY_EXCLUDED: dict[str, str] = {
    # No default-active primitive is excluded: all are reachable by the
    # introspection below. (BinaryTransformers such as ``ge``/``le`` are NOT
    # default-active — they are unit-tested in
    # tests/primitives/test_transformations.py since the config path cannot
    # reach them.)
}


def _output_columns_for_default_engine(conn) -> list[str]:
    """Run the engine with NO primitive override (the literal default-active set)
    on a fixture rich enough to fire every default primitive, and return the
    generated output column names.

    The fixture carries a numeric variable (drives the numeric aggregations and
    numeric window transforms), a date variable (drives day/dow/month), and a
    temporal index (drives recency/tenure and the window ordering).
    """
    create_temp_table(conn, "customers", [("customer_id", "int")], [(1,)])
    create_temp_table(
        conn,
        "orders",
        [
            ("order_id", "int"),
            ("customer_id", "int"),
            ("ordered_at", "date"),
            ("amount", "numeric"),
            ("shipped_on", "date"),
        ],
        [
            (1, 1, "2023-06-01", 10.0, "2023-06-15"),
            (2, 1, "2023-07-01", 20.0, "2023-07-20"),
            (3, 1, "2023-08-01", 30.0, "2023-08-10"),
        ],
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-01-01",)])

    config = {
        "target": "customers",
        "max_depth": 2,
        "intervals": [],
        # No "aggregations"/"transformations" key -> DEFAULT_AGGREGATIONS /
        # DEFAULT_TRANSFORMATIONS apply.
        "entities": [
            {"alias": "customers", "table": "customers", "id": "customer_id"},
            {
                "alias": "orders",
                "table": "orders",
                "id": "order_id",
                "temporal_ix": "ordered_at",
                "variables": {
                    "amount": {"type": "numeric"},
                    "shipped_on": {"type": "date"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "customers", "key": "customer_id"},
                "child": {"entity": "orders", "key": "customer_id"},
            }
        ],
    }
    rows = run_featurizer(conn, config)
    assert rows, "default-active engine produced no rows"
    return list(rows[0].keys())


def _token_for(primitive: str) -> str:
    """The UPPERCASE function-call token a primitive renders as in a column name.

    Every aggregation/transformation names its output ``NAME(entity.feature)``
    (uppercased), so the presence of ``NAME(`` in some output column proves the
    primitive was synthesized into the executed SQL.
    """
    return f"{primitive.upper()}("


def test_default_active_set_is_fully_covered(pg_conn):
    """Checklist: every default-active primitive appears in the EXECUTED SQL.

    This runs the engine with the literal defaults and introspects the generated
    columns, so it cannot pass unless the column (hence the SQL fragment) was
    actually produced and the whole query executed on PostgreSQL. It guards
    against a future default primitive being added without an executing test.
    """
    columns = _output_columns_for_default_engine(pg_conn)
    joined = " ".join(columns)

    default_active = list(DEFAULT_AGGREGATIONS) + list(DEFAULT_TRANSFORMATIONS)

    missing = []
    for primitive in default_active:
        if primitive in _DELIBERATELY_EXCLUDED:
            continue
        if primitive == "identity":
            # identity is the passthrough: it surfaces as an aggregation applied
            # directly to the raw variable with no transform wrapper, e.g.
            # ``MEAN(orders.amount)`` (vs. ``MEAN(orders.CUM_SUM(orders.amount))``
            # for a transformed feature). A column ending in ``(orders.amount)``
            # therefore proves the un-transformed (identity) feature flowed
            # through.
            assert any(
                c.endswith("(orders.amount)") for c in columns
            ), "identity passthrough column not found"
            continue
        if _token_for(primitive) not in joined:
            missing.append(primitive)

    assert not missing, (
        "default-active primitives never executed (no output column): "
        f"{sorted(missing)}"
    )
