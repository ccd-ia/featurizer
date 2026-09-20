# coding: utf-8

"""Four transformers that rendered SQL PostgreSQL would not run (issue #23).

The 83-transformer sweep in ``tests/test_transformer_quoting.py`` only asks
whether a transformer *executes*. These tests pin what it returns, because each
of these had a second fault behind the first:

- ``hourly_bin``: no ``end`` on the ``case``; behind it, ``numeric <@
  int4range`` has no operator (``extract`` returns numeric since PostgreSQL 14)
  and ``extract(hour from <date>)`` raises.
- ``daily_bin``: no ``end``; behind it, both branches said ``'weekday'`` and the
  ISO ranges left Sunday (7) unmatched.
- ``cumprod``: the ``case`` guard cannot stop ``ln`` — a window aggregate is
  evaluated over the whole frame before the ``case`` picks a branch.
- ``ema_7`` / ``ema_14``: the weight is ``exp(decay * days_since_1970)``, which
  overflows ``float8``; a ``double precision`` or ``real`` column dragged the
  multiplication into floating point.

``cdf`` was the fifth entry on the issue. It was left broken until a child's read
was cut on the as-of date (issue #27); its values are pinned in
``tests/integration/test_asof_bounded_child_read.py``.
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from featurizer import Featurizer

# Saturday 03:00, Sunday 12:00, Monday 23:30.
STAMPS = ["2024-01-06 03:00", "2024-01-07 12:00", "2024-01-08 23:30"]


def _query(tmp_path: Path, transformer: str, vtype: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: wide\n"
        "max_depth: 1\n"
        "intervals: []\n"
        "aggregations: []\n"
        f"transformations: [{transformer}]\n"
        "entities:\n"
        "  - alias: wide\n"
        "    id: entity_id\n"
        "    table: wide\n"
        "    temporal_ix: as_of\n"
        "    variables:\n"
        "      x:\n"
        f"        type: {vtype}\n"
    )
    return Featurizer(str(path)).query


def _run(tmp_path: Path, transformer: str, vtype: str, sqltype: str, values: list[str]):
    """One entity, one row per stamp; returns the transformer column in time order."""
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    query = _query(tmp_path, transformer, vtype)
    rows = ", ".join(
        f"(1, timestamp '{stamp}', {value})" for stamp, value in zip(STAMPS, values)
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            f"create temp table wide (entity_id int, as_of timestamp, x {sqltype})"
        )
        cur.execute(f"insert into wide values {rows}")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute("insert into as_of_dates values (date '2030-01-01')")
        cur.execute(query)
        columns = [c.name for c in cur.description]
        out = columns.index(f"{transformer.upper()}(wide.x)")
        when = columns.index("as_of")
        result = [row[out] for row in sorted(cur.fetchall(), key=lambda r: r[when])]
        conn.rollback()
    return result


def _stamps_as_sql(cast: str = "") -> list[str]:
    return [f"timestamp '{s}'{cast}" for s in STAMPS]


# ------------------------------------------------------------- rendering


@pytest.mark.parametrize("transformer", ["hourly_bin", "daily_bin"])
def test_bin_case_expressions_are_closed(tmp_path, transformer) -> None:
    query = _query(tmp_path, transformer, "timestamp")
    alias = f'as "{transformer.upper()}(wide.x)"'
    expression = query[query.index("case") : query.index(alias)]
    assert "end" in [line.strip() for line in expression.splitlines()]


def test_daily_bin_has_a_weekend_and_reaches_sunday(tmp_path) -> None:
    query = _query(tmp_path, "daily_bin", "timestamp")
    assert "int4range(1,6) then 'weekday'" in query
    assert "int4range(6,8) then 'weekend'" in query


def test_cumprod_guards_inside_the_aggregate(tmp_path) -> None:
    query = _query(tmp_path, "cumprod", "numeric")
    assert 'sum(ln(case when "x" > 0 then "x" end))' in query


def test_ema_keeps_the_weight_out_of_floating_point(tmp_path) -> None:
    query = _query(tmp_path, "ema_7", "numeric")
    assert 'sum("x"::numeric * exp(' in query
    assert 'extract(epoch from "as_of")::numeric' in query


# ------------------------------------------------------------- execution


@pytest.mark.integration
def test_daily_bin_labels_the_weekend_including_sunday(tmp_path) -> None:
    got = _run(tmp_path, "daily_bin", "timestamp", "timestamp", _stamps_as_sql())
    assert got == ["weekend", "weekend", "weekday"]


@pytest.mark.integration
def test_daily_bin_accepts_a_date_column(tmp_path) -> None:
    got = _run(tmp_path, "daily_bin", "date", "date", _stamps_as_sql("::date"))
    assert got == ["weekend", "weekend", "weekday"]


@pytest.mark.integration
def test_hourly_bin_labels_the_hour(tmp_path) -> None:
    got = _run(tmp_path, "hourly_bin", "timestamp", "timestamp", _stamps_as_sql())
    assert got == ["night", "midday", "night"]


@pytest.mark.integration
def test_hourly_bin_bins_a_date_as_midnight(tmp_path) -> None:
    """``extract(hour from <date>)`` raises; ``hour`` answers '00' for a date."""
    got = _run(tmp_path, "hourly_bin", "date", "date", _stamps_as_sql("::date"))
    assert got == ["night", "night", "night"]


@pytest.mark.integration
def test_cumprod_is_the_running_product(tmp_path) -> None:
    got = _run(tmp_path, "cumprod", "numeric", "double precision", ["2", "3", "4"])
    assert got == pytest.approx([2.0, 6.0, 24.0])


@pytest.mark.integration
@pytest.mark.parametrize("poison", ["-3", "0"], ids=["negative", "zero"])
def test_cumprod_goes_null_instead_of_raising(tmp_path, poison) -> None:
    """The documented limitation, which never happened: ``ln`` raised first."""
    got = _run(tmp_path, "cumprod", "numeric", "double precision", ["2", poison, "4"])
    assert got[0] == pytest.approx(2.0)
    assert got[1:] == [None, None]


def _ema_oracle(decay: float, values: list[float]) -> list[float]:
    """The same average, weighted relative to the current row instead of 1970."""
    days = [datetime.fromisoformat(s).timestamp() / 86400.0 for s in STAMPS]
    out = []
    for i in range(len(values)):
        weights = [math.exp(decay * (days[j] - days[i])) for j in range(i + 1)]
        out.append(sum(w * v for w, v in zip(weights, values)) / sum(weights))
    return out


@pytest.mark.integration
@pytest.mark.parametrize("sqltype", ["double precision", "real", "numeric", "integer"])
@pytest.mark.parametrize("name,decay", [("ema_7", 0.25), ("ema_14", 0.15)])
def test_ema_gives_the_same_answer_on_every_numeric_type(
    tmp_path, name, decay, sqltype
) -> None:
    """``double precision`` and ``real`` raised NumericValueOutOfRange."""
    got = _run(tmp_path, name, "numeric", sqltype, ["10", "20", "30"])

    assert all(isinstance(v, Decimal) for v in got)
    assert [float(v) for v in got] == pytest.approx(
        _ema_oracle(decay, [10.0, 20.0, 30.0]), rel=1e-12
    )
