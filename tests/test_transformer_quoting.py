# coding: utf-8

"""Transformers quote the column they wrap (issue #18).

#13 quoted the places a declared column name is *projected*. A transformer does
not project its input — it wraps it in SQL, and it wrapped ``feature.name``
raw. A declared column named like an aggregate call rendered::

    abs(MEAN(games.goals))   as "ABS(wide.MEAN(games.goals))"

which PostgreSQL reads as ``abs()`` over the aggregate ``MEAN()`` over a column
of a table ``games`` that is not in the FROM clause. Every transformer that
wraps its input had the fault — 76 of the 83 registered.

The fix quotes the input at each SQL-emission site through ``_col``. What moves
and what does not, pinned below:

- output column names and labels: unchanged (ADR-0007);
- a declared variable's own ``definition``: unchanged;
- a *derived* feature's ``definition`` — which is its SQL — now carries the
  quoted input (``abs(age)`` becomes ``abs("age")``). That text is persisted
  in ``<stem>_manifest.definition``; no known consumer reads it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import get_transformers, list_transformations

WEIRD = "MEAN(games.goals)"


def _config(tmp_path: Path, variable: str, transformations: list[str], vtype="numeric"):
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: wide\n"
        "max_depth: 1\n"
        "intervals: []\n"
        "aggregations: []\n"
        f"transformations: [{', '.join(transformations)}]\n"
        "entities:\n"
        "  - alias: wide\n"
        "    id: entity_id\n"
        "    table: wide\n"
        "    temporal_ix: as_of\n"
        "    variables:\n"
        f"      {json.dumps(variable)}:\n"
        f"        type: {vtype}\n"
    )
    return str(path)


# ------------------------------------------------------------- rendering


def test_abs_wraps_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["abs"])).query
    assert f'abs("{WEIRD}")' in query
    assert f"abs({WEIRD})" not in query


def test_window_transformers_wrap_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["lag_1", "cum_sum"])).query
    assert f'lag("{WEIRD}", 1)' in query
    assert f'sum("{WEIRD}")' in query


def test_text_transformers_wrap_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, "body text", ["num_chars"], "text")).query
    assert 'char_length("body text")' in query


def test_output_names_do_not_move(tmp_path) -> None:
    """ADR-0007: names are built by _name_label, which never sees _col."""
    f = Featurizer(_config(tmp_path, "age", ["abs", "lag_1"]))
    labels = {e.label for e in f.feature_manifest}
    assert {"ABS(wide.age)", "LAG_1(wide.age)"} <= labels


def test_a_declared_variables_own_definition_does_not_move(tmp_path) -> None:
    f = Featurizer(_config(tmp_path, "age", ["identity"]))
    (entry,) = [e for e in f.feature_manifest if e.kind == "variable"]
    assert entry.definition == "age"


def test_a_derived_definition_carries_the_quoted_input(tmp_path) -> None:
    """The one persisted change, pinned so it cannot drift again unnoticed."""
    f = Featurizer(_config(tmp_path, "age", ["abs"]))
    (entry,) = [e for e in f.feature_manifest if e.kind == "derived"]
    assert entry.definition.strip() == 'abs("age")'


# ------------------------------------------------------------- execution

SQL_TYPE = {"numeric": "double precision", "date": "date", "text": "text"}
SAMPLE = {
    "numeric": ["10.0", "20.0", "5.5"],
    "date": ["date '2024-01-01'", "date '2024-02-15'", "date '2024-03-31'"],
    "text": ["'alpha beta'", "'Gamma!'", "'delta, epsilon.'"],
}

# Fails identically on a PLAIN column name, so it says nothing about quoting.
# ``cdf`` renders ``cum_dist()``, which PostgreSQL does not have. It stays
# broken on purpose: ``cume_dist()`` reads rows dated after the as-of date
# (issue #27). The other four from issue #23 are fixed; their values are pinned
# in tests/primitives/test_transformer_defects.py.
KNOWN_BROKEN = {"cdf"}
# Not configurable as a bare transformations: entry.
NOT_STANDALONE = {"identity", "in_array"}


def _sweep_cases():
    for name in sorted(list_transformations()):
        if name in NOT_STANDALONE:
            continue
        types = getattr(get_transformers([name])[name], "input_types", None) or [
            "numeric"
        ]
        vtype = next((t for t in types if t in SQL_TYPE), None)
        if vtype is None:
            continue
        marks = (
            [
                pytest.mark.xfail(
                    reason="deliberately left broken, see issue #27", strict=True
                )
            ]
            if name in KNOWN_BROKEN
            else []
        )
        yield pytest.param(name, vtype, id=name, marks=marks)


@pytest.mark.integration
@pytest.mark.parametrize("name,vtype", list(_sweep_cases()))
def test_every_transformer_executes_over_an_aggregate_shaped_column(
    tmp_path, name, vtype
) -> None:
    """76 of 83 raised UndefinedTable before the fix. Now: parity with plain names."""
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    query = Featurizer(_config(tmp_path, WEIRD, [name], vtype)).query
    rows = ", ".join(
        f"({i + 1}, date '2024-0{i + 1}-01', {v})" for i, v in enumerate(SAMPLE[vtype])
    )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(
            f'create temp table wide (entity_id int, as_of date, "{WEIRD}" '
            f"{SQL_TYPE[vtype]})"
        )
        cur.execute(f"insert into wide values {rows}")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute("insert into as_of_dates values (date '2024-06-01')")
        cur.execute(query)
        assert len(cur.fetchall()) == 3
        conn.rollback()
