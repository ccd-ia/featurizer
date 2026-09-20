# coding: utf-8

"""A declared name means what it means in PostgreSQL (issue #44).

#13, #24 and #29 delimited every declared column name, byte for byte. That was
right for a name that cannot be a bare identifier — ``MEAN(games.goals)``,
``Amount USD`` — and wrong for one that can: a config that spelled a column
``totalAmount`` against a table created with unquoted DDL (PostgreSQL stores
``totalamount``) ran on v1.2.0, where the name was interpolated bare and folded,
and stopped running once the name was delimited exactly.

The rule is PostgreSQL's own:

- a valid bare identifier folds to lower case, and is then delimited (which is
  what makes a reserved word usable);
- anything else is delimited byte for byte;
- a name the config already wraps in double quotes is taken exactly. That is
  how a genuinely mixed-case column is asked for, as it is in SQL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from featurizer import Featurizer
from featurizer.primitives.abstractions import quote_if_bare

RULE = [
    # a valid bare identifier: folded, then delimited
    ("amount", '"amount"'),
    ("totalAmount", '"totalamount"'),
    ("Region", '"region"'),
    ("_x9$", '"_x9$"'),
    ("order", '"order"'),  # reserved: delimiting is what makes it usable
    ("Order", '"order"'),
    # PostgreSQL folds the ASCII letters of an identifier and leaves the rest
    ("Año", '"año"'),
    ("AÑO", '"aÑo"'),
    # not a bare identifier: there is one way to read it
    ("Amount USD", '"Amount USD"'),
    ("MEAN(games.goals)", '"MEAN(games.goals)"'),
    ("9lives", '"9lives"'),
    ("$amount", '"$amount"'),
    ('he"llo', '"he""llo"'),
    # already delimited: exact, the config asked for it
    ('"entityId"', '"entityId"'),
    ('"MEAN(orders.amount)"', '"MEAN(orders.amount)"'),
]


@pytest.mark.parametrize("declared,rendered", RULE)
def test_the_rule(declared: str, rendered: str) -> None:
    assert quote_if_bare(declared) == rendered


def test_the_rule_is_idempotent() -> None:
    """A rendered name is delimited, so rendering it again changes nothing."""
    for declared, _ in RULE:
        once = quote_if_bare(declared)
        assert quote_if_bare(once) == once


# ------------------------------------------------------------- execution


def _config(tmp_path: Path, target_variable: str, child_variable: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: series\n"
        "max_depth: 2\n"
        "intervals: []\n"
        "aggregations: [max]\n"
        "transformations: [identity, abs]\n"
        "entities:\n"
        "  - alias: series\n"
        "    id: series_id\n"
        "    table: series\n"
        "    variables:\n"
        f"      {json.dumps(target_variable)}:\n"
        "        type: numeric\n"
        "  - alias: events\n"
        "    id: event_id\n"
        "    table: events\n"
        "    temporal_ix: ts\n"
        "    variables:\n"
        f"      {json.dumps(child_variable)}:\n"
        "        type: numeric\n"
        "relationships:\n"
        "  - parent: {entity: series, key: series_id}\n"
        "    child: {entity: events, key: series_id}\n"
    )
    return str(path)


def _run(query: str, region_ddl: str, amount_ddl: str) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"create temp table series (series_id int, {region_ddl} numeric)")
        cur.execute("insert into series values (1, 2)")
        cur.execute(
            "create temp table events (event_id int, series_id int, ts date, "
            f"{amount_ddl} numeric)"
        )
        cur.execute("insert into events values (1, 1, '2024-03-01', -3)")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute("insert into as_of_dates values ('2024-06-01')")
        cur.execute(query)
        names = [d.name for d in cur.description]
        row = dict(zip(names, cur.fetchone()))
        conn.rollback()
    return row


@pytest.mark.integration
def test_a_mixed_case_spelling_of_a_folded_column_runs_as_it_did_on_v1_2_0(
    tmp_path,
) -> None:
    """Unquoted DDL, so PostgreSQL stores ``region`` and ``totalamount``."""
    query = Featurizer(_config(tmp_path, "Region", "totalAmount")).query
    row = _run(query, region_ddl="Region", amount_ddl="totalAmount")
    # The columns v1.2.0 (011aaf2) returned for this config, measured in #44.
    assert row["region"] == 2
    assert row["MAX(events.totalAmount)"] == -3
    assert row["MAX(events.ABS(events.totalAmount))"] == 3


@pytest.mark.integration
def test_a_genuinely_mixed_case_column_is_asked_for_with_its_quotes(tmp_path) -> None:
    """Quoted DDL (what pandas writes), so the stored names keep their case."""
    query = Featurizer(_config(tmp_path, '"Region"', '"totalAmount"')).query
    row = _run(query, region_ddl='"Region"', amount_ddl='"totalAmount"')
    assert row["Region"] == 2
    assert row["MAX(events.totalAmount)"] == -3


@pytest.mark.integration
def test_a_reserved_word_in_any_case_runs(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, "Order", "Select")).query
    row = _run(query, region_ddl='"order"', amount_ddl='"select"')
    assert row["order"] == 2
    assert row["MAX(events.Select)"] == -3


# ------------------------------------------------------------- names


def test_output_names_keep_the_declared_spelling(tmp_path) -> None:
    """ADR-0007: a derived name is built from the declared name, not from the
    rendered reference, so the fold does not reach it."""
    f = Featurizer(_config(tmp_path, "Region", "totalAmount"))
    labels = {entry.label for entry in f.feature_manifest}
    assert "MAX(events.totalAmount)" in labels
    assert "ABS(series.MAX(events.totalAmount))" in labels


def test_a_pre_quoted_name_leaves_no_quotes_in_an_output_name(tmp_path) -> None:
    f = Featurizer(_config(tmp_path, '"Region"', '"totalAmount"'))
    for entry in f.feature_manifest:
        assert '"' not in entry.label, entry.label
    labels = {entry.label for entry in f.feature_manifest}
    assert "MAX(events.totalAmount)" in labels
