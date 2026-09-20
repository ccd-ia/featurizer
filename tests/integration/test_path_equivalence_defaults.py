"""The render paths agree on the curated default primitives (matrix cell 14).

Every path-equality test before issue #36 used ``identity`` / ``abs`` and one
as-of date, and the TEMP-table path turned out to be wrong with windows, with an
interval and with two dates. Those three are pinned now, on hand-picked
transformers. This asks the same question of the set a config gets when it
selects nothing: ``DEFAULT_AGGREGATIONS`` x ``DEFAULT_TRANSFORMATIONS``, one
interval, two as-of dates.

One numeric variable per entity keeps that matrix at 545 columns, which still
fits one query, so all three paths are compared: the single query, column
groups over inline CTEs, and a materialized child chain.
"""

from __future__ import annotations

import datetime
import tempfile

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import ColumnGroupSharder

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

AS_OF = [datetime.date(2024, 6, 1), datetime.date(2024, 8, 1)]


def _config(**extra) -> dict:
    config = {
        "target": "series",
        "max_depth": 2,
        "intervals": ["P90D"],
        "entities": [
            {
                "alias": "series",
                "table": "series",
                "id": "series_id",
                "variables": {"size": {"type": "numeric"}},
            },
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": "ts",
                "variables": {"x": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            }
        ],
    }
    config.update(extra)
    return config


def _seed(conn, *, paired: bool) -> None:
    create_temp_table(
        conn,
        "series",
        [("series_id", "int"), ("size", "double precision")],
        [(1, 10.0), (2, 20.0), (3, 30.0)],
    )
    create_temp_table(
        conn,
        "events",
        [
            ("event_id", "int"),
            ("series_id", "int"),
            ("ts", "date"),
            ("x", "double precision"),
        ],
        [
            (10, 1, "2024-02-10", 1.5),
            (11, 1, "2024-04-20", 4.0),
            (12, 1, "2024-05-30", 2.5),
            (13, 1, "2024-07-15", 9.0),
            (14, 2, "2024-05-01", 6.0),
            (15, 2, "2024-07-20", 3.0),
            (16, 1, "2024-09-30", 99.0),
        ],
    )
    if paired:
        create_temp_table(
            conn,
            "as_of_dates",
            [("as_of_date", "date"), ("cohort_id", "int")],
            [(AS_OF[0], 1), (AS_OF[0], 2), (AS_OF[1], 1), (AS_OF[1], 3)],
        )
    else:
        create_temp_table(
            conn, "as_of_dates", [("as_of_date", "date")], [(d,) for d in AS_OF]
        )


def _matrix(
    conn, config: dict, *, paired: bool, groups_of: int | None = None, **kwargs
) -> tuple[dict, bool, int]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    featurizer = Featurizer(handle.name, **kwargs)
    grouped = (
        ColumnGroupSharder(featurizer._plan, max_columns_per_group=groups_of).build()
        if groups_of
        else featurizer._grouped()
    )
    joined: dict = {}
    with conn.transaction(force_rollback=True):
        _seed(conn, paired=paired)
        with conn.cursor() as cur:
            if grouped.materialization is not None:
                for statement in grouped.materialization.ddl:
                    cur.execute(statement)
            for sql in grouped.queries.values():
                cur.execute(sql)
                names = [d.name for d in cur.description]
                for row in cur.fetchall():
                    record = dict(zip(names, row))
                    key = (record["as_of_date"], record["series_id"])
                    joined.setdefault(key, {}).update(record)
    return joined, grouped.materialization is not None, len(grouped.queries)


@pytest.mark.parametrize("paired", [False, True], ids=["dense", "paired-cohort"])
def test_inline_and_materialized_child_chains_agree_on_the_defaults(pg_conn, paired):
    config = _config(**({"as_of_dates": {"id_column": "cohort_id"}} if paired else {}))
    single, _, n_single = _matrix(pg_conn, config, paired=paired)
    groups, groups_materialized, n_groups = _matrix(
        pg_conn, config, paired=paired, groups_of=150
    )
    temp, temp_materialized, _ = _matrix(
        pg_conn, config, paired=paired, materialize_threshold=1
    )
    assert n_single == 1 and n_groups > 1
    assert not groups_materialized and temp_materialized
    assert len(next(iter(single.values()))) > 400, "not the curated defaults"

    for name, other in (("column groups", groups), ("TEMP tables", temp)):
        assert set(other) == set(single), name
        differing = [
            (key, column, value, other[key].get(column))
            for key, row in single.items()
            for column, value in row.items()
            if other[key].get(column) != value
        ]
        assert not differing, (
            f"{name}: {len(differing)} cells differ, e.g. {differing[:3]}"
        )
    inline = single
    # Not an empty agreement: windows and intervals produced values.
    assert any(
        value is not None
        for row in inline.values()
        for column, value in row.items()
        if "ROLLING_MEDIAN_7" in column or "interval=P90D" in column
    )
