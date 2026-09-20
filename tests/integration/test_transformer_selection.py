"""A transformer can be selected without ``identity`` (issue #38).

Every documented example lists ``identity`` first, so nothing ever ran a
``transformations:`` list that leaves it out. Such a list failed as soon as a
child entity had a variable its parent aggregates::

    events_transform as (select "event_id", "ts", abs("x") as "ABS(events.x)" ...)
    events_aggs_for_series as (select ..., max( "x" ) ... from events_transform)
    -- UndefinedColumn: column "x" does not exist

The planner had planned ``MAX(events.x)`` and everything built on it, and the
manifest listed it; only the child's transform was short one projection. A
non-target transform now also carries the raw columns its parent reads, as
helpers. They are not output: without ``identity`` the target still emits only
what the selected transformers produce.
"""

from __future__ import annotations

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import get_transformers, list_transformations

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

SQL_TYPE = {
    "numeric": "double precision",
    "date": "date",
    "text": "text",
    "categorical": "text",
    "boolean": "boolean",
}
SAMPLE = {
    "numeric": [1.0, 2.0, 0.5],
    "date": ["2023-01-05", "2023-02-15", "2023-03-31"],
    "text": ["alpha beta", "Gamma!", "delta, epsilon."],
    "categorical": ["a", "b", "a"],
    "boolean": [True, False, True],
}
# Not configurable as a bare transformations: entry.
NOT_STANDALONE = {"identity", "in_array"}


def _config(transformations: list[str], vtype: str) -> dict:
    return {
        "target": "series",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max", "min", "count"],
        "transformations": transformations,
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
                "variables": {"x": {"type": vtype}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": "series_id"},
                "child": {"entity": "events", "key": "series_id"},
            }
        ],
    }


def _featurizer(config: dict, tmp_path) -> Featurizer:
    import yaml

    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return Featurizer(str(path))


def _seed(conn, vtype: str) -> None:
    create_temp_table(
        conn,
        "series",
        [("series_id", "int"), ("size", "double precision")],
        [(1, 10.0), (2, 20.0)],
    )
    rows = [
        (month, 1, f"2024-{month:02d}-01", value)
        for month, value in enumerate(SAMPLE[vtype], start=1)
    ]
    create_temp_table(
        conn,
        "events",
        [
            ("event_id", "int"),
            ("series_id", "int"),
            ("ts", "date"),
            ("x", SQL_TYPE[vtype]),
        ],
        rows,
    )
    create_temp_table(conn, "as_of_dates", [("as_of_date", "date")], [("2024-06-01",)])


def _cases():
    for name in sorted(list_transformations()):
        if name in NOT_STANDALONE:
            continue
        types = getattr(get_transformers([name])[name], "input_types", None) or [
            "numeric"
        ]
        vtype = (
            "numeric"
            if "numeric" in types
            else next((t for t in types if t in SQL_TYPE), None)
        )
        if vtype is not None:
            yield pytest.param(name, vtype, id=name)


def _columns(conn, featurizer: Featurizer) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(featurizer.query)
        return [d.name for d in cur.description]


@pytest.mark.parametrize("name,vtype", list(_cases()))
def test_every_transformer_runs_selected_alone(pg_conn, tmp_path, name, vtype) -> None:
    """104 of 162 such runs raised UndefinedColumn on de03142."""
    _seed(pg_conn, vtype)
    featurizer = _featurizer(_config([name], vtype), tmp_path)
    columns = _columns(pg_conn, featurizer)
    # The result is what the manifest says it is: the helpers are not output.
    promised = [entry.column.strip('"') for entry in featurizer.feature_manifest]
    assert sorted(columns) == sorted(["as_of_date", "series_id", *promised])


def test_without_identity_the_raw_columns_are_not_output(pg_conn, tmp_path) -> None:
    _seed(pg_conn, "numeric")
    columns = _columns(pg_conn, _featurizer(_config(["abs"], "numeric"), tmp_path))
    assert "size" not in columns
    assert "MAX(events.x)" not in columns
    # What the planner had always planned for this config, now computable:
    assert "ABS(series.MAX(events.x))" in columns
    assert "ABS(series.MAX(events.ABS(events.x)))" in columns
    assert "ABS(series.size)" in columns


def test_values_match_the_same_config_with_identity(pg_conn, tmp_path) -> None:
    """Adding ``identity`` adds columns; it cannot change the ones both have."""
    _seed(pg_conn, "numeric")

    def row(transformations: list[str]) -> dict:
        featurizer = _featurizer(_config(transformations, "numeric"), tmp_path)
        with pg_conn.cursor() as cur:
            cur.execute(featurizer.query)
            names = [d.name for d in cur.description]
            return {
                r[names.index("series_id")]: dict(zip(names, r)) for r in cur.fetchall()
            }

    alone, with_identity = row(["abs", "lag_1"]), row(["identity", "abs", "lag_1"])
    for series_id, values in alone.items():
        for column, value in values.items():
            assert with_identity[series_id][column] == value, column
