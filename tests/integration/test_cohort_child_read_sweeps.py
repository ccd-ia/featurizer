"""Every primitive gives the dense value over a child the paired cohort narrowed.

The #10 follow-up cuts a child's read to the rows of the date's cohort. Whether
that is safe is a property of each primitive: a transformer's window must still
see every row of its partition, an aggregation must still see every row of its
group. ``population_level`` is the one declaration the planner reads; this file
is the net under it, and under the next primitive somebody registers.

Method: the same config and data, run dense and run paired, compared on the
pairs. Three child shapes, because the cut takes three forms:

- ``series_id``: the child's id is the join key, so a partition is a key group;
- ``event_id``: one row per id;
- ``track``: an id that CROSSES series (track 1 holds rows of series 1 and 2), so
  a window reads rows of an entity the cohort may not name. The cut has to keep
  whole partitions here, and a cut on the key alone gets ``lag_1`` wrong.
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
import yaml

from featurizer import Featurizer
from featurizer.primitives.utils import (
    get_transformers,
    list_aggregations,
    list_transformations,
)
from tests.test_aggregation_quoting import SPATIAL, _input_type
from tests.test_aggregation_quoting import _config as _aggregation_config

pytestmark = pytest.mark.integration

DATES = ["2024-04-01", "2024-06-01"]
# Different ids per date, and never the whole population on one date.
PAIRS = [("2024-04-01", 1), ("2024-04-01", 3), ("2024-06-01", 2), ("2024-06-01", 3)]
SERIES = (1, 2, 3, 4)
STAMPS = ["2024-01-05", "2024-02-10", "2024-03-15", "2024-04-20", "2024-05-25"]
SQL_TYPE = {
    "numeric": "double precision",
    "date": "date",
    "text": "text",
    "categorical": "text",
    "boolean": "boolean",
}
# Not configurable as a bare ``transformations:`` entry.
NOT_STANDALONE = {"identity", "in_array"}


def _value(vtype: str, series_id: int, position: int):
    """Distinct per series and per row, so a row read from the wrong entity, or a
    missing one, shows up in every statistic."""
    n = series_id * 10 + position
    return {
        # Small on purpose: the transformer list applies at every level, so
        # ``exp`` runs on ``max(exp(x))`` as well, and exp(exp(4)) overflows.
        "numeric": round(0.2 + (n * 1.37 % 17) / 7, 3),
        "date": f"2023-{(n % 12) + 1:02d}-{(n % 27) + 1:02d}",
        "text": f"word{n} tail{series_id}, x{position}.",
        "categorical": "ab"[(series_id + position) % 2],
        "boolean": (series_id + position) % 3 == 0,
    }[vtype]


def _literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _run(query: str, vtype: str, *, paired: bool, latlon: bool = False) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    rows, event_id = [], 0
    for position, stamp in enumerate(STAMPS):
        for series_id in SERIES:
            event_id += 1
            track = 1 if series_id in (1, 2) else 2
            rows.append(
                f"({event_id}, {series_id}, {track}, date '{stamp}', "
                f"{_literal(_value(vtype, series_id, position))}, "
                f"{19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1), (2), (3), (4)")
        cur.execute(
            "create temp table events (event_id int, series_id int, track int, "
            f"ts date, x {SQL_TYPE[vtype]}, lat double precision, "
            "lon double precision)"
        )
        cur.execute(f"insert into events values {', '.join(rows)}")
        if paired:
            cur.execute(
                "create temp table as_of_dates (as_of_date date, cohort_id int)"
            )
            cur.executemany("insert into as_of_dates values (%s, %s)", PAIRS)
        else:
            cur.execute("create temp table as_of_dates (as_of_date date)")
            cur.executemany(
                "insert into as_of_dates values (%s)", [(d,) for d in DATES]
            )
        cur.execute(query)
        names = [d.name for d in cur.description]
        out = {}
        for row in cur.fetchall():
            record = dict(zip(names, row))
            key = (str(record.pop("as_of_date")), record.pop("series_id"))
            out[key] = record
        conn.rollback()
    return out


def _same(a, b) -> bool:
    """Equal, up to the last bits of a float.

    A narrowed read feeds the same rows to an aggregate in another order, and a
    float sum depends on the order: ``cosinor_amplitude_weekly`` on data without
    a weekly cycle is 2.309e-11 one way and 2.308e-11 the other. The tolerance
    is nine orders of magnitude under the sample values.
    """
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float, Decimal)) and isinstance(b, (int, float, Decimal)):
        return float(a) == pytest.approx(float(b), rel=1e-9, abs=1e-9)
    return a == b


def _parity(tmp_path, config: dict, vtype: str, *, latlon: bool = False) -> None:
    dense_path = tmp_path / "dense.yaml"
    dense_path.write_text(yaml.safe_dump(config, sort_keys=False))
    paired_path = tmp_path / "paired.yaml"
    paired_path.write_text(
        yaml.safe_dump(
            {**config, "as_of_dates": {"id_column": "cohort_id"}}, sort_keys=False
        )
    )
    dense_f = Featurizer(str(dense_path), validate=False)
    paired_f = Featurizer(str(paired_path), validate=False)
    assert dense_f.feature_manifest, "no feature to check"

    narrowed = "_cohort" in paired_f.query.split("series_synth as (")[0]
    population = any(
        getattr(t, "population_level", False)
        for t in get_transformers(config["transformations"]).values()
    )
    # Every primitive is narrowed except the ones that say they read the
    # population; a sweep that narrowed nothing would prove nothing.
    assert narrowed is not population

    dense = _run(dense_f.query, vtype, paired=False, latlon=latlon)
    paired = _run(paired_f.query, vtype, paired=True, latlon=latlon)

    assert set(paired) == set(PAIRS)
    for pair in PAIRS:
        assert set(paired[pair]) == set(dense[pair])
        for column, value in paired[pair].items():
            assert _same(value, dense[pair][column]), (pair, column)


def _transformer_config(name: str, vtype: str, child_id: str) -> dict:
    return {
        "target": "series",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max", "min", "nunique", "count"],
        "transformations": ["identity", name],
        "entities": [
            {"alias": "series", "table": "series", "id": "series_id"},
            {
                "alias": "events",
                "table": "events",
                "id": child_id,
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


def _sweep_cases():
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
        if vtype is None:
            continue
        for child_id in ("series_id", "event_id", "track"):
            yield pytest.param(name, vtype, child_id, id=f"{name}-by-{child_id}")


@pytest.mark.parametrize("name,vtype,child_id", list(_sweep_cases()))
def test_every_transformer_on_a_narrowed_child_gives_the_dense_value(
    tmp_path, name, vtype, child_id
) -> None:
    _parity(tmp_path, _transformer_config(name, vtype, child_id), vtype)


@pytest.mark.parametrize("name", sorted(list_aggregations()))
def test_every_aggregation_over_a_narrowed_child_gives_the_dense_value(
    tmp_path, name
) -> None:
    vtype = _input_type(name)
    column_type = "numeric" if vtype == "index" else vtype
    latlon = ("lat", "lon") if name in SPATIAL else None
    path = _aggregation_config(tmp_path, "x", [name], column_type, latlon=latlon)
    with open(path) as handle:
        config = yaml.safe_load(handle)
    _parity(tmp_path, config, column_type, latlon=bool(latlon))
