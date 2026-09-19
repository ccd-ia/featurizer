# coding: utf-8

"""Aggregations quote the column they wrap (issue #29).

The aggregation-side twin of #18. #13 quoted the places a declared column name
is *projected*, #24 quoted the input of every transformer. An aggregation does
neither — it wraps its input in SQL, and it wrapped ``feature.name`` raw. A
declared child column named like an aggregate call rendered::

    sum( MEAN(games.goals) )   as "SUM(events.MEAN(games.goals))"

which PostgreSQL reads as ``sum()`` over the aggregate ``MEAN()`` over a column
of a table ``games`` that is not in the FROM clause. Every aggregation that
emits a feature had the fault.

The fix quotes the input at each SQL-emission site through ``_col`` (the wrapped
column), ``_tix`` (the entity's temporal index, which the ``index``-typed
aggregations wrap and every interval filter reads) and ``_latlon`` (the declared
lat / lon columns the spatial aggregations read). What moves and what does not,
pinned below:

- output column names and labels: unchanged (ADR-0007);
- the companion pre-aggregation CTE names (ADR-0010): unchanged — the family key
  is built from the raw name, never from the quoted reference;
- a declared variable's own ``definition``: unchanged;
- a *derived* feature's ``definition`` — which is its SQL — now carries the
  quoted input (``sum( age )`` becomes ``sum( "age" )``). That text is persisted
  in ``<stem>_manifest.definition``; no known consumer reads it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from featurizer import Featurizer
from featurizer.primitives.utils import get_aggregations, list_aggregations

WEIRD = "MEAN(games.goals)"
WEIRD_TS = "Event Date"
WEIRD_LATLON = ("Lat Deg", "Lon Deg")
INTERVAL = "P90D"

# Predicate-driven aggregations fire only on a variable that declares these.
PREDICATES = {"target": "b", "terminal": "b", "a": "a", "b": "b"}


def _config(
    tmp_path: Path,
    variable: str,
    aggregations: list[str],
    vtype: str = "numeric",
    *,
    temporal_ix: str = "ts",
    intervals: tuple[str, ...] = (INTERVAL,),
    latlon: tuple[str, str] | None = None,
) -> str:
    predicates = (
        f"        predicates: {json.dumps(PREDICATES)}\n"
        if vtype == "categorical"
        else ""
    )
    spatial_ix = (
        f"    spatial_ix: {{lat: {json.dumps(latlon[0])}, lon: {json.dumps(latlon[1])}}}\n"
        if latlon
        else ""
    )
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: series\n"
        "max_depth: 2\n"
        f"intervals: [{', '.join(intervals)}]\n"
        f"aggregations: [{', '.join(aggregations)}]\n"
        "transformations: [identity]\n"
        "entities:\n"
        "  - alias: series\n"
        "    id: series_id\n"
        "    table: series\n"
        "  - alias: events\n"
        "    id: event_id\n"
        "    table: events\n"
        f"    temporal_ix: {json.dumps(temporal_ix)}\n"
        f"{spatial_ix}"
        "    variables:\n"
        f"      {json.dumps(variable)}:\n"
        f"        type: {vtype}\n"
        f"{predicates}"
        "relationships:\n"
        "  - parent: {entity: series, key: series_id}\n"
        "    child: {entity: events, key: series_id}\n"
    )
    return str(path)


# ------------------------------------------------------------- rendering


def test_sum_wraps_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["sum"])).query
    assert f'sum( "{WEIRD}" )' in query
    assert f"sum( {WEIRD} )" not in query


def test_ordered_set_aggregations_order_by_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["median", "iqr"])).query
    assert f'within group(order by "{WEIRD}")' in query
    assert f"within group(order by {WEIRD})" not in query


def test_expression_aggregations_wrap_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["cv", "range", "skewness"])).query
    assert f'stddev("{WEIRD}")' in query
    assert f'max("{WEIRD}")' in query
    assert f'power("{WEIRD}",3)' in query


def test_the_set_based_prepass_reads_the_quoted_column(tmp_path) -> None:
    """ADR-0010: the pre-pass builds its own references to <child>_transform."""
    query = Featurizer(_config(tmp_path, WEIRD, ["gini", "theil"])).query
    assert f'events_transform."{WEIRD}" as val' in query
    assert f"events_transform.{WEIRD}" not in query


def test_a_categorical_prepass_groups_by_the_quoted_column(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, WEIRD, ["entropy"], "categorical")).query
    assert f'group by events_transform.series_id, events_transform."{WEIRD}"' in query


def test_the_temporal_index_is_quoted_where_an_aggregation_wraps_it(tmp_path) -> None:
    query = Featurizer(
        _config(tmp_path, "x", ["recency", "time_span"], temporal_ix=WEIRD_TS)
    ).query
    assert f'max("{WEIRD_TS}")' in query
    assert f"max({WEIRD_TS})" not in query


def test_spatial_aggregations_read_the_quoted_lat_lon(tmp_path) -> None:
    query = Featurizer(
        _config(tmp_path, "x", ["spatial_std"], latlon=WEIRD_LATLON, intervals=())
    ).query
    assert 'var_samp(sub."Lat Deg") + var_samp(sub."Lon Deg")' in query


def test_the_interval_filter_reads_the_quoted_temporal_index(tmp_path) -> None:
    query = Featurizer(_config(tmp_path, "x", ["sum"], temporal_ix=WEIRD_TS)).query
    assert f'@> "{WEIRD_TS}"::date' in query
    assert f"@> {WEIRD_TS}::date" not in query


def test_output_names_do_not_move(tmp_path) -> None:
    """ADR-0007: names are built by _build_name / _build_label, never by _col."""
    f = Featurizer(_config(tmp_path, "age", ["sum", "median", "gini"]))
    labels = {e.label for e in f.feature_manifest}
    assert {
        "SUM(events.age)",
        f"SUM(events.age|interval={INTERVAL})",
        "MEDIAN(events.age)",
        "GINI(events.age)",
    } <= labels
    columns = {e.column for e in f.feature_manifest}
    assert '"SUM(events.age)"' in columns or "SUM(events.age)" in columns


def test_companion_cte_names_do_not_move(tmp_path) -> None:
    """The ADR-0010 family key is the raw name; a quoted one would rename the CTE."""
    query = Featurizer(_config(tmp_path, "age", ["gini", "entropy"])).query
    assert "events_gini_age_all_preaggs_for_series as (" in query
    assert f"events_gini_age_{INTERVAL}_preaggs_for_series as (" in query


def test_a_declared_variables_own_definition_does_not_move(tmp_path) -> None:
    f = Featurizer(_config(tmp_path, "age", ["sum"]))
    _ = f.query
    (events,) = [e for e in f.entities if e.alias == "events"]
    (variable,) = [v for v in events.features if v.name == "age"]
    assert variable.definition == "age"


def test_a_derived_definition_carries_the_quoted_input(tmp_path) -> None:
    """The one persisted change, pinned so it cannot drift again unnoticed."""
    f = Featurizer(_config(tmp_path, "age", ["sum"], intervals=()))
    (entry,) = [e for e in f.feature_manifest if e.label == "SUM(events.age)"]
    assert " ".join(entry.definition.split()) == 'sum( "age" )'


# ------------------------------------------------------------- execution

SQL_TYPE = {"numeric": "double precision", "categorical": "text", "boolean": "boolean"}
# Four series; the values are positive so every domain-guarded aggregation
# (harmonic / geometric mean, theil) returns a number and not NULL.
ROWS = {
    "numeric": [1.0, 4.0, 9.0, 16.0, 2.5, 7.0],
    "categorical": ["a", "b", "a", "b", "b", "a"],
    "boolean": [True, False, True, True, False, True],
}
AS_OF = "2024-06-01"
# Inside P90D of the as-of date, so interval and whole-history features both
# see rows, and both drift windows (recent, baseline) are populated.
STAMPS = [
    "2023-12-20",
    "2024-01-15",
    "2024-02-10",
    "2024-03-20",
    "2024-04-15",
    "2024-05-20",
]

SPATIAL = {"bbox_area", "distance_travelled", "radius_of_gyration", "spatial_std"}


def _input_type(name: str) -> str:
    types = getattr(get_aggregations([name])[name], "input_types", None) or ["numeric"]
    return next(t for t in types if t in SQL_TYPE or t == "index")


def _column_cases():
    for name in sorted(list_aggregations()):
        vtype = _input_type(name)
        if vtype != "index":
            yield pytest.param(name, vtype, id=name)


def _index_cases():
    for name in sorted(list_aggregations()):
        if _input_type(name) == "index":
            yield pytest.param(name, id=name)


def _literal(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f"'{value}'"
    return repr(value)


def _execute(
    query: str,
    column: str,
    vtype: str,
    temporal_ix: str,
    latlon: tuple[str, str] = ("lat", "lon"),
) -> list[dict]:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")

    rows = []
    event_id = 0
    for series_id in (1, 2):
        for stamp, value in zip(STAMPS, ROWS[vtype]):
            event_id += 1
            rows.append(
                f"({event_id}, {series_id}, date '{stamp}', {_literal(value)}, "
                f"{19.0 + event_id / 10}, {-99.0 - event_id / 7})"
            )
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute("create temp table series (series_id int)")
        cur.execute("insert into series values (1), (2)")
        cur.execute(
            f'create temp table events (event_id int, series_id int, "{temporal_ix}" '
            f'date, "{column}" {SQL_TYPE[vtype]}, "{latlon[0]}" double precision, '
            f'"{latlon[1]}" double precision)'
        )
        cur.execute(f"insert into events values {', '.join(rows)}")
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.execute(f"insert into as_of_dates values (date '{AS_OF}')")
        cur.execute(query)
        names = [d.name for d in cur.description]
        out = [dict(zip(names, row)) for row in cur.fetchall()]
        conn.rollback()
    return out


def _values_by_label(
    featurizer: Featurizer, rows: list[dict], rename: tuple[str, str]
) -> dict:
    """``{(series_id, label): value}`` with the weird name folded to the plain one."""
    old, new = rename
    by_column = {e.column.strip('"'): e.label for e in featurizer.feature_manifest}
    out = {}
    for row in rows:
        for column, value in row.items():
            label = by_column.get(column)
            if label is not None:
                out[(row["series_id"], label.replace(old, new))] = value
    return out


def _parity(tmp_path, name, vtype, *, weird_column, weird_ts, spatial=False) -> None:
    plain_dir, weird_dir = tmp_path / "plain", tmp_path / "weird"
    plain_dir.mkdir()
    weird_dir.mkdir()
    plain_latlon = ("lat", "lon") if spatial else None
    weird_latlon = WEIRD_LATLON if spatial else None
    plain = Featurizer(_config(plain_dir, "x", [name], vtype, latlon=plain_latlon))
    weird = Featurizer(
        _config(
            weird_dir,
            weird_column,
            [name],
            vtype,
            temporal_ix=weird_ts,
            latlon=weird_latlon,
        )
    )
    token = f"{name.upper()}("
    assert [e for e in plain.feature_manifest if e.label.startswith(token)], (
        f"{name} emitted no feature, so the sweep would prove nothing about it"
    )
    expected = _values_by_label(
        plain, _execute(plain.query, "x", vtype, "ts"), ("x", "x")
    )
    fold = (weird_column, "x") if weird_column != "x" else (weird_ts, "ts")
    actual = _values_by_label(
        weird,
        _execute(
            weird.query, weird_column, vtype, weird_ts, weird_latlon or ("lat", "lon")
        ),
        fold,
    )
    assert actual == expected
    assert any(k[1].startswith(token) and v is not None for k, v in actual.items()), (
        f"{name} returned only NULLs, so value parity would prove nothing"
    )


@pytest.mark.integration
@pytest.mark.parametrize("name,vtype", list(_column_cases()))
def test_every_aggregation_executes_over_an_aggregate_shaped_column(
    tmp_path, name, vtype
) -> None:
    """Every one raised UndefinedTable before the fix. Now: value parity with ``x``."""
    _parity(tmp_path, name, vtype, weird_column=WEIRD, weird_ts="ts")


@pytest.mark.integration
@pytest.mark.parametrize("name", list(_index_cases()))
def test_every_index_aggregation_executes_over_a_non_identifier_temporal_index(
    tmp_path, name
) -> None:
    """The ``index``-typed aggregations wrap the temporal index, not a variable."""
    _parity(
        tmp_path,
        name,
        "numeric",
        weird_column="x",
        weird_ts=WEIRD_TS,
        spatial=name in SPATIAL,
    )


def test_the_sweeps_cover_the_whole_registry() -> None:
    covered = {p.values[0] for p in _column_cases()} | {
        p.values[0] for p in _index_cases()
    }
    assert covered == set(list_aggregations())
