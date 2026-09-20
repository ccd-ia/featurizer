# coding: utf-8

"""Every declared identifier is quoted where SQL reads it, whatever its role (#46).

#13, #24 and #29 each quoted a declared *variable* at the site its reporter hit:
the synth's projection, a transformer's input, an aggregation's input. A
declared name reaches SQL in other roles too — an entity id in a window's
``partition by`` and a shard's ``using (…)``, a temporal index in a window's
``order by``, a relationship key in a ``group by`` and a join — and those were
interpolated raw. Measured on c2c21a7: variables 3 of 3 shapes run, every other
role 4 of 4 fail with ``syntax error at or near …``.

This file sweeps identifier ROLE x name shape, on all three render paths (the
single query, column groups, TEMP tables), and asserts value parity against the
same data under plain names, so the class closes and not the next site.

What must not move (ADR-0015): ``GroupedQueries.key_columns`` and the other
key metadata keep their form. Only the SQL text is quoted.
"""

from __future__ import annotations

import datetime
import os
from collections import OrderedDict
from pathlib import Path

import pytest
import yaml

from featurizer import Featurizer
from featurizer.sharding import ColumnGroupSharder

# role -> the plain name the scenario uses for it
PLAIN = {
    "target id": "series_id",
    "child id": "event_id",
    "child foreign key": "series_ref",
    "child temporal index": "ts",
    "carried index variable": "batch_id",
    "grandchild id": "item_id",
    "grandchild foreign key": "event_ref",
    "grandchild temporal index": "item_ts",
}
SHAPES = {
    "a space": "Odd Name",
    "a reserved word": "order",
    "aggregate-shaped": "MEAN(games.goals)",
}
AS_OF = [datetime.date(2024, 6, 1), datetime.date(2024, 8, 1)]


def _config(names: dict) -> dict:
    return {
        "target": "series",
        "max_depth": 3,
        "intervals": ["P90D"],
        # Small on purpose (57 features): a window (``partition by`` the id,
        # ``order by`` the temporal index), a rolling percentile (the by-name
        # re-scan of the synth), ``count`` over the ids and keys, ``recency``
        # over the temporal index, and one interval filter.
        "aggregations": ["max", "count", "recency"],
        "transformations": ["identity", "lag_1", "rolling_median_7"],
        "entities": [
            {
                "alias": "series",
                "table": "series",
                "id": names["target id"],
                "variables": {"size": {"type": "numeric"}},
            },
            {
                "alias": "events",
                "table": "events",
                "id": names["child id"],
                "temporal_ix": names["child temporal index"],
                "variables": {
                    "x": {"type": "numeric"},
                    names["carried index variable"]: {"type": "index"},
                },
            },
            {
                "alias": "items",
                "table": "items",
                "id": names["grandchild id"],
                "temporal_ix": names["grandchild temporal index"],
                "variables": {"price": {"type": "numeric"}},
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": names["target id"]},
                "child": {"entity": "events", "key": names["child foreign key"]},
            },
            {
                "parent": {"entity": "events", "key": names["child id"]},
                "child": {"entity": "items", "key": names["grandchild foreign key"]},
            },
        ],
    }


def _write(config: dict, tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _seed(cur, names: dict) -> None:
    n = names
    cur.execute(
        f"create temp table series ({_q(n['target id'])} int, size double precision)"
    )
    cur.execute("insert into series values (1, 10.0), (2, 20.0), (3, 30.0)")
    cur.execute(
        f"create temp table events ({_q(n['child id'])} int, "
        f"{_q(n['child foreign key'])} int, {_q(n['child temporal index'])} date, "
        f"x double precision, {_q(n['carried index variable'])} int)"
    )
    cur.execute(
        "insert into events values "
        "(10, 1, '2024-03-10', 1.5, 7), (11, 1, '2024-04-20', 4.0, 7), "
        "(12, 1, '2024-05-30', 2.5, 8), (13, 1, '2024-07-15', 9.0, 8), "
        "(14, 2, '2024-05-01', 6.0, 9), (15, 1, '2024-09-30', 99.0, 9)"
    )
    cur.execute(
        f"create temp table items ({_q(n['grandchild id'])} int, "
        f"{_q(n['grandchild foreign key'])} int, "
        f"{_q(n['grandchild temporal index'])} date, price double precision)"
    )
    cur.execute(
        "insert into items values "
        "(100, 10, '2024-03-10', 3.0), (101, 10, '2024-03-11', 5.0), "
        "(102, 11, '2024-04-20', 2.0), (103, 13, '2024-07-15', 8.0), "
        "(104, 14, '2024-05-01', 1.0), (105, 15, '2024-09-30', 50.0)"
    )
    cur.execute("create temp table as_of_dates (as_of_date date)")
    cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])


def _rows(cur, sql: str) -> list[dict]:
    cur.execute(sql)
    columns = [d.name for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _single(cur, config_path: str) -> list[dict]:
    return _rows(cur, Featurizer(config_path).query)


def _column_groups(cur, config_path: str, id_column: str) -> list[dict]:
    built = ColumnGroupSharder(
        Featurizer(config_path)._plan, max_columns_per_group=25
    ).build()
    assert len(built.queries) > 1, "expected the small budget to force >1 group"
    return _rejoin(cur, built.queries, id_column)


def _temp_tables(cur, config_path: str, id_column: str) -> list[dict]:
    grouped = Featurizer(config_path, materialize_threshold=1)._grouped()
    assert grouped.materialization is not None, "expected a TEMP-table preamble"
    for ddl in grouped.materialization.ddl:
        cur.execute(ddl)
    return _rejoin(cur, grouped.queries, id_column)


def _rejoin(cur, queries: "OrderedDict[str, str]", id_column: str) -> list[dict]:
    joined: dict = {}
    for sql in queries.values():
        for row in _rows(cur, sql):
            joined.setdefault((row["as_of_date"], row[id_column]), {}).update(row)
    return list(joined.values())


PATHS = {
    "single query": lambda cur, path, id_column: _single(cur, path),
    "column groups": _column_groups,
    "TEMP tables": _temp_tables,
}


def _by_label(config_path: str, rows: list[dict], names: dict) -> dict:
    """``{(as_of_date, target id, label): value}`` with every odd name folded
    back to its plain one, so two runs are comparable cell by cell."""
    labels = {
        entry.column.strip('"'): entry.label
        for entry in Featurizer(config_path).feature_manifest
    }
    id_column = names["target id"]
    out = {}
    for row in rows:
        for column, value in row.items():
            label = labels.get(column)
            if label is None:
                continue
            for role, plain in PLAIN.items():
                if names[role] != plain:
                    label = label.replace(names[role], plain)
            out[(row["as_of_date"], row[id_column], label)] = value
    return out


def _matrix(tmp_path: Path, names: dict, path_name: str) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    config_path = _write(_config(names), tmp_path)
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        _seed(cur, names)
        rows = PATHS[path_name](cur, config_path, names["target id"])
        conn.rollback()
    return _by_label(config_path, rows, names)


_PLAIN_MATRIX: dict = {}


def _plain_matrix(tmp_path_factory, path_name: str) -> dict:
    if path_name not in _PLAIN_MATRIX:
        tmp = tmp_path_factory.mktemp("plain")
        _PLAIN_MATRIX[path_name] = _matrix(tmp, dict(PLAIN), path_name)
    return _PLAIN_MATRIX[path_name]


def _cases():
    """Every shape on the single query; one shape on the two slower paths.

    A shape decides whether a name parses; a path decides whether the site
    that emits it was reached. The two are independent, so the slower paths
    need only one shape that cannot be written bare to show they quote.
    """
    for role in PLAIN:
        for shape, odd in SHAPES.items():
            for path_name in PATHS:
                if path_name != "single query" and shape != "a space":
                    continue
                yield pytest.param(
                    role, odd, path_name, id=f"{role}={shape}-{path_name}"
                )


@pytest.mark.integration
@pytest.mark.parametrize("role,odd,path_name", list(_cases()))
def test_a_non_bare_identifier_runs_and_changes_no_value(
    tmp_path, tmp_path_factory, role, odd, path_name
) -> None:
    expected = _plain_matrix(tmp_path_factory, path_name)
    assert expected, "the plain scenario produced no features"
    names = {**PLAIN, role: odd}
    assert _matrix(tmp_path, names, path_name) == expected


@pytest.mark.integration
def test_the_three_paths_agree_on_the_plain_scenario(tmp_path_factory) -> None:
    """The oracle itself: if the paths disagreed under plain names, parity
    against each path's own baseline would prove nothing about the others."""
    single = _plain_matrix(tmp_path_factory, "single query")
    assert _plain_matrix(tmp_path_factory, "column groups") == single
    assert _plain_matrix(tmp_path_factory, "TEMP tables") == single


# --------------------------------------------------------------------------- #
# The planner passes and the as-of lookup read declared columns of their own:
# an edge table's source / target / timestamp, the neighbour-state entity's id,
# temporal index and measures, the peer-group categorical and its measures,
# lat / lon on both sides of a spatial relationship, and the keys, temporal
# indexes and transferred variable of an as-of lookup. Their CTEs are emitted
# verbatim on every render path, so the single query is enough here.
# --------------------------------------------------------------------------- #

PASS_PLAIN = {
    "target id": "series_id",
    "target temporal index": "opened",
    "peer-group categorical": "grp",
    "peer-group measure": "size",
    "left latitude": "lat",
    "left longitude": "lon",
    "right id (spatial)": "site_id",
    "right latitude": "site_lat",
    "right longitude": "site_lon",
    "right temporal index (spatial)": "site_ts",
    "edge source": "src",
    "edge target": "dst",
    "edge timestamp": "linked",
    "neighbour id": "state_id",
    "neighbour temporal index": "state_ts",
    "neighbour measure": "risk",
    "neighbour share": "flagged",
    "child foreign key": "series_ref",
    "child temporal index": "ts",
    "lookup parent key": "zone_id",
    "lookup child key": "zone_ref",
    "lookup temporal index": "rate_ts",
    "looked-up variable": "level",
}


def _pass_config(n: dict) -> dict:
    return {
        "target": "series",
        "max_depth": 3,
        # No interval and ``max`` only. A child that receives an as-of lookup
        # and is then aggregated fails on PLAIN names under an interval or an
        # index-typed aggregation (issue #48), which would hide what this sweep
        # measures. The interval filter and ``count`` are swept above.
        "intervals": [],
        "aggregations": ["max"],
        "transformations": ["identity"],
        "entities": [
            {
                "alias": "series",
                "table": "series",
                "id": n["target id"],
                "temporal_ix": n["target temporal index"],
                "spatial_ix": {"lat": n["left latitude"], "lon": n["left longitude"]},
                "variables": {
                    n["peer-group categorical"]: {"type": "categorical"},
                    n["peer-group measure"]: {"type": "numeric"},
                },
                "peer_groups": [
                    {
                        "by": n["peer-group categorical"],
                        "measures": [n["peer-group measure"]],
                    }
                ],
            },
            {
                "alias": "events",
                "table": "events",
                "id": "event_id",
                "temporal_ix": n["child temporal index"],
                "variables": {"x": {"type": "numeric"}},
            },
            {
                "alias": "rates",
                "table": "rates",
                "id": "rate_id",
                "temporal_ix": n["lookup temporal index"],
                "variables": {n["looked-up variable"]: {"type": "numeric"}},
            },
            {
                "alias": "sites",
                "table": "sites",
                "id": n["right id (spatial)"],
                "temporal_ix": n["right temporal index (spatial)"],
                "spatial_ix": {
                    "lat": n["right latitude"],
                    "lon": n["right longitude"],
                },
            },
            {
                "alias": "states",
                "table": "states",
                "id": n["neighbour id"],
                "temporal_ix": n["neighbour temporal index"],
                "variables": {
                    n["neighbour measure"]: {"type": "numeric"},
                    n["neighbour share"]: {"type": "boolean"},
                },
            },
        ],
        "relationships": [
            {
                "parent": {"entity": "series", "key": n["target id"]},
                "child": {"entity": "events", "key": n["child foreign key"]},
            },
            {
                "parent": {"entity": "rates", "key": n["lookup parent key"]},
                "child": {"entity": "events", "key": n["lookup child key"]},
                "temporal": {"mode": "as_of"},
            },
        ],
        "spatial_relationships": [
            {
                "name": "near_sites",
                "left": "series",
                "right": "sites",
                "within_m": 5000,
                "bandwidth_m": 1000,
            }
        ],
        "graph_relationships": [
            {
                "name": "links",
                "left": "series",
                "right": "states",
                "edges": {
                    "table": "links",
                    "source": n["edge source"],
                    "target": n["edge target"],
                    "timestamp": n["edge timestamp"],
                },
                "directed": True,
            }
        ],
    }


def _pass_seed(cur, n: dict) -> None:
    cur.execute(
        f"create temp table series ({_q(n['target id'])} int, "
        f"{_q(n['target temporal index'])} date, {_q(n['left latitude'])} double "
        f"precision, {_q(n['left longitude'])} double precision, "
        f"{_q(n['peer-group categorical'])} text, "
        f"{_q(n['peer-group measure'])} double precision)"
    )
    cur.execute(
        "insert into series values "
        "(1, '2024-01-01', 19.40, -99.10, 'a', 10.0), "
        "(2, '2024-01-05', 19.41, -99.11, 'a', 20.0), "
        "(3, '2024-02-01', 19.42, -99.12, 'a', 40.0), "
        "(4, '2024-07-01', 19.43, -99.13, 'b', 80.0)"
    )
    cur.execute(
        f"create temp table events (event_id int, {_q(n['child foreign key'])} int, "
        f"{_q(n['lookup child key'])} int, {_q(n['child temporal index'])} date, "
        "x double precision)"
    )
    cur.execute(
        "insert into events values "
        "(10, 1, 7, '2024-03-10', 1.5), (11, 1, 7, '2024-05-20', 4.0), "
        "(12, 2, 8, '2024-05-01', 6.0), (13, 1, 7, '2024-09-30', 99.0)"
    )
    cur.execute(
        f"create temp table rates (rate_id int, {_q(n['lookup parent key'])} int, "
        f"{_q(n['lookup temporal index'])} date, "
        f"{_q(n['looked-up variable'])} double precision)"
    )
    cur.execute(
        "insert into rates values (1, 7, '2024-01-01', 0.5), "
        "(2, 7, '2024-04-01', 0.7), (3, 8, '2024-02-01', 0.9), "
        "(4, 7, '2024-09-01', 9.9)"
    )
    cur.execute(
        f"create temp table sites ({_q(n['right id (spatial)'])} int, "
        f"{_q(n['right temporal index (spatial)'])} date, "
        f"{_q(n['right latitude'])} double precision, "
        f"{_q(n['right longitude'])} double precision)"
    )
    cur.execute(
        "insert into sites values (1, '2024-01-01', 19.401, -99.101), "
        "(2, '2024-03-01', 19.412, -99.109), (3, '2024-09-01', 19.400, -99.100)"
    )
    cur.execute(
        f"create temp table states ({_q(n['neighbour id'])} int, "
        f"{_q(n['neighbour temporal index'])} date, "
        f"{_q(n['neighbour measure'])} double precision, "
        f"{_q(n['neighbour share'])} boolean)"
    )
    cur.execute(
        "insert into states values (1, '2024-01-01', 0.1, true), "
        "(2, '2024-01-01', 0.4, false), (3, '2024-01-01', 0.8, true)"
    )
    cur.execute(
        f"create temp table links ({_q(n['edge source'])} int, "
        f"{_q(n['edge target'])} int, {_q(n['edge timestamp'])} date)"
    )
    cur.execute(
        "insert into links values (1, 2, '2024-02-01'), (1, 3, '2024-05-01'), "
        "(2, 3, '2024-03-01'), (1, 2, '2024-09-15')"
    )
    cur.execute("create temp table as_of_dates (as_of_date date)")
    cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])


def _pass_matrix(tmp_path: Path, names: dict) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    config_path = _write(_pass_config(names), tmp_path)
    featurizer = Featurizer(config_path)
    labels = {e.column.strip('"'): e.label for e in featurizer.feature_manifest}
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        _pass_seed(cur, names)
        rows = _rows(cur, featurizer.query)
        conn.rollback()
    out = {}
    for row in rows:
        for column, value in row.items():
            label = labels.get(column)
            if label is None:
                continue
            for role, plain in PASS_PLAIN.items():
                if names[role] != plain:
                    label = label.replace(names[role], plain)
            out[(row["as_of_date"], row[names["target id"]], label)] = value
    return out


_PASS_PLAIN_MATRIX: dict = {}


@pytest.mark.integration
@pytest.mark.parametrize("role", list(PASS_PLAIN))
def test_a_planner_pass_reads_a_non_bare_identifier(
    tmp_path, tmp_path_factory, role
) -> None:
    if not _PASS_PLAIN_MATRIX:
        _PASS_PLAIN_MATRIX.update(
            _pass_matrix(tmp_path_factory.mktemp("pass-plain"), dict(PASS_PLAIN))
        )
    # Every family the scenario configures produced something to compare.
    labels = {label for _, _, label in _PASS_PLAIN_MATRIX}
    for family in ("PEER_", "COLOCATION_COUNT", "DEGREE", "NEIGHBOUR_MEAN", ".level"):
        assert any(family in label for label in labels), (family, sorted(labels)[:8])
    names = {**PASS_PLAIN, role: "Odd Name"}
    assert _pass_matrix(tmp_path, names) == _PASS_PLAIN_MATRIX


# --------------------------------------------------------------------------- #
# The edge-entity graph family (``entities[].edge``): degree, reciprocity and
# the recursive neighbour families read the edge table's columns directly.
# --------------------------------------------------------------------------- #

EDGE_PLAIN = {
    "node id": "user_id",
    "edge source": "follower_id",
    "edge target": "followee_id",
    "edge weight": "strength",
    "edge timestamp": "created_at",
}


def _edge_config(n: dict) -> dict:
    return {
        "target": "users",
        "max_depth": 1,
        "intervals": [],
        "aggregations": ["mean"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "users", "table": "users", "id": n["node id"]},
            {
                "alias": "follows",
                "table": "follows",
                "edge": {
                    "node": "users",
                    "source": n["edge source"],
                    "target": n["edge target"],
                    "weight": n["edge weight"],
                    "timestamp": n["edge timestamp"],
                    "features": [
                        "degree",
                        "reciprocity",
                        "k_hop_2",
                        "clustering",
                        "common_neighbours",
                        "jaccard",
                        "adamic_adar",
                    ],
                },
            },
        ],
    }


def _edge_matrix(tmp_path: Path, names: dict) -> dict:
    import psycopg

    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("No PostgreSQL configured: set DATABASE_URL")
    featurizer = Featurizer(_write(_edge_config(names), tmp_path))
    with psycopg.connect(url) as conn, conn.cursor() as cur:
        cur.execute(f"create temp table users ({_q(names['node id'])} int)")
        cur.execute("insert into users values (1), (2), (3), (4)")
        cur.execute(
            f"create temp table follows ({_q(names['edge source'])} int, "
            f"{_q(names['edge target'])} int, {_q(names['edge weight'])} double "
            f"precision, {_q(names['edge timestamp'])} date)"
        )
        cur.execute(
            "insert into follows values (1, 2, 0.5, '2024-02-01'), "
            "(2, 1, 1.5, '2024-03-01'), (2, 3, 2.0, '2024-04-01'), "
            "(1, 3, 1.0, '2024-07-01'), (3, 4, 4.0, '2024-09-15')"
        )
        cur.execute("create temp table as_of_dates (as_of_date date)")
        cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in AS_OF])
        rows = _rows(cur, featurizer.query)
        conn.rollback()
    id_column = names["node id"]
    return {
        (row["as_of_date"], row[id_column], column): value
        for row in rows
        for column, value in row.items()
        if column not in ("as_of_date", id_column)
    }


_EDGE_PLAIN_MATRIX: dict = {}


@pytest.mark.integration
@pytest.mark.parametrize("role", list(EDGE_PLAIN))
def test_the_edge_entity_family_reads_a_non_bare_identifier(
    tmp_path, tmp_path_factory, role
) -> None:
    if not _EDGE_PLAIN_MATRIX:
        _EDGE_PLAIN_MATRIX.update(
            _edge_matrix(tmp_path_factory.mktemp("edge-plain"), dict(EDGE_PLAIN))
        )
    columns = {column for _, _, column in _EDGE_PLAIN_MATRIX}
    assert any(c.startswith("WEIGHTED_OUT_DEGREE") for c in columns), sorted(columns)
    assert any(c.startswith("RECIPROCITY") for c in columns), sorted(columns)
    names = {**EDGE_PLAIN, role: "Odd Name"}
    assert _edge_matrix(tmp_path, names) == _EDGE_PLAIN_MATRIX


# --------------------------------------------------------------------------- #
# What does NOT move. ADR-0015 freezes ``GroupedQueries.key_columns`` and the
# group tables' leading columns, so the key METADATA keeps the declared names,
# bare; only the SQL beside it is delimited (``ShardableCTE.key_projections``).
# --------------------------------------------------------------------------- #


def test_key_metadata_keeps_the_declared_names(tmp_path) -> None:
    names = {**PLAIN, "target id": "Odd Name", "child id": "Event Id"}
    featurizer = Featurizer(_write(_config(names), tmp_path), materialize_threshold=1)
    assert featurizer._grouped().key_columns == ["as_of_date", "Odd Name"]

    plan = featurizer._plan
    target_transform = plan.cte_specs["series_transform"]
    assert target_transform.key_columns[0] == "Odd Name"
    assert target_transform.sql_keys[0] == '"Odd Name"'
    assert plan.materialization_keys["events_transform"].join_key == "Event Id"
    agg = plan.cte_specs["events_aggs_for_series"]
    assert agg.key_columns == ["events_transform.series_ref"]
    assert agg.sql_keys == ['events_transform."series_ref"']


def test_key_metadata_of_a_plain_config_is_what_it_was(tmp_path) -> None:
    featurizer = Featurizer(_write(_config(dict(PLAIN)), tmp_path))
    assert featurizer._grouped().key_columns == ["as_of_date", "series_id"]
    synth = featurizer._plan.cte_specs["series_synth"]
    assert synth.key_columns == ["series.series_id"]
