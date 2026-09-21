"""A config that validates runs (matrix cells 16 and 17).

Validation and execution were tested apart. ``test.yml`` validates every
shipped example config, and only ``just examples`` — run by hand — executes
them. Generated shapes were not executed at all, and three issues in two weeks
were a config that validated and produced SQL PostgreSQL would not run, or did
not mean what it said (#7, #9, #38).

Part one executes each shipped example the way its README says to. Part two
crosses one small fixture with the shapes a config can take and asserts the
one-line property: ``is_valid`` implies the rendered query executes.

Shapes that have their own sweep are not repeated here: a transformer list
without ``identity`` (tests/integration/test_transformer_selection.py) and
declared names that are not bare identifiers (tests/test_declared_name_rule.py,
tests/test_identifier_roles_quoting.py).
"""

from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from featurizer.validation import ConfigValidator

from ._harness import create_temp_table

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parent.parent.parent
EXAMPLES = sorted(p for p in (REPO / "examples").iterdir() if p.name[:2].isdigit())


# ------------------------------------------------------------- shipped examples


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_a_shipped_example_runs_end_to_end(example: Path) -> None:
    """``create_data.py`` then ``run_example.py --execute``, as ``just example``
    does. Each example owns a schema (``example_NN``) that its seeding script
    drops and recreates, so this is repeatable on the ephemeral database."""
    if not (os.environ.get("DATABASE_URL") or os.environ.get("PGDATABASE")):
        pytest.skip("No PostgreSQL configured: set DATABASE_URL or PG* env vars")
    for script, args in (("create_data.py", []), ("run_example.py", ["--execute"])):
        result = subprocess.run(
            [sys.executable, str(example / script), *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert result.returncode == 0, (
            f"{example.name}/{script} exited {result.returncode}\n"
            f"{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
        )


# ------------------------------------------------------------- generated shapes

BASE = {
    "target": "series",
    "max_depth": 3,
    "intervals": ["P90D"],
    "aggregations": ["max", "count", "recency"],
    "transformations": ["identity", "lag_1"],
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
            "variables": {"x": {"type": "numeric"}, "kind": {"type": "categorical"}},
        },
        {
            "alias": "items",
            "table": "items",
            "id": "item_id",
            "temporal_ix": "item_ts",
            "variables": {"price": {"type": "numeric"}},
        },
        {
            "alias": "rates",
            "table": "rates",
            "id": "rate_id",
            "temporal_ix": "rate_ts",
            "variables": {"level": {"type": "numeric"}},
        },
    ],
    "relationships": [
        {
            "parent": {"entity": "series", "key": "series_id"},
            "child": {"entity": "events", "key": "series_id"},
        },
        {
            "parent": {"entity": "events", "key": "event_id"},
            "child": {"entity": "items", "key": "event_id"},
        },
    ],
}


def _shape(**changes) -> dict:
    config = copy.deepcopy(BASE)
    config.update(changes)
    return config


def _without_temporal_index(alias: str) -> dict:
    config = copy.deepcopy(BASE)
    for entity in config["entities"]:
        if entity["alias"] == alias:
            del entity["temporal_ix"]
    return config


def _with_lookup(*, on: str, child_timestamp: str | None = None, **changes) -> dict:
    """``rates`` looked up as-of. ``on`` is the entity that receives it."""
    config = _shape(**changes)
    temporal = {"mode": "as_of"}
    if child_timestamp:
        temporal["child_timestamp"] = child_timestamp
    config["relationships"].append(
        {
            "parent": {"entity": "rates", "key": "zone_id"},
            "child": {"entity": on, "key": "zone_id"},
            "temporal": temporal,
        }
    )
    return config


def _target_is_the_child() -> dict:
    """``events`` as the target: it has a parent (a direct pull) and a child."""
    config = _shape(target="events")
    return config


SHAPES = [
    pytest.param(_shape(), id="the-base-shape"),
    pytest.param(_shape(max_depth=1), id="depth-1"),
    pytest.param(_shape(max_depth=2), id="depth-2"),
    pytest.param(_shape(intervals=[]), id="no-intervals"),
    pytest.param(_shape(intervals=["P30D", "P1Y"]), id="two-intervals"),
    pytest.param(_shape(aggregations=[]), id="aggregations-empty"),
    pytest.param(_shape(transformations=[]), id="transformations-empty"),
    # Depth 2 on purpose: the curated defaults over three levels are some
    # 90,000 columns in 104 groups, minutes of work that proves nothing more.
    pytest.param(
        {
            k: v
            for k, v in _shape(max_depth=2).items()
            if k not in ("aggregations", "transformations")
        },
        id="curated-defaults",
    ),
    pytest.param(_without_temporal_index("items"), id="leaf-without-temporal-index"),
    pytest.param(
        _shape(intervals=[])
        | {"entities": _without_temporal_index("events")["entities"]},
        id="child-without-temporal-index",
    ),
    pytest.param(_shape(as_of_boundary="exclusive"), id="exclusive-boundary"),
    pytest.param(_target_is_the_child(), id="target-with-a-parent-and-a-child"),
    pytest.param(
        _with_lookup(on="events", intervals=[], aggregations=["max"]),
        id="lookup-on-a-child-whole-history-max",
    ),
    # Issue #48: these three did not run. The lookup's index columns were
    # listed among the child's features and never transferred (``count``), the
    # interval filter read the SOURCE's temporal index, and a child_timestamp
    # the lookup entity does not declare was not carried.
    pytest.param(_with_lookup(on="events"), id="lookup-on-a-child-interval-and-count"),
    pytest.param(
        {
            k: v
            for k, v in _with_lookup(on="events", max_depth=3).items()
            if k != "aggregations"
        },
        id="lookup-on-a-child-curated-default-aggregations",
    ),
    pytest.param(
        _with_lookup(on="events", child_timestamp="published"),
        id="lookup-with-an-undeclared-child_timestamp",
    ),
    pytest.param(
        _with_lookup(on="events", target="events"),
        id="lookup-on-the-target",
    ),
]


def _seed(conn) -> None:
    create_temp_table(
        conn,
        "series",
        [("series_id", "int"), ("size", "double precision")],
        [(1, 10.0), (2, 20.0)],
    )
    create_temp_table(
        conn,
        "events",
        [
            ("event_id", "int"),
            ("series_id", "int"),
            ("zone_id", "int"),
            ("ts", "date"),
            ("x", "double precision"),
            ("kind", "text"),
        ],
        [
            (10, 1, 7, "2024-03-10", 1.5, "a"),
            (11, 1, 7, "2024-05-20", 4.0, "b"),
            (12, 2, 8, "2024-05-01", 6.0, "a"),
        ],
    )
    create_temp_table(
        conn,
        "items",
        [
            ("item_id", "int"),
            ("event_id", "int"),
            ("item_ts", "date"),
            ("price", "double precision"),
        ],
        [(100, 10, "2024-03-10", 3.0), (101, 11, "2024-05-20", 2.0)],
    )
    create_temp_table(
        conn,
        "rates",
        [
            ("rate_id", "int"),
            ("zone_id", "int"),
            ("rate_ts", "date"),
            ("published", "date"),
            ("level", "double precision"),
        ],
        [
            (1, 7, "2024-01-01", "2024-01-03", 0.5),
            (2, 8, "2024-02-01", "2024-02-03", 0.9),
        ],
    )
    create_temp_table(
        conn,
        "as_of_dates",
        [("as_of_date", "date")],
        [("2024-06-01",), ("2024-08-01",)],
    )


def _run(conn, config: dict) -> int:
    """Execute whatever the config renders to — one query, or the column groups
    (and their TEMP-table preamble) when it is too wide for one — and return
    the number of rows read."""
    import tempfile

    import yaml

    from featurizer import Featurizer

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    grouped = Featurizer(handle.name)._grouped()
    n_rows = 0
    with conn.cursor() as cur:
        if grouped.materialization is not None:
            for statement in grouped.materialization.ddl:
                cur.execute(statement)
        for sql in grouped.queries.values():
            cur.execute(sql)
            n_rows += len(cur.fetchall())
    return n_rows


@pytest.mark.parametrize("config", SHAPES)
def test_a_config_that_validates_runs(pg_conn, config) -> None:
    result = ConfigValidator().validate(config)
    assert result.is_valid, [error.message for error in result.errors]
    _seed(pg_conn)
    assert _run(pg_conn, config), "the query ran and returned no rows"
