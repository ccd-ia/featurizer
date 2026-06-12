"""Helpers for the realistic-dataset integration tier.

Pattern for every realistic test::

    config = load_config("food_inspections")
    cohort = make_cohort(food_db, source_table="food_inspections.facilities",
                         order_by="license_no", n=50)
    retarget(config, "facilities", cohort)
    make_as_of_dates(food_db, ["2015-01-01", "2015-07-01", "2016-07-01"])
    rows = run_featurizer(food_db, config)
    assert feature(rows, as_of="2015-07-01", id_col="license_no",
                   entity_id=1234, col_substr="COUNT(") == expect_sql(
        food_db, "select count(*) from food_inspections.inspections "
                 "where license_no = 1234 and inspection_date <= '2015-07-01'")

The cohort and ``as_of_dates`` are session TEMP tables on the rolled-back
``pg_conn`` connection, so they resolve ahead of permanent tables in the
generated SQL and vanish at teardown — seeded schemas are never written to.
The TEMP cohort is also the performance lever: featurizer computes features
only for the sampled target rows.

Extension protocol for new feature families (M1c Markov sequence, M1b-2
recursive graph, ...): add the new registry names to the ``aggregations:`` /
``edge.features`` lists of the relevant config under ``datasets/configs/``,
then add one test per family following the fixed pattern above —
``feature(...)`` compared against an ``expect_sql(...)`` independent
recomputation (and, optionally, frozen constants guarded by
``seed_meta.subset_sha256``). Inspection ``results`` sequences are the
realistic Markov substrate; ``chain_edges`` / ``project_edges`` are the
realistic graph substrate.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import yaml

CONFIG_DIR = Path(__file__).parent / "datasets" / "configs"


def load_config(name: str, **overrides: Any) -> dict:
    """Load ``datasets/configs/<name>.yaml`` as a dict, applying overrides."""
    with (CONFIG_DIR / f"{name}.yaml").open() as handle:
        config = yaml.safe_load(handle)
    config.update(overrides)
    return copy.deepcopy(config)


def retarget(config: dict, alias: str, table: str) -> dict:
    """Point entity ``alias`` at ``table`` (typically the TEMP cohort)."""
    for entity in config["entities"]:
        if entity["alias"] == alias:
            entity["table"] = table
            return config
    raise ValueError(f"No entity with alias {alias!r} in config")


def make_cohort(
    conn: Any,
    *,
    source_table: str,
    order_by: str,
    n: int,
    name: str | None = None,
) -> str:
    """Create a deterministic TEMP cohort table sampled from ``source_table``.

    Returns the (unqualified) temp table name; pass it to :func:`retarget`.
    """
    cohort = name or f"cohort_{source_table.rsplit('.', 1)[-1]}"
    with conn.cursor() as cur:
        cur.execute(
            f"create temp table {cohort} on commit drop as "
            f"select * from {source_table} order by {order_by} limit {int(n)}"
        )
    return cohort


def make_child_subset(
    conn: Any,
    *,
    source_table: str,
    key_col: str,
    cohort: str,
    cohort_key: str,
    name: str | None = None,
) -> str:
    """Create a TEMP copy of a child table restricted to the cohort's keys.

    The generated aggregation CTEs group the *entire* child table (correlated
    SubqueryAggregator features then rescan that un-indexed CTE per group, per
    as-of date) — on a realistic table that is quadratic and unusably slow.
    Restricting children to the cohort keeps the SQL shape and the real data
    values identical while bounding the work to the sampled entities. Scale
    behaviour of cheap aggregators is covered separately by a full-table
    smoke test.
    """
    subset = name or f"subset_{source_table.rsplit('.', 1)[-1]}"
    with conn.cursor() as cur:
        cur.execute(
            f"create temp table {subset} on commit drop as "
            f"select s.* from {source_table} s "
            f"where s.{key_col} in (select {cohort_key} from {cohort})"
        )
    return subset


def make_as_of_dates(conn: Any, dates: Sequence[str | date]) -> None:
    """Create the TEMP ``as_of_dates`` table the generated SQL expects."""
    with conn.cursor() as cur:
        cur.execute("create temp table as_of_dates (as_of_date date) on commit drop")
        cur.executemany(
            "insert into as_of_dates values (%s)", [(str(d),) for d in dates]
        )


def feature_columns(rows: Sequence[dict], col_substr: str) -> list[str]:
    """Output column names containing ``col_substr`` (case-insensitive).

    An exact column-name match short-circuits to that single column, so
    ``"DEGREE(facilities.chain_edges)"`` selects the total-degree column even
    though it is also a substring of ``IN_DEGREE(...)`` / ``OUT_DEGREE(...)``.
    """
    if not rows:
        return []
    if col_substr in rows[0]:
        return [col_substr]
    needle = col_substr.lower()
    return [col for col in rows[0] if needle in col.lower()]


def feature(
    rows: Sequence[dict],
    *,
    as_of: str | date,
    id_col: str,
    entity_id: Any,
    col_substr: str,
) -> Any:
    """Value of the single feature column matching ``col_substr`` for one
    ``(as_of_date, entity)`` cell of the output grid.

    Raises with the candidate column names when the match is not unique —
    feature names are generated, so this keeps test failures self-explanatory.
    """
    matches = feature_columns(rows, col_substr)
    if len(matches) != 1:
        available = "\n  ".join(sorted(rows[0])) if rows else "(no rows)"
        raise AssertionError(
            f"Expected exactly one column matching {col_substr!r}, got "
            f"{matches!r}. Available columns:\n  {available}"
        )
    column = matches[0]
    wanted = str(as_of)
    cell = [
        row
        for row in rows
        if str(row["as_of_date"]) == wanted and row.get(id_col) == entity_id
    ]
    if len(cell) != 1:
        raise AssertionError(
            f"Expected exactly one row for as_of={wanted!r} {id_col}={entity_id!r}, "
            f"got {len(cell)}"
        )
    return cell[0][column]


def expect_sql(conn: Any, sql: str, params: Sequence[Any] = ()) -> Any:
    """Run an independent recomputation query and return its single scalar."""
    with conn.cursor() as cur:
        cur.execute(sql, params)
        row = cur.fetchone()
    assert row is not None, f"expect_sql returned no rows: {sql}"
    return row[0]
