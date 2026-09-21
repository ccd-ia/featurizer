"""Live-DB revalidation matrix — the release-discipline harness.

Runs featurizer against the three live triage databases (``dirtyduck``,
``chicago311``, ``donorschoose``) in the 3-DB × 3-variant matrix that gated
v0.6.0 and v0.8.0, and records per-cell artifacts under
``specs/live-db-revalidation-v100/raw/``. Reconstructed and committed for
v1.0 — the earlier snapshots were produced by an uncommitted script, a gap
this file closes.

Variants (the v0.8.0 definitions, recovered by count-calibration — narrow and
all-agg reproduce the v0.8.0 feature counts exactly; wide matches dirtyduck
1,252 and chicago311 907 exactly and lands within ~6% on donorschoose, the
0.9.x planner emitting slightly more features for the same config):

* ``narrow``  — the triage experiment's own ``feature_config`` as-is.
* ``all-agg`` — same entity graph, aggregations = the 65-name
  ``DEFAULT_AGGREGATIONS`` default-active set, transformations = ``identity``.
* ``wide``    — all-agg × the 14-transformer set below.

Environment: the live DBs are the triage-side Docker containers; credentials
are read from ``<triage>/<dataset>-database.yaml`` (never hardcoded here).
Runs are read-only — everything happens on one connection whose transaction
is rolled back (TEMP ``as_of_dates`` + TEMP shard tables vanish with it).

The single-date cell above is the one every published figure since ADR-0009
was measured with, and it is kept exactly as it is. It cannot see the cost of a
*paired* cohort, though: with one as-of date the dense ``as_of_dates x target``
product is exactly the wanted rows. ``--dates N`` adds the case that can
(issue #10). It spans N monthly as-of dates, pairs each date with the entities
that had an event in the month before it, and records the rows the dense query
computes against the rows on that diagonal — then runs the same config with
``as_of_dates: {id_column: …}`` and checks the paired result equals the dense
one on those pairs. Its artifacts go to their own directory,
``specs/paired-cohorts/raw/``, so the v1.0.0 record stays what it was.

``--jit-compare`` runs that same single-date cell four times, under ``jit``
off, on, on, off (issue #53): PostgreSQL compiles every expression of a wide
target list before it reads a row, and every figure published before
2026-09-20 was taken with the server default, ``jit = on``. Its artifacts go to
``specs/jit-on-off/raw/``.

Usage::

    uv run python -m benchmarks.final_matrix --dry-run          # counts only, no DB
    uv run python -m benchmarks.final_matrix --db dirtyduck     # one DB, all variants
    uv run python -m benchmarks.final_matrix --db donorschoose --variant wide
    uv run python -m benchmarks.final_matrix                    # the full matrix
    uv run python -m benchmarks.final_matrix --db dirtyduck --variant narrow --dates 6
    uv run python -m benchmarks.final_matrix --jit-compare       # the matrix, jit off/on
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List
from unittest import mock

import yaml

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "specs" / "live-db-revalidation-v100"
)
# The checkout was renamed; the harness could not find its configs by default.
DEFAULT_TRIAGE_DIR = Path.home() / "projects" / "triage-pg"
MULTI_DATE_ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "specs" / "paired-cohorts"
)
JIT_ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "specs" / "jit-on-off"

#: dataset -> the triage experiment file its ``feature_config`` comes from.
DATASETS: Dict[str, str] = {
    "dirtyduck": "example/dirtyduck/experiment.yaml",
    "chicago311": "example/chicago311/experiment.yaml",
    "donorschoose": "example/donorschoose/experiment.yaml",
}

#: The wide variant's transformer set — the historical "14 transformers
#: (lags, rolling, EMA, cusum…)". Recovered by calibrating against the
#: v0.8.0 feature counts: exactly four of these fire on dirtyduck/chicago311
#: (abs, cum_sum, ln, sqrt — 4×245 = the exact 980-feature wide delta; ln/sqrt
#: are also the v0.6.0 "ln of negative" crash trail), the window family fires
#: only where the target carries a temporal ordering (donorschoose).
WIDE_TRANSFORMERS: List[str] = [
    "identity",
    "abs",
    "cum_sum",
    "ln",
    "sqrt",
    "lag_1",
    "lag_3",
    "lag_7",
    "rolling_mean_3",
    "rolling_std_7",
    "ema_7",
    "pct_change_1",
    "cusum",
    "diff",
]


def load_feature_config(triage_dir: Path, dataset: str) -> Dict[str, Any]:
    """The ``feature_config`` block of the dataset's triage experiment."""
    with (triage_dir / DATASETS[dataset]).open() as handle:
        return yaml.safe_load(handle)["feature_config"]


def all_aggregation_names() -> List[str]:
    """The 65-name default-active aggregation set (the v0.8.0 "all-agg").

    ``DEFAULT_AGGREGATIONS`` — not the full registry: the extra registered
    families need special config (predicates / spatial_ix / boolean columns)
    and were never part of the matrix definition."""
    from featurizer.primitives.aggregations import DEFAULT_AGGREGATIONS

    return sorted(DEFAULT_AGGREGATIONS)


def build_variant(config: Dict[str, Any], variant: str) -> Dict[str, Any]:
    """Derive the matrix variant from the narrow (triage) config."""
    import copy

    config = copy.deepcopy(config)
    if variant == "narrow":
        return config
    config["aggregations"] = all_aggregation_names()
    config["transformations"] = (
        ["identity"] if variant == "all-agg" else list(WIDE_TRANSFORMERS)
    )
    return config


def featurizer_for(config: Dict[str, Any]):
    from featurizer import Featurizer

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
        return Featurizer(handle.name, validate=False)


def connect(triage_dir: Path, dataset: str):
    """psycopg connection from the triage ``<dataset>-database.yaml``."""
    import psycopg

    with (triage_dir / f"{dataset}-database.yaml").open() as handle:
        creds = yaml.safe_load(handle)
    return psycopg.connect(
        host=creds["host"],
        port=creds["port"],
        dbname=creds["db"],
        user=creds["user"],
        password=creds["pass"],
        autocommit=False,
    )


def latest_knowledge_date(conn, config: Dict[str, Any]) -> str:
    """``max(temporal_ix) + 1 day`` over the child event stream — the as-of
    date under which every event is knowable (the most-work, deterministic
    choice; recorded in the artifact for reproducibility)."""
    events = next(
        e
        for e in config["entities"]
        if e.get("temporal_ix") and e["alias"] != config["target"]
    )
    with conn.cursor() as cur:
        cur.execute(
            f"select (max({events['temporal_ix']})::date + 1)::text "
            f"from {events['table']}"
        )
        return cur.fetchone()[0]


def run_cell(
    triage_dir: Path, dataset: str, variant: str, *, dry_run: bool
) -> Dict[str, Any]:
    """One matrix cell. Returns the artifact record (and writes it)."""
    config = build_variant(load_feature_config(triage_dir, dataset), variant)

    t0 = time.perf_counter()
    f = featurizer_for(config)
    groups = f.query_groups  # forces plan + render for every shard
    render_s = time.perf_counter() - t0

    record: Dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "features": len(f.feature_manifest),
        "shards": len(groups),
        "render_seconds": round(render_s, 1),
        "featurizer_version": _version(),
    }

    if not dry_run:
        conn = connect(triage_dir, dataset)
        try:
            as_of = latest_knowledge_date(conn, config)
            with conn.cursor() as cur:
                cur.execute(
                    "create temp table as_of_dates (as_of_date date) on commit drop"
                )
                cur.execute("insert into as_of_dates values (%s)", (as_of,))
            t1 = time.perf_counter()
            frame = f.to_dataframe(connection=conn)
            exec_s = time.perf_counter() - t1
            dup = frame.columns.duplicated().sum()
            record.update(
                {
                    "as_of_date": as_of,
                    "rows": int(len(frame)),
                    "cols": int(frame.shape[1]),
                    "dup_names": int(dup),
                    "exec_seconds": round(exec_s, 1),
                    "status": "materialized",
                }
            )
        finally:
            conn.rollback()
            conn.close()

    out = ARTIFACT_DIR / "raw" / f"{dataset}-{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return record


#: Recorded with every JIT cell: what decides whether PostgreSQL compiles a
#: query, and what else moves a wall-clock between two servers.
JIT_SERVER_SETTINGS = (
    "server_version",
    "jit",
    "jit_above_cost",
    "jit_inline_above_cost",
    "jit_optimize_above_cost",
    "max_parallel_workers_per_gather",
    "shared_buffers",
    "work_mem",
)

#: The first run reads a cold cache, and it is a jit-off run on purpose: the
#: penalty lands on the side issue #53 expects to win. ``off`` run 1 minus run 2
#: is the size of the cache effect; ``on`` run 1 minus run 2 is what is left of it.
JIT_ORDER = ("off", "on", "on", "off")


@contextlib.contextmanager
def _engine_leaves_jit_alone() -> Iterator[None]:
    """Since the measurement, the engine turns ``jit`` off around its own
    statements (``featurizer.executor.jit_disabled``), on a caller's connection
    too. A ``jit = on`` run needs it not to, or it measures ``off`` twice."""
    import featurizer.executor as executor
    import featurizer.featurizer as engine

    def untouched(conn: Any) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()

    with contextlib.ExitStack() as stack:
        for module in (executor, engine):
            stack.enter_context(mock.patch.object(module, "jit_disabled", untouched))
        yield


def _frame_digest(frame) -> str:
    """A digest of the frame's values that does not depend on row order."""
    import hashlib

    import numpy as np
    import pandas as pd

    frame = frame.reset_index()
    rows = np.zeros(len(frame), dtype="uint64")
    for _, column in frame.items():  # by position: names may repeat
        # psycopg returns arrays as lists and json as dicts; neither hashes
        values = column.astype(str) if column.dtype == object else column
        hashed = pd.util.hash_pandas_object(values, index=False).to_numpy()
        rows = rows * np.uint64(1000003) + hashed
    return hashlib.sha256(np.sort(rows).tobytes()).hexdigest()


def run_jit_cell(triage_dir: Path, dataset: str, variant: str) -> Dict[str, Any]:
    """The single-date cell under ``jit`` off, on, on, off (issue #53).

    One connection, one rolled-back transaction per run, so ``set local jit``
    and a run's TEMP tables are gone before the next one starts. The cell is
    :func:`run_cell`'s, at the same as-of date, so a jit-on time here is
    comparable with the published figures. Every run is recorded; the two
    summary figures are the best of each setting's two runs.
    """
    config = build_variant(load_feature_config(triage_dir, dataset), variant)
    f = featurizer_for(config)
    groups = f.query_groups  # forces plan + render for every shard

    record: Dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "case": "jit-on-off",
        "features": len(f.feature_manifest),
        "shards": len(groups),
        "featurizer_version": _version(),
        "commit": _commit(),
    }

    conn = connect(triage_dir, dataset)
    try:
        as_of = latest_knowledge_date(conn, config)
        server: Dict[str, Any] = {}
        with conn.cursor() as cur:
            for name in JIT_SERVER_SETTINGS:
                cur.execute(f"show {name}")
                server[name] = cur.fetchone()[0]
            cur.execute("select pg_jit_available()")
            server["pg_jit_available"] = cur.fetchone()[0]
        conn.rollback()

        runs: List[Dict[str, Any]] = []
        for jit in JIT_ORDER:
            with conn.cursor() as cur:
                cur.execute(f"set local jit = {jit}")
                cur.execute(
                    "create temp table as_of_dates (as_of_date date) on commit drop"
                )
                cur.execute("insert into as_of_dates values (%s)", (as_of,))
            t0 = time.perf_counter()
            with _engine_leaves_jit_alone():
                frame = f.to_dataframe(connection=conn)
            exec_s = time.perf_counter() - t0
            conn.rollback()
            runs.append(
                {
                    "jit": jit,
                    "exec_seconds": round(exec_s, 2),
                    "rows": int(len(frame)),
                    "cols": int(frame.shape[1]),
                    "digest": _frame_digest(frame),
                }
            )
            del frame  # donorschoose wide is about 1 GB a frame
        record.update(
            {
                "as_of_date": as_of,
                "server": server,
                "runs": runs,
                "jit_on_seconds": min(
                    r["exec_seconds"] for r in runs if r["jit"] == "on"
                ),
                "jit_off_seconds": min(
                    r["exec_seconds"] for r in runs if r["jit"] == "off"
                ),
                "values_identical": len({r["digest"] for r in runs}) == 1,
                "status": "materialized",
            }
        )
    finally:
        conn.rollback()
        conn.close()

    out = JIT_ARTIFACT_DIR / "raw" / f"{dataset}-{variant}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return record


#: The pair table's id column in the multi-date case.
PAIR_ID_COLUMN = "cohort_entity_id"


def _pairing(config: Dict[str, Any]) -> Dict[str, str]:
    """What the multi-date case needs to build a diagonal from the config.

    The events entity is the one :func:`latest_knowledge_date` uses; an entity
    is in a date's cohort when it has an event in the month before that date.
    Only the plain shape is handled — the events' relationship to the target
    must join on the target's own id — because that is the shape of all three
    live datasets and a wrong pairing would measure the wrong thing.
    """
    target = next(e for e in config["entities"] if e["alias"] == config["target"])
    events = next(
        e
        for e in config["entities"]
        if e.get("temporal_ix") and e["alias"] != config["target"]
    )
    rel = next(
        r
        for r in config["relationships"]
        if r["parent"]["entity"] == target["alias"]
        and r["child"]["entity"] == events["alias"]
    )
    if rel["parent"]["key"] != target["id"]:
        raise ValueError(
            f"multi-date case: relationship {target['alias']} <- {events['alias']} "
            f"joins on {rel['parent']['key']!r}, not on the target id "
            f"{target['id']!r}; the diagonal cannot be derived from the events. "
            "Extend _pairing() for this shape before benchmarking it."
        )
    return {
        "target_id": target["id"],
        "events_table": events["table"],
        "events_ts": events["temporal_ix"],
        "events_key": rel["child"]["key"],
    }


def _create_pairs(cur, pairing: Dict[str, str], table: str, last: str, n: int) -> None:
    """``table(as_of_date, <PAIR_ID_COLUMN>)``: each of ``n`` monthly dates ending
    at ``last``, paired with the ids that had an event in the month before it."""
    cur.execute(
        f"create temp table {table} (as_of_date date, {PAIR_ID_COLUMN} bigint) "
        "on commit drop"
    )
    cur.execute(
        f"""
        insert into {table}
        select d::date, ev.{pairing["events_key"]}
        from generate_series(%s::date - (%s - 1) * interval '1 month',
                             %s::date, interval '1 month') as d
        join lateral (
            select distinct {pairing["events_key"]}
            from {pairing["events_table"]}
            where {pairing["events_ts"]} < d::date
              and {pairing["events_ts"]} >= d::date - interval '1 month'
        ) ev on true
        """,
        (last, n, last),
    )
    cur.execute(f"analyze {table}")


def run_multi_date_cell(
    triage_dir: Path, dataset: str, variant: str, n_dates: int
) -> Dict[str, Any]:
    """Dense vs paired over ``n_dates`` monthly as-of dates (issue #10).

    Two transactions on one connection, each rolled back, so each run gets its
    own TEMP ``as_of_dates``: the distinct dates for the dense run, the pair
    table itself for the paired run.
    """
    import copy

    config = build_variant(load_feature_config(triage_dir, dataset), variant)
    pairing = _pairing(config)
    paired_config = copy.deepcopy(config)
    paired_config["as_of_dates"] = {"id_column": PAIR_ID_COLUMN}

    dense_f = featurizer_for(config)
    paired_f = featurizer_for(paired_config)
    record: Dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "case": "multi-date",
        "features": len(dense_f.feature_manifest),
        "featurizer_version": _version(),
    }

    conn = connect(triage_dir, dataset)
    try:
        last = latest_knowledge_date(conn, config)

        with conn.cursor() as cur:
            _create_pairs(cur, pairing, "cohort_pairs", last, n_dates)
            cur.execute(f"select as_of_date, {PAIR_ID_COLUMN} from cohort_pairs")
            wanted = set(cur.fetchall())
            cur.execute(
                "create temp table as_of_dates on commit drop as "
                "select distinct as_of_date from cohort_pairs"
            )
        t0 = time.perf_counter()
        dense = dense_f.to_dataframe(connection=conn)
        dense_s = time.perf_counter() - t0
        conn.rollback()

        with conn.cursor() as cur:
            _create_pairs(cur, pairing, "as_of_dates", last, n_dates)
        t1 = time.perf_counter()
        paired = paired_f.to_dataframe(connection=conn)
        paired_s = time.perf_counter() - t1

        # Compare VALUES on the pairs, column by column. ``DataFrame.equals``
        # also compares dtypes, and those legitimately differ: an entity with
        # no events gives NULL in the dense frame (float64) where the paired
        # cohort, all of which have events, stays int64.
        keys = ["as_of_date", pairing["target_id"]]
        dense = dense.reset_index()
        paired = paired.reset_index().sort_values(keys).reset_index(drop=True)
        on_pairs = (
            dense.merge(paired[keys], on=keys).sort_values(keys).reset_index(drop=True)
        )
        differing = [
            col
            for col in on_pairs.columns
            if len(on_pairs) != len(paired)
            or not (
                (on_pairs[col] == paired[col])
                | (on_pairs[col].isna() & paired[col].isna())
            ).all()
        ]
        same = (
            len(paired) == len(wanted)
            and list(on_pairs.columns) == list(paired.columns)
            and not differing
        )
        record.update(
            {
                "last_as_of_date": last,
                "dates": n_dates,
                "rows_computed_dense": int(len(dense)),
                "rows_on_diagonal": len(wanted),
                "dense_to_diagonal": round(len(dense) / max(len(wanted), 1), 1),
                "rows_paired": int(len(paired)),
                "dense_exec_seconds": round(dense_s, 1),
                "paired_exec_seconds": round(paired_s, 1),
                "paired_equals_dense_on_pairs": bool(same),
                "columns_differing": len(differing),
                "status": "materialized",
            }
        )
    finally:
        conn.rollback()
        conn.close()

    out = MULTI_DATE_ARTIFACT_DIR / "raw" / f"{dataset}-{variant}-dates{n_dates}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return record


def _version() -> str:
    from importlib.metadata import version

    try:
        return version("featurizer")
    except Exception:
        return "dev"


def _commit() -> str:
    """The checkout's commit: the package version does not move between tags."""
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", choices=sorted(DATASETS), help="one dataset only")
    parser.add_argument(
        "--variant", choices=["narrow", "all-agg", "wide"], help="one variant only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="render + count only (no database connection)",
    )
    parser.add_argument(
        "--dates",
        type=int,
        metavar="N",
        help="run the multi-date paired-cohort case over N monthly as-of dates "
        "instead of the single-date cell (issue #10)",
    )
    parser.add_argument(
        "--jit-compare",
        action="store_true",
        help="run the single-date cell under jit off, on, on, off and record "
        "every run (issue #53)",
    )
    parser.add_argument(
        "--triage-dir",
        type=Path,
        default=DEFAULT_TRIAGE_DIR,
        help="the triage checkout holding experiment + database YAMLs",
    )
    args = parser.parse_args()
    if args.jit_compare and (args.dates or args.dry_run):
        parser.error("--jit-compare is its own case; drop --dates / --dry-run")

    datasets = [args.db] if args.db else sorted(DATASETS)
    variants = [args.variant] if args.variant else ["narrow", "all-agg", "wide"]
    for dataset in datasets:
        for variant in variants:
            if args.jit_compare:
                run_jit_cell(args.triage_dir, dataset, variant)
            elif args.dates:
                if args.dry_run:
                    parser.error("--dates needs a database; drop --dry-run")
                run_multi_date_cell(args.triage_dir, dataset, variant, args.dates)
            else:
                run_cell(args.triage_dir, dataset, variant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
