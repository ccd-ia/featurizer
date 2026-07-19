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

Usage::

    uv run python -m benchmarks.final_matrix --dry-run          # counts only, no DB
    uv run python -m benchmarks.final_matrix --db dirtyduck     # one DB, all variants
    uv run python -m benchmarks.final_matrix --db donorschoose --variant wide
    uv run python -m benchmarks.final_matrix                    # the full matrix
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent / "specs" / "live-db-revalidation-v100"
)
DEFAULT_TRIAGE_DIR = Path.home() / "projects" / "triage"

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


def _version() -> str:
    from importlib.metadata import version

    try:
        return version("featurizer")
    except Exception:
        return "dev"


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
        "--triage-dir",
        type=Path,
        default=DEFAULT_TRIAGE_DIR,
        help="the triage checkout holding experiment + database YAMLs",
    )
    args = parser.parse_args()

    datasets = [args.db] if args.db else sorted(DATASETS)
    variants = [args.variant] if args.variant else ["narrow", "all-agg", "wide"]
    for dataset in datasets:
        for variant in variants:
            run_cell(args.triage_dir, dataset, variant, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
