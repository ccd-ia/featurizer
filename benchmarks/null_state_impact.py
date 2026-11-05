"""What ADR-0020 (a NULL category is a state, #86) moves on the live databases.

Builds each triage experiment's entity graph with ``aggregations`` set to the
11 sequence primitives and ``transformations: [identity]``, runs it at the
latest knowledge date of the dataset (``final_matrix.latest_knowledge_date``)
and writes canonical rows, so the same command run on ``master`` and on the
fix gives two files to diff. It also counts NULLs per categorical child column,
which is what explains a moved cell.

Read-only: one connection per dataset, its transaction rolled back (the TEMP
``as_of_dates`` vanishes with it). Credentials come from the triage-side
``<dataset>-database.yaml``, as in ``final_matrix``.

Usage::

    uv run python -m benchmarks.null_state_impact --db dirtyduck --out before.json
    uv run python -m benchmarks.null_state_impact --diff before.json after.json
    uv run python -m benchmarks.null_state_impact --db dirtyduck --nulls

To run the ``master`` engine with this script, put a ``master`` checkout first
on the import path: ``PYTHONPATH=../featurizer-before uv run python -m …``. The
output records which ``featurizer`` was imported.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Tuple

from benchmarks import final_matrix

SEQUENCE_PRIMITIVES = [
    "longest_streak",
    "markov_conditional_entropy",
    "max_transition_prob",
    "ngram_2_freq",
    "ngram_3_freq",
    "recurrence_interval",
    "rework_count",
    "sequence_entropy",
    "state_volatility",
    "time_in_current_state",
    "transition_matrix_summary",
]


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _config(
    triage_dir: Path, dataset: str, aggregations: List[str] = SEQUENCE_PRIMITIVES
) -> Dict[str, Any]:
    config = final_matrix.load_feature_config(triage_dir, dataset)
    config["aggregations"] = list(aggregations)
    config["transformations"] = ["identity"]
    return config


def _cell(value: Any) -> Any:
    """A JSON-stable cell: NaN and NA become None, numbers become float/int."""
    import pandas as pd

    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _is_feature(column: str) -> bool:
    return "(" in column


def run(
    triage_dir: Path, dataset: str, aggregations: List[str] = SEQUENCE_PRIMITIVES
) -> Dict[str, Any]:
    """Canonical rows of ``aggregations`` over the dataset's entity graph; also
    used by ``benchmarks.interval_window_impact`` with its own list."""
    import featurizer

    config = _config(triage_dir, dataset, aggregations)
    f = final_matrix.featurizer_for(config)
    conn = final_matrix.connect(triage_dir, dataset)
    try:
        as_of = final_matrix.latest_knowledge_date(conn, config)
        with conn.cursor() as cur:
            cur.execute(
                "create temp table as_of_dates (as_of_date date) on commit drop"
            )
            cur.execute("insert into as_of_dates values (%s)", (as_of,))
        frame = f.to_dataframe(connection=conn)
    finally:
        conn.rollback()
        conn.close()
    # The frame is indexed by (as_of_date, target id): that is the row key.
    # Columns without a call in their name (one-hots, carried keys) are not
    # aggregations and are left out.
    keys = [str(name) for name in frame.index.names]
    frame = frame.reset_index()
    features = sorted(c for c in frame.columns if _is_feature(c))
    rows = [
        {c: _cell(v) for c, v in record.items()}
        for record in frame[keys + features].to_dict("records")
    ]
    rows.sort(key=lambda r: tuple(str(r[k]) for k in keys))
    return {
        "dataset": dataset,
        "as_of_date": as_of,
        "featurizer": str(Path(featurizer.__file__).resolve().parent.parent),
        "keys": keys,
        "features": features,
        "rows": rows,
    }


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def diff(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """Moved cells per column, moved entities, moved columns per primitive."""
    assert before["keys"] == after["keys"], "key columns differ"
    assert before["features"] == after["features"], "feature columns differ"
    keys = before["keys"]

    def index(doc: Dict[str, Any]) -> Dict[Tuple[str, ...], Dict[str, Any]]:
        return {tuple(str(r[k]) for k in keys): r for r in doc["rows"]}

    b, a = index(before), index(after)
    assert len(b) == len(before["rows"]) and len(a) == len(after["rows"]), (
        f"the key {keys} is not unique: rows would collapse"
    )
    assert b.keys() == a.keys(), "entity sets differ"
    cells: Counter = Counter()
    entities = set()
    for key, row in b.items():
        for column in before["features"]:
            if not _same(row[column], a[key][column]):
                cells[column] += 1
                entities.add(key)
    per_primitive: Counter = Counter(c.split("(")[0].lower() for c in cells)
    return {
        "dataset": before["dataset"],
        "rows": len(b),
        "feature_columns": len(before["features"]),
        "moved_cells": sum(cells.values()),
        "moved_entities": len(entities),
        "moved_columns": len(cells),
        "moved_cells_per_column": dict(sorted(cells.items())),
        "moved_columns_per_primitive": dict(sorted(per_primitive.items())),
    }


def nulls(triage_dir: Path, dataset: str) -> List[Dict[str, Any]]:
    """Per categorical child column: rows and NULL rows within the as-of bound."""
    config = _config(triage_dir, dataset)
    out: List[Dict[str, Any]] = []
    conn = final_matrix.connect(triage_dir, dataset)
    try:
        as_of = final_matrix.latest_knowledge_date(conn, config)
        with conn.cursor() as cur:
            for entity in config["entities"]:
                tix = entity.get("temporal_ix")
                if entity["alias"] == config["target"] or not tix:
                    continue
                for name, spec in (entity.get("variables") or {}).items():
                    if (spec or {}).get("type") != "categorical":
                        continue
                    cur.execute(
                        f"select count(*), count(*) filter (where {_quote(name)} is null) "
                        f"from {entity['table']} where {_quote(tix)} <= %s::date",
                        (as_of,),
                    )
                    total, null = cur.fetchone()
                    out.append(
                        {
                            "dataset": dataset,
                            "entity": entity["alias"],
                            "column": name,
                            "rows": total,
                            "null_rows": null,
                            "null_share": round(null / total, 6) if total else None,
                        }
                    )
    finally:
        conn.rollback()
        conn.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", choices=sorted(final_matrix.DATASETS))
    parser.add_argument(
        "--triage-dir", type=Path, default=final_matrix.DEFAULT_TRIAGE_DIR
    )
    parser.add_argument("--out", type=Path, help="write canonical rows to FILE")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--nulls", action="store_true")
    args = parser.parse_args()

    if args.diff:
        before, after = (json.loads(p.read_text()) for p in args.diff)
        print(json.dumps(diff(before, after), indent=2))
        return
    if not args.db:
        parser.error("--db is required with --out or --nulls")
    if args.nulls:
        print(json.dumps(nulls(args.triage_dir, args.db), indent=2))
    if args.out:
        result = run(args.triage_dir, args.db)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result) + "\n")
        print(
            f"{args.db}: {len(result['rows'])} rows x {len(result['features'])} "
            f"features at {result['as_of_date']} from {result['featurizer']} "
            f"-> {args.out}"
        )


if __name__ == "__main__":
    main()
