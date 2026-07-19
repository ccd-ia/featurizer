"""0.9.x feature families at realistic scale — the v1.0 additions no earlier
release measured.

Three workloads against the live triage databases (read-only: TEMP tables on
a rolled-back connection, same contract as :mod:`benchmarks.final_matrix`):

* ``graph``      — the native ``graph_relationships`` planner pass over a
  derived chain/school edge table on the real cohort, DEGREE verified
  against hand SQL on a sample; wall-clock + EXPLAIN shape recorded.
* ``centrality`` — ``CentralityBridge.materialize_snapshots`` across the
  cohort's as-of dates: the O(windows × build) cost the docs assert, cheap
  tier vs ``include_heavy``, measured.
* ``text``       — a dependency-free text bridge (``SentimentBridge``, en)
  over real inspector comments, materialized and spine-aggregated end to
  end, MEAN verified against hand SQL for a sample entity.

Edge derivations mirror the seeded harness exactly (facilities sharing a
normalized name / projects sharing a school, chain size 2..50, one row per
unordered pair, ``knowable_at = greatest`` of the endpoints' appearance —
the edge exists once BOTH endpoints do).

Usage::

    uv run python -m benchmarks.bridge_workloads                       # all three
    uv run python -m benchmarks.bridge_workloads --workload graph
    uv run python -m benchmarks.bridge_workloads --workload graph --db donorschoose
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict, List

from .final_matrix import (
    ARTIFACT_DIR,
    DEFAULT_TRIAGE_DIR,
    connect,
    featurizer_for,
    load_feature_config,
)

#: Three as-of dates spanning each cohort's event history (the backtest shape
#: the snapshot cost model is about); the last is the full-history date used
#: by final_matrix.
AS_OF_DATES: Dict[str, List[str]] = {
    "dirtyduck": ["2016-01-01", "2017-01-01", "2017-12-30"],
    "donorschoose": ["2013-01-01", "2013-07-01", "2014-01-01"],
}

#: Edge-table derivations (TEMP; the seeded-harness rules on the live data).
EDGE_SQL: Dict[str, str] = {
    "dirtyduck": """
        create temp table bench_edges on commit drop as
        with named as (
            select entity_id, lower(trim(facility)) as chain_name, start_time
            from ontology.entities
            where facility is not null and trim(facility) <> ''
        ),
        chains as (
            select chain_name from named
            group by chain_name having count(*) between 2 and 50
        )
        select a.entity_id as src,
               b.entity_id as dst,
               greatest(a.start_time, b.start_time) as knowable_at
        from named a
        join named b using (chain_name)
        join chains using (chain_name)
        where a.entity_id < b.entity_id
    """,
    "donorschoose": """
        create temp table bench_edges on commit drop as
        with named as (
            select entity_id, schoolid, date_posted
            from ontology.entities
            where schoolid is not null
        ),
        schools as (
            select schoolid from named
            group by schoolid having count(*) between 2 and 50
        )
        select a.entity_id as src,
               b.entity_id as dst,
               greatest(a.date_posted, b.date_posted) as knowable_at
        from named a
        join named b using (schoolid)
        join schools using (schoolid)
        where a.entity_id < b.entity_id
    """,
}


def _make_as_of_dates(conn: Any, dates: List[str]) -> None:
    with conn.cursor() as cur:
        cur.execute("create temp table as_of_dates (as_of_date date) on commit drop")
        cur.executemany("insert into as_of_dates values (%s)", [(d,) for d in dates])


def _write(record: Dict[str, Any], name: str) -> None:
    out = ARTIFACT_DIR / "raw" / f"bridge-{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


# ------------------------------------------------------------------ #
# Workload 1: the native graph_relationships planner pass
# ------------------------------------------------------------------ #


def run_graph(triage_dir, dataset: str) -> Dict[str, Any]:
    config = load_feature_config(triage_dir, dataset)
    target = config["target"]
    target_entity = next(e for e in config["entities"] if e["alias"] == target)
    config["graph_relationships"] = [
        {
            "name": "chains",
            "left": target,
            "edges": {
                "table": "bench_edges",
                "source": "src",
                "target": "dst",
                "timestamp": "knowable_at",
            },
            "directed": False,
            "features": ["degree"],
        }
    ]

    conn = connect(triage_dir, dataset)
    try:
        with conn.cursor() as cur:
            cur.execute(EDGE_SQL[dataset])
            cur.execute("select count(*) from bench_edges")
            n_edges = cur.fetchone()[0]
        _make_as_of_dates(conn, AS_OF_DATES[dataset])

        f = featurizer_for(config)
        sql = f.query
        t0 = time.perf_counter()
        frame = f.to_dataframe(connection=conn)
        exec_s = time.perf_counter() - t0

        degree_col = next(c for c in frame.columns if c == "DEGREE(chains)")

        # Hand-SQL verification on a sample: DEGREE(chains) at each as-of date
        # = edges (either direction) with knowable_at <= as_of.
        id_col = target_entity["id"]
        sample = (
            frame.reset_index()
            .loc[lambda d: d[degree_col].fillna(0) > 0, ["as_of_date", id_col]]
            .head(5)
        )
        checked = 0
        with conn.cursor() as cur:
            for _, row in sample.iterrows():
                cur.execute(
                    "select count(*) from bench_edges "
                    "where (src = %s or dst = %s) and knowable_at <= %s",
                    (row[id_col], row[id_col], row["as_of_date"]),
                )
                expected = float(cur.fetchone()[0])
                got = float(
                    frame.xs(
                        (row["as_of_date"], row[id_col]),
                        level=("as_of_date", id_col),
                    )[degree_col].iloc[0]
                )
                assert got == expected, (
                    f"DEGREE mismatch for {row[id_col]}@{row['as_of_date']}: "
                    f"featurizer={got} hand-SQL={expected}"
                )
                checked += 1

            # EXPLAIN shape of the generated query (planning only, no exec).
            cur.execute("explain (costs off) " + sql)
            plan = [r[0] for r in cur.fetchall()]

        record = {
            "workload": "graph_relationships",
            "dataset": dataset,
            "edges": int(n_edges),
            "as_of_dates": AS_OF_DATES[dataset],
            "rows": int(len(frame)),
            "features": len(f.feature_manifest),
            "degree_columns": sorted(
                c for c in frame.columns if c.startswith("DEGREE(")
            ),
            "hand_sql_verified": checked,
            "exec_seconds": round(exec_s, 1),
            "explain_top": plan[:4],
            "explain_nodes": len(plan),
        }
    finally:
        conn.rollback()
        conn.close()
    _write(record, f"graph-{dataset}")
    return record


# ------------------------------------------------------------------ #
# Workload 2: CentralityBridge snapshot sequences (cheap vs heavy)
# ------------------------------------------------------------------ #


def run_centrality(triage_dir, dataset: str) -> Dict[str, Any]:
    from datetime import date

    from featurizer.bridge import CentralityBridge

    # The bridge slices in Python, so the as-of dates must be date objects
    # (the DB hands back datetime.date for knowable_at).
    as_of_dates = [date.fromisoformat(d) for d in AS_OF_DATES[dataset]]
    record: Dict[str, Any] = {
        "workload": "centrality_snapshots",
        "dataset": dataset,
        "as_of_dates": AS_OF_DATES[dataset],
    }
    for tier, heavy in (("cheap", False), ("heavy", True)):
        bridge = CentralityBridge(
            source_col="src",
            target_col="dst",
            directed=False,
            include_heavy=heavy,
            name="centrality",
        )
        conn = connect(triage_dir, dataset)
        try:
            with conn.cursor() as cur:
                cur.execute(EDGE_SQL[dataset])
                cur.execute("select count(*) from bench_edges")
                n_edges = cur.fetchone()[0]
            t0 = time.perf_counter()
            bridge.materialize_snapshots(
                conn,
                source_table="bench_edges",
                output_table="bench_centrality",
                as_of_dates=as_of_dates,
                causal_col="knowable_at",
                content_cols=["src", "dst"],
            )
            seconds = time.perf_counter() - t0
            with conn.cursor() as cur:
                cur.execute("select count(*) from bench_centrality")
                n_rows = cur.fetchone()[0]
            record[f"{tier}_seconds"] = round(seconds, 1)
            record[f"{tier}_snapshot_rows"] = int(n_rows)
            record["edges"] = int(n_edges)
        finally:
            conn.rollback()
            conn.close()
    record["heavy_over_cheap"] = round(
        record["heavy_seconds"] / max(record["cheap_seconds"], 1e-9), 1
    )
    _write(record, f"centrality-{dataset}")
    return record


# ------------------------------------------------------------------ #
# Workload 3: a text bridge end to end (materialize -> spine-aggregate)
# ------------------------------------------------------------------ #

#: Inspector-comment cap: the bridge computes in-process; the cap keeps the
#: workload bounded and is RECORDED in the artifact (no silent truncation).
TEXT_ROW_CAP = 50_000


def run_text(triage_dir) -> Dict[str, Any]:
    """SentimentBridge (en) over dirtyduck violation comments, aggregated to
    facilities through the ordinary SQL spine."""
    from featurizer.bridge import SentimentBridge

    conn = connect(triage_dir, "dirtyduck")
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from clean.violations where comment is not null"
            )
            total = cur.fetchone()[0]
            cur.execute(f"""
                create temp table bench_comments on commit drop as
                select row_number() over (order by v.date, v.inspection, v.code)
                           as comment_id,
                       e.entity_id,
                       v.date,
                       v.comment
                from clean.violations v
                join ontology.entities e on e.license_num = v.license_num
                where v.comment is not null
                limit {TEXT_ROW_CAP}
            """)
            cur.execute("select count(*) from bench_comments")
            n_rows = cur.fetchone()[0]

        bridge = SentimentBridge(pk_col="comment_id", text_col="comment", language="en")
        t0 = time.perf_counter()
        bridge.materialize(
            conn,
            source_table="bench_comments",
            pk="comment_id",
            output_table="bench_sentiment",
            carry_cols=["entity_id", "date"],
            content_cols=["comment"],
        )
        bridge_s = time.perf_counter() - t0

        fragment = bridge.emit_yaml(
            output_table="bench_sentiment",
            pk="comment_id",
            parent_alias="facilities",
            parent_key="entity_id",
            fk="entity_id",
            temporal_ix="date",
        )
        fragment["entity"]["temporal_ix"] = "date"
        config = load_feature_config(triage_dir, "dirtyduck")
        config["entities"].append(fragment["entity"])
        config["relationships"].append(fragment["relationship"])
        config["aggregations"] = ["count", "mean", "min", "max"]
        config["transformations"] = ["identity"]

        _make_as_of_dates(conn, AS_OF_DATES["dirtyduck"][-1:])
        f = featurizer_for(config)
        t1 = time.perf_counter()
        frame = f.to_dataframe(connection=conn)
        spine_s = time.perf_counter() - t1

        mean_col = "MEAN(sentiment.sentiment)"
        sample = (
            frame.reset_index()
            .loc[lambda d: d[mean_col].notna(), ["as_of_date", "entity_id"]]
            .head(3)
        )
        checked = 0
        with conn.cursor() as cur:
            for _, row in sample.iterrows():
                cur.execute(
                    "select avg(sentiment) from bench_sentiment "
                    "where entity_id = %s and date <= %s",
                    (row["entity_id"], row["as_of_date"]),
                )
                expected = float(cur.fetchone()[0])
                got = float(
                    frame.xs(
                        (row["as_of_date"], row["entity_id"]),
                        level=("as_of_date", "entity_id"),
                    )[mean_col].iloc[0]
                )
                assert abs(got - expected) < 1e-9, (
                    f"MEAN(sentiment) mismatch for {row['entity_id']}: "
                    f"featurizer={got} hand-SQL={expected}"
                )
                checked += 1

        record = {
            "workload": "text_bridge_end_to_end",
            "dataset": "dirtyduck",
            "bridge": "SentimentBridge(en)",
            "comments_total": int(total),
            "comments_processed": int(n_rows),
            "row_cap": TEXT_ROW_CAP,
            "bridge_seconds": round(bridge_s, 1),
            "spine_seconds": round(spine_s, 1),
            "rows": int(len(frame)),
            "sentiment_columns": sorted(c for c in frame.columns if "sentiment" in c),
            "hand_sql_verified": checked,
        }
    finally:
        conn.rollback()
        conn.close()
    _write(record, "text-dirtyduck")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", choices=["graph", "centrality", "text"], help="one workload"
    )
    parser.add_argument(
        "--db", choices=sorted(EDGE_SQL), help="one dataset (graph/centrality)"
    )
    parser.add_argument("--triage-dir", type=str, default=str(DEFAULT_TRIAGE_DIR))
    args = parser.parse_args()

    from pathlib import Path

    triage_dir = Path(args.triage_dir)
    graph_dbs = [args.db] if args.db else sorted(EDGE_SQL)
    workloads = [args.workload] if args.workload else ["graph", "centrality", "text"]
    for workload in workloads:
        if workload == "graph":
            for dataset in graph_dbs:
                run_graph(triage_dir, dataset)
        elif workload == "centrality":
            for dataset in graph_dbs:
                run_centrality(triage_dir, dataset)
        else:
            run_text(triage_dir)


if __name__ == "__main__":
    main()
