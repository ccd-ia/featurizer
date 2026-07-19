#!/usr/bin/env python
"""Run Example 6: text → edges → centrality → spine (two-stage bridges)."""

import argparse
import os
import sys
from pathlib import Path

# Add repo root (featurizer) and examples/ (_db) to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import _db
import psycopg
import yaml

from featurizer import Featurizer
from featurizer.bridge import (
    CentralityBridge,
    NearDuplicateEdgeBridge,
    SentimentBridge,
)

SCHEMA = "example_06"
BRIDGE_TABLES = ("bridge_sentiment", "text_edges", "post_centrality")


def run_bridges() -> dict:
    """Stage the φ precomputes: sentiment column, induced edges, centrality
    snapshots — all persisted as real tables (the ADR-0014 asset flow).

    Returns the emit_yaml fragments so main() can prove config.yaml matches.
    """
    conn = psycopg.connect(_db.require_conninfo())
    with conn.cursor() as cur:
        cur.execute(f'set search_path to "{SCHEMA}"')
        for table in BRIDGE_TABLES:  # idempotent re-runs
            cur.execute(f"drop table if exists {table}")
        cur.execute("select as_of_date from as_of_dates order by as_of_date")
        as_of_dates = [row[0] for row in cur.fetchall()]

    # Path 1: reduce each post to a sentiment scalar (dependency-free,
    # Spanish-register default lexicon).
    sentiment = SentimentBridge(pk_col="post_id", text_col="body")
    sentiment.materialize(
        conn,
        source_table="posts",
        pk="post_id",
        carry_cols=["author_id", "posted_at"],
        content_cols=["body"],
        output_table="bridge_sentiment",
        persist=True,
    )
    print("  ✓ bridge_sentiment: one valence column per post")

    # Path 2, stage 1: near-duplicate text induces an (src, dst, ts) edge
    # table — each edge knowable at the LATER post of the pair.
    edges = NearDuplicateEdgeBridge(
        pk_col="post_id",
        entity_col="author_id",
        text_col="body",
        ts_col="posted_at",
    )
    edges.materialize_edges(
        conn,
        source_table="posts",
        output_table="text_edges",
        content_cols=["post_id", "author_id", "posted_at", "body"],
        persist=True,
    )

    # Path 2, stage 2: rebuild the graph per as-of window from pre-t₀ edges
    # (never slice one full-history graph) → (node, as_of_date) snapshots.
    centrality = CentralityBridge(source_col="src", target_col="dst", directed=False)
    centrality.materialize_snapshots(
        conn,
        source_table="text_edges",
        output_table="post_centrality",
        as_of_dates=as_of_dates,
        causal_col="ts",
        content_cols=["src", "dst"],
        entity_col="node_id",
        as_of_col="as_of_date",
        persist=True,
    )
    with conn.cursor() as cur:
        cur.execute("select count(*) from text_edges")
        n_edges = cur.fetchone()[0]
        cur.execute("select count(*) from post_centrality")
        n_snap = cur.fetchone()[0]
    print(f"  ✓ text_edges: {n_edges} induced edge(s)")
    print(f"  ✓ post_centrality: {n_snap} (node, as_of_date) snapshot rows")
    conn.commit()
    conn.close()

    return {
        "sentiment": sentiment.emit_yaml(
            output_table="bridge_sentiment",
            pk="post_id",
            parent_alias="authors",
            parent_key="author_id",
            fk="author_id",
            temporal_ix="posted_at",
        ),
        "centrality": centrality.emit_yaml(
            output_table="post_centrality",
            pk="node_id",
            parent_alias="authors",
            parent_key="author_id",
            fk="node_id",
            temporal_ix="as_of_date",
        ),
    }


def assert_config_matches(config_path: Path, fragments: dict) -> None:
    """config.yaml's bridge entities must equal the emit_yaml fragments —
    the declared config cannot drift from what the bridges actually emit."""
    declared = yaml.safe_load(config_path.read_text())
    entities = {e["alias"]: e for e in declared["entities"]}
    for name, fragment in fragments.items():
        assert (
            entities[name] == fragment["entity"]
        ), f"config.yaml entity '{name}' drifted from emit_yaml()"
        assert (
            fragment["relationship"] in declared["relationships"]
        ), f"config.yaml relationship for '{name}' drifted from emit_yaml()"
    print("  ✓ config.yaml matches the bridges' emit_yaml fragments")


def main():
    parser = argparse.ArgumentParser(
        description="Run Example 6: text → edges → centrality → spine"
    )
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against PostgreSQL"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"

    if args.execute:
        print("\n🌉 Stage 1+2: φ-bridges (sentiment, induced edges, centrality)")
        fragments = run_bridges()
        assert_config_matches(config_path, fragments)

    print("\nLoading configuration...")
    featurizer = Featurizer(str(config_path))

    print("\n📊 Feature Generation Summary")
    print(f"  Target entity: {featurizer.target.alias}")
    print("  Bridge children: sentiment (Path 1), centrality (Path 2 snapshots)")

    if args.show_sql:
        print("\n📝 Generated SQL Query:")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    if args.execute:
        print("\n⚙️  Stage 3: the SQL spine aggregates the bridge columns...")
        os.environ["DATABASE_URL"] = _db.records_url(SCHEMA)
        try:
            df = featurizer.to_dataframe()
            print("✓ Query executed successfully!")
            print(f"\nResults shape: {df.shape}")
            show = [
                c
                for c in df.columns
                if "MAX(centrality.degree)" in c
                or "MAX(centrality.clustering)" in c
                or "MEAN(sentiment.sentiment)" in c
            ]
            print("\nThe coordination signature (per author, per as-of):")
            print(df[show].to_string())
            if args.output:
                output_path = Path(__file__).parent / args.output
                df.to_csv(output_path)
                print(f"\n✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"\n✗ Error executing query: {e}")
            print(
                "  Is the data loaded? Run `python create_data.py` (or "
                "`just example 06`) against a running PostgreSQL (`just db-up`)."
            )
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
