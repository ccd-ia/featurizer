"""End-to-end φ-bridge: materialize a column, then the SQL spine aggregates it.

Runs the pure-Python MarkovSurprisalBridge against a synthetic event table on the
live database, then declares the materialized column (via ``emit_yaml``) as a
``Variable`` and confirms featurizer's aggregation of it matches an independent
recomputation over the same column — the bridge → spine handoff the architecture
depends on. A pgvector-gated embeddings probe rounds out the optional path.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from featurizer.bridge import MarkovSurprisalBridge

from ._harness import create_temp_table, run_featurizer
from ._realistic import expect_sql, feature, make_as_of_dates

pytestmark = [pytest.mark.integration]

AS_OF_DATES = ["2020-02-15", "2020-12-31"]


def _seed_events(conn) -> None:
    create_temp_table(
        conn,
        "owners",
        [("owner_id", "int")],
        [(1,), (2,)],
    )
    create_temp_table(
        conn,
        "events",
        [("event_id", "int"), ("owner_id", "int"), ("ts", "date"), ("state", "text")],
        [
            (1, 1, date(2020, 1, 1), "A"),
            (2, 1, date(2020, 2, 1), "B"),
            (3, 1, date(2020, 3, 1), "A"),
            (4, 2, date(2020, 1, 1), "C"),
            (5, 2, date(2020, 2, 1), "C"),
        ],
    )


def test_bridge_column_flows_through_the_sql_spine(pg_conn):
    """Materialize φ, declare it as a Variable, and check MEAN over the parent
    equals an independent average of the materialized column (causally cut)."""
    _seed_events(pg_conn)
    bridge = MarkovSurprisalBridge(
        pk_col="event_id", fk_col="owner_id", order_col="ts", state_col="state"
    )
    bridge.materialize(
        pg_conn,
        source_table="events",
        pk="event_id",
        carry_cols=["owner_id", "ts"],
        content_cols=["state"],
        output_table="bridge_events",
        causal_col="ts",
        fit_before=date(2020, 12, 31),
    )

    fragment = bridge.emit_yaml(
        output_table="bridge_events",
        pk="event_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    config = {
        "target": "owners",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["mean", "max", "count"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "owners", "table": "owners", "id": "owner_id"},
            fragment["entity"],
        ],
        "relationships": [fragment["relationship"]],
    }
    make_as_of_dates(pg_conn, AS_OF_DATES)
    rows = run_featurizer(pg_conn, config)
    assert len(rows) == len(AS_OF_DATES) * 2

    for as_of in AS_OF_DATES:
        for owner_id in (1, 2):
            got = feature(
                rows,
                as_of=as_of,
                id_col="owner_id",
                entity_id=owner_id,
                col_substr="MEAN(markov_surprisal.markov_surprisal)",
            )
            want = expect_sql(
                pg_conn,
                "select avg(markov_surprisal) from bridge_events "
                "where owner_id = %s and ts <= %s",
                (owner_id, as_of),
            )
            if want is None:
                assert got is None
            else:
                assert math.isclose(float(got), float(want), rel_tol=1e-9)


def test_bridge_count_is_causally_cut(pg_conn):
    """COUNT of the bridge events at the early as-of is strictly less than at the
    late one for owner 1 (events span the cutoff) — the spine's bound applies."""
    _seed_events(pg_conn)
    bridge = MarkovSurprisalBridge(
        pk_col="event_id", fk_col="owner_id", order_col="ts", state_col="state"
    )
    bridge.materialize(
        pg_conn,
        source_table="events",
        pk="event_id",
        carry_cols=["owner_id", "ts"],
        content_cols=["state"],
        output_table="bridge_events",
        causal_col="ts",
        fit_before=date(2020, 12, 31),
    )
    fragment = bridge.emit_yaml(
        output_table="bridge_events",
        pk="event_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    config = {
        "target": "owners",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["count"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "owners", "table": "owners", "id": "owner_id"},
            fragment["entity"],
        ],
        "relationships": [fragment["relationship"]],
    }
    make_as_of_dates(pg_conn, AS_OF_DATES)
    rows = run_featurizer(pg_conn, config)

    def count(as_of):
        return feature(
            rows,
            as_of=as_of,
            id_col="owner_id",
            entity_id=1,
            col_substr="COUNT(markov_surprisal.event_id)",
        )

    assert int(count("2020-02-15")) == 2  # e1, e2
    assert int(count("2020-12-31")) == 3  # e1, e2, e3


def test_embeddings_bridge_requires_pgvector(pg_conn):
    """The embeddings exemplar needs both sentence-transformers and the pgvector
    extension; skip cleanly when either is unavailable."""
    pytest.importorskip("sentence_transformers")
    with pg_conn.cursor() as cur:
        cur.execute("select 1 from pg_available_extensions where name = 'vector'")
        if cur.fetchone() is None:
            pytest.skip("pgvector extension is not available on this server")
        cur.execute("create extension if not exists vector")

    from featurizer.bridge import SentenceEmbeddingBridge

    create_temp_table(
        pg_conn,
        "docs",
        [("doc_id", "int"), ("body", "text")],
        [(1, "inspection passed"), (2, "critical violation found")],
    )
    bridge = SentenceEmbeddingBridge(pk_col="doc_id", text_col="body")
    bridge.materialize(
        pg_conn,
        source_table="docs",
        pk="doc_id",
        carry_cols=[],
        content_cols=["body"],
        output_table="bridge_docs",
    )
    assert (
        expect_sql(
            pg_conn,
            "select count(*) from bridge_docs where sentence_embedding is not null",
        )
        == 2
    )
