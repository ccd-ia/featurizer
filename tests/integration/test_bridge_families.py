"""End-to-end 0.9.0 bridge families: materialize → the SQL spine aggregates.

Each family materializes its φ column(s) against the live database, splices the
``emit_yaml`` fragment into a config, and proves the spine's aggregation
matches an independent recomputation with the causal cut applied. The
dependency-free NLP bridges run everywhere; spaCy-model and graph cases gate on
their optional deps. Graph families (Phase 3) land in this file too.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from featurizer.bridge import (
    CentralityBridge,
    CommunityBridge,
    LanguageIdBridge,
    NERCountsBridge,
    SentimentBridge,
)

from ._harness import create_temp_table, run_featurizer
from ._realistic import expect_sql, feature, make_as_of_dates

pytestmark = [pytest.mark.integration]

AS_OF_DATES = ["2020-06-01", "2020-12-31"]


def _seed_docs(conn) -> None:
    create_temp_table(conn, "owners", [("owner_id", "int")], [(1,), (2,)])
    create_temp_table(
        conn,
        "docs",
        [("doc_id", "int"), ("owner_id", "int"), ("ts", "date"), ("body", "text")],
        [
            (1, 1, date(2020, 1, 1), "Excelente servicio, muy bueno."),
            (2, 1, date(2020, 2, 1), "Terrible retraso otra vez"),
            (3, 1, date(2020, 9, 1), "bueno"),  # knowable only at the late as-of
            (4, 2, date(2020, 1, 15), "The delivery was terrible and late"),
        ],
    )


def _spine_config(fragment: dict, aggregations: list[str]) -> dict:
    return {
        "target": "owners",
        "max_depth": 2,
        "intervals": [],
        "aggregations": aggregations,
        "transformations": ["identity"],
        "entities": [
            {"alias": "owners", "table": "owners", "id": "owner_id"},
            fragment["entity"],
        ],
        "relationships": [fragment["relationship"]],
    }


def test_sentiment_bridge_flows_through_the_spine(pg_conn):
    """MEAN(sentiment) over the parent equals an independent causally-cut
    average of the materialized column — per owner, per as-of."""
    _seed_docs(pg_conn)
    bridge = SentimentBridge(pk_col="doc_id", text_col="body", language="xx")
    bridge.materialize(
        pg_conn,
        source_table="docs",
        pk="doc_id",
        carry_cols=["owner_id", "ts"],
        content_cols=["body"],
        output_table="bridge_sentiment",
    )
    fragment = bridge.emit_yaml(
        output_table="bridge_sentiment",
        pk="doc_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    make_as_of_dates(pg_conn, AS_OF_DATES)
    rows = run_featurizer(pg_conn, _spine_config(fragment, ["mean", "count"]))
    assert len(rows) == len(AS_OF_DATES) * 2

    for as_of in AS_OF_DATES:
        for owner_id in (1, 2):
            got = feature(
                rows,
                as_of=as_of,
                id_col="owner_id",
                entity_id=owner_id,
                col_substr="MEAN(sentiment.sentiment)",
            )
            want = expect_sql(
                pg_conn,
                "select avg(sentiment) from bridge_sentiment "
                "where owner_id = %s and ts <= %s",
                (owner_id, as_of),
            )
            assert (got is None) == (want is None)
            if want is not None:
                assert math.isclose(float(got), float(want), rel_tol=1e-9)

    # The September doc is outside the June window: the causal cut is visible.
    early = feature(
        rows,
        as_of="2020-06-01",
        id_col="owner_id",
        entity_id=1,
        col_substr="COUNT(sentiment.doc_id)",
    )
    late = feature(
        rows,
        as_of="2020-12-31",
        id_col="owner_id",
        entity_id=1,
        col_substr="COUNT(sentiment.doc_id)",
    )
    assert (int(early), int(late)) == (2, 3)


def test_language_id_multicolumn_materializes_categorical(pg_conn):
    """The MultiColumnBridge DDL path on real PostgreSQL: a categorical value
    column lands as text and the detected codes are correct per document."""
    _seed_docs(pg_conn)
    bridge = LanguageIdBridge(pk_col="doc_id", text_col="body")
    bridge.materialize(
        pg_conn,
        source_table="docs",
        pk="doc_id",
        carry_cols=["owner_id", "ts"],
        content_cols=["body"],
        output_table="bridge_language",
    )
    col_type = expect_sql(
        pg_conn,
        "select data_type from information_schema.columns "
        "where table_name = 'bridge_language' and column_name = 'language'",
    )
    assert col_type == "text"
    assert (
        expect_sql(
            pg_conn,
            "select language from bridge_language where doc_id = 1",
        )
        == "es"
    )
    assert (
        expect_sql(
            pg_conn,
            "select language from bridge_language where doc_id = 4",
        )
        == "en"
    )

    # The fragment splices into the spine like any entity (nunique over the
    # categorical column, causally cut: owner 1 is all-Spanish).
    fragment = bridge.emit_yaml(
        output_table="bridge_language",
        pk="doc_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    make_as_of_dates(pg_conn, AS_OF_DATES)
    rows = run_featurizer(pg_conn, _spine_config(fragment, ["nunique"]))
    got = feature(
        rows,
        as_of="2020-12-31",
        id_col="owner_id",
        entity_id=1,
        col_substr="NUNIQUE(language_id.language)",
    )
    assert int(got) == 1


def test_persisted_bridge_table_survives_commit(pg_conn):
    """persist=True writes a real table (an orchestrated asset, ADR-0014)."""
    _seed_docs(pg_conn)
    bridge = SentimentBridge(pk_col="doc_id", text_col="body")
    try:
        bridge.materialize(
            pg_conn,
            source_table="docs",
            pk="doc_id",
            carry_cols=["owner_id", "ts"],
            content_cols=["body"],
            output_table="bridge_sentiment_asset",
            persist=True,
        )
        is_temp = expect_sql(
            pg_conn,
            "select n.nspname like 'pg_temp%%' from pg_class c "
            "join pg_namespace n on n.oid = c.relnamespace "
            "where c.relname = 'bridge_sentiment_asset'",
        )
        assert is_temp is False
    finally:
        with pg_conn.cursor() as cur:
            cur.execute("drop table if exists bridge_sentiment_asset")


def test_ner_counts_multicolumn_via_real_model(pg_conn):
    """Skip-gated on the spaCy Spanish model: one parse, five columns, spine
    MEAN over one of them."""
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("es_core_news_sm")
    except OSError:
        pytest.skip("es_core_news_sm not downloaded")

    create_temp_table(pg_conn, "owners", [("owner_id", "int")], [(1,)])
    create_temp_table(
        pg_conn,
        "docs",
        [("doc_id", "int"), ("owner_id", "int"), ("ts", "date"), ("body", "text")],
        [
            (
                1,
                1,
                date(2020, 1, 1),
                "Juan Pérez firmó un contrato con la empresa Pemex en Veracruz.",
            )
        ],
    )
    bridge = NERCountsBridge(pk_col="doc_id", text_col="body", language="es")
    bridge.materialize(
        pg_conn,
        source_table="docs",
        pk="doc_id",
        carry_cols=["owner_id", "ts"],
        content_cols=["body"],
        output_table="bridge_ner",
    )
    persons = expect_sql(pg_conn, "select persons from bridge_ner where doc_id = 1")
    assert persons is not None and persons >= 1

    fragment = bridge.emit_yaml(
        output_table="bridge_ner",
        pk="doc_id",
        parent_alias="owners",
        parent_key="owner_id",
        fk="owner_id",
        temporal_ix="ts",
    )
    make_as_of_dates(pg_conn, ["2020-06-01"])
    rows = run_featurizer(pg_conn, _spine_config(fragment, ["mean"]))
    got = feature(
        rows,
        as_of="2020-06-01",
        id_col="owner_id",
        entity_id=1,
        col_substr="MEAN(ner_counts.persons)",
    )
    assert float(got) >= 1.0


def _seed_graph(conn) -> None:
    create_temp_table(
        conn, "nodes", [("node_id", "text")], [("A",), ("B",), ("C",), ("D",)]
    )
    create_temp_table(
        conn,
        "edges",
        [("src", "text"), ("dst", "text"), ("ts", "date")],
        [
            ("A", "B", date(2020, 1, 1)),
            ("B", "C", date(2020, 2, 1)),
            ("A", "C", date(2020, 3, 1)),
            ("A", "D", date(2020, 9, 1)),  # future at the June window
        ],
    )


def test_graph_centrality_snapshot_stream_through_the_spine(pg_conn):
    """materialize_snapshots on real PG: per-(node, as_of) rows, the June
    window excludes the September edge, and the spine trends the metric."""
    _seed_graph(pg_conn)
    bridge = CentralityBridge(source_col="src", target_col="dst", directed=False)
    bridge.materialize_snapshots(
        pg_conn,
        source_table="edges",
        output_table="bridge_centrality",
        as_of_dates=[date(2020, 6, 1), date(2020, 12, 31)],
        causal_col="ts",
        content_cols=["src", "dst"],
        entity_col="node_id",
        as_of_col="as_of_date",
    )
    # June: triangle only (3 nodes); December: pendant arrived (4 nodes).
    assert (
        expect_sql(
            pg_conn,
            "select count(*) from bridge_centrality where as_of_date = %s",
            (date(2020, 6, 1),),
        )
        == 3
    )
    assert (
        expect_sql(
            pg_conn,
            "select degree from bridge_centrality "
            "where node_id = 'A' and as_of_date = %s",
            (date(2020, 6, 1),),
        )
        == 2.0
    )
    assert (
        expect_sql(
            pg_conn,
            "select degree from bridge_centrality "
            "where node_id = 'A' and as_of_date = %s",
            (date(2020, 12, 31),),
        )
        == 3.0
    )

    # The snapshot table is an ordinary event stream: MAX(degree) as-of June
    # sees only the June snapshot; as-of December sees both.
    fragment = bridge.emit_yaml(
        output_table="bridge_centrality",
        pk="node_id",
        parent_alias="nodes",
        parent_key="node_id",
        fk="node_id",
        temporal_ix="as_of_date",
    )
    config = {
        "target": "nodes",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "nodes", "table": "nodes", "id": "node_id"},
            fragment["entity"],
        ],
        "relationships": [fragment["relationship"]],
    }
    make_as_of_dates(pg_conn, ["2020-06-01", "2020-12-31"])
    rows = run_featurizer(pg_conn, config)

    def max_degree(as_of):
        return feature(
            rows,
            as_of=as_of,
            id_col="node_id",
            entity_id="A",
            col_substr="MAX(centrality.degree)",
        )

    assert float(max_degree("2020-06-01")) == 2.0
    assert float(max_degree("2020-12-31")) == 3.0


def test_graph_community_membership_materializes_per_node(pg_conn):
    """materialize_nodes on real PG: categorical membership column + the
    two-clique structure recovered."""
    create_temp_table(
        pg_conn,
        "clique_edges",
        [("src", "text"), ("dst", "text")],
        [
            ("a1", "a2"),
            ("a1", "a3"),
            ("a2", "a3"),
            ("b1", "b2"),
            ("b1", "b3"),
            ("b2", "b3"),
            ("a3", "b1"),
        ],
    )
    bridge = CommunityBridge(source_col="src", target_col="dst")
    bridge.materialize_nodes(
        pg_conn,
        source_table="clique_edges",
        output_table="bridge_community",
        content_cols=["src", "dst"],
        node_col="node_id",
    )
    assert (
        expect_sql(pg_conn, "select count(distinct community_id) from bridge_community")
        == 2
    )
    same_clique = expect_sql(
        pg_conn,
        "select count(distinct community_id) from bridge_community "
        "where node_id in ('a1', 'a2', 'a3')",
    )
    assert same_clique == 1
    assert (
        expect_sql(
            pg_conn,
            "select data_type from information_schema.columns where "
            "table_name = 'bridge_community' and column_name = 'community_id'",
        )
        == "text"
    )


def test_pipeline_text_to_edges_to_centrality_to_spine(pg_conn):
    """The Path-2 two-stage wiring end to end on real PG: near-duplicate text
    induces an edge table; the centrality bridge snapshots it per window; the
    spine trends the metric. The copy-paste pair completes only in September,
    so the June window has no graph at all."""
    from featurizer.bridge import NearDuplicateEdgeBridge

    paste = (
        "El contrato fue firmado sin licitación previa por la empresa "
        "constructora del corredor interoceánico en marzo"
    )
    create_temp_table(
        pg_conn, "authors", [("author_id", "text")], [("a",), ("b",), ("c",)]
    )
    create_temp_table(
        pg_conn,
        "posts",
        [
            ("post_id", "int"),
            ("author_id", "text"),
            ("posted_at", "date"),
            ("body", "text"),
        ],
        [
            (1, "a", date(2020, 1, 10), paste),
            (2, "b", date(2020, 9, 10), paste),  # the copy appears here
            (3, "c", date(2020, 2, 1), "sin coincidencias con nadie más"),
        ],
    )

    # Stage 1: text -> edge table (knowable at the LATER document).
    edge_bridge = NearDuplicateEdgeBridge(
        pk_col="post_id", entity_col="author_id", text_col="body", ts_col="posted_at"
    )
    edge_bridge.materialize_edges(
        pg_conn,
        source_table="posts",
        output_table="text_edges",
        content_cols=["post_id", "author_id", "posted_at", "body"],
    )
    assert expect_sql(pg_conn, "select count(*) from text_edges") == 1
    assert expect_sql(pg_conn, "select ts from text_edges") == date(2020, 9, 10)

    # Stage 2: edge table -> per-(node, as_of) centrality snapshots.
    centrality = CentralityBridge(source_col="src", target_col="dst", directed=False)
    centrality.materialize_snapshots(
        pg_conn,
        source_table="text_edges",
        output_table="text_centrality",
        as_of_dates=[date(2020, 6, 1), date(2020, 12, 31)],
        causal_col="ts",
        content_cols=["src", "dst"],
        entity_col="node_id",
        as_of_col="as_of_date",
    )
    # June window: the pair is not knowable yet -> no graph, no rows.
    assert (
        expect_sql(
            pg_conn,
            "select count(*) from text_centrality where as_of_date = %s",
            (date(2020, 6, 1),),
        )
        == 0
    )

    # Stage 3: the spine aggregates the snapshot stream as-of.
    fragment = centrality.emit_yaml(
        output_table="text_centrality",
        pk="node_id",
        parent_alias="authors",
        parent_key="author_id",
        fk="node_id",
        temporal_ix="as_of_date",
    )
    config = {
        "target": "authors",
        "max_depth": 2,
        "intervals": [],
        "aggregations": ["max"],
        "transformations": ["identity"],
        "entities": [
            {"alias": "authors", "table": "authors", "id": "author_id"},
            fragment["entity"],
        ],
        "relationships": [fragment["relationship"]],
    }
    make_as_of_dates(pg_conn, ["2020-06-01", "2020-12-31"])
    rows = run_featurizer(pg_conn, config)

    def degree(as_of, author):
        return feature(
            rows,
            as_of=as_of,
            id_col="author_id",
            entity_id=author,
            col_substr="MAX(centrality.degree)",
        )

    assert degree("2020-06-01", "a") is None  # nothing knowable in June
    assert float(degree("2020-12-31", "a")) == 1.0
    assert float(degree("2020-12-31", "b")) == 1.0
    assert degree("2020-12-31", "c") is None  # never in the induced graph
