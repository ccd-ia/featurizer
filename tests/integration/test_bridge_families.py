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

from featurizer.bridge import LanguageIdBridge, NERCountsBridge, SentimentBridge

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
        [(1, 1, date(2020, 1, 1), "Juan Pérez trabaja en Petróleos Mexicanos.")],
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
