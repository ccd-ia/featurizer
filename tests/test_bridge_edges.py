"""Unit tests for the text-induced edge builders (no database).

A copy-paste pair across two entities must yield an edge (knowable at the
LATER document's timestamp) while distinct texts and same-entity self-copies
yield none; co-mentions pair the names inside one document. datasketch sits in
the dev group so these execute (not skip) under plain ``uv sync``.
"""

from __future__ import annotations

from datetime import date

import pytest

from featurizer.bridge import CoMentionEdgeBridge, NearDuplicateEdgeBridge

PASTE = (
    "El contrato fue firmado sin licitación previa por la empresa "
    "constructora del corredor interoceánico en marzo"
)
DISTINCT = "Hoy llovió toda la tarde en la ciudad y se suspendió el partido"


def _neardup(**kw) -> NearDuplicateEdgeBridge:
    return NearDuplicateEdgeBridge(
        pk_col="doc_id", entity_col="owner", text_col="body", ts_col="ts", **kw
    )


def test_copy_paste_pair_yields_one_edge_at_the_later_ts():
    rows = [
        {"doc_id": 1, "owner": "a", "ts": date(2020, 1, 1), "body": PASTE},
        {"doc_id": 2, "owner": "b", "ts": date(2020, 3, 1), "body": PASTE},
        {"doc_id": 3, "owner": "c", "ts": date(2020, 1, 5), "body": DISTINCT},
    ]
    edges = _neardup().compute_edges(rows)
    # One edge, canonically ordered, knowable when the COPY appears.
    assert edges == [("a", "b", date(2020, 3, 1))]


def test_distinct_texts_yield_no_edge():
    rows = [
        {"doc_id": 1, "owner": "a", "ts": date(2020, 1, 1), "body": PASTE},
        {"doc_id": 2, "owner": "b", "ts": date(2020, 1, 2), "body": DISTINCT},
    ]
    assert _neardup().compute_edges(rows) == []


def test_self_copies_are_not_coordination():
    rows = [
        {"doc_id": 1, "owner": "a", "ts": date(2020, 1, 1), "body": PASTE},
        {"doc_id": 2, "owner": "a", "ts": date(2020, 2, 1), "body": PASTE},
    ]
    assert _neardup().compute_edges(rows) == []


def test_repeated_duplicates_yield_repeated_edges():
    """Three entities pasting the same text: every pair gets its edge — the
    graph bridges read repetition as weight."""
    rows = [
        {"doc_id": i, "owner": o, "ts": date(2020, 1, i), "body": PASTE}
        for i, o in ((1, "a"), (2, "b"), (3, "c"))
    ]
    edges = _neardup().compute_edges(rows)
    assert sorted((s, d) for s, d, _ in edges) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]


def test_empty_and_missing_text_is_skipped():
    rows = [
        {"doc_id": 1, "owner": "a", "ts": date(2020, 1, 1), "body": ""},
        {"doc_id": 2, "owner": "b", "ts": date(2020, 1, 2), "body": None},
    ]
    assert _neardup().compute_edges(rows) == []
    with pytest.raises(ValueError, match="threshold"):
        _neardup(threshold=0.0)


def test_co_mentions_pair_names_within_one_document():
    bridge = CoMentionEdgeBridge(text_col="body", ts_col="ts")
    rows = [
        {
            "ts": date(2020, 1, 1),
            "body": "pemex firmó un convenio con Odebrecht y con Sedena",
        },
        {"ts": date(2020, 2, 1), "body": "sin nombres propios aquí"},
    ]
    edges = bridge.compute_edges(rows)
    assert edges == [("Odebrecht", "Sedena", date(2020, 1, 1))]


def test_co_mentions_accept_a_custom_extractor():
    bridge = CoMentionEdgeBridge(
        text_col="body",
        ts_col="ts",
        extract=lambda text: [w for w in text.split() if w.startswith("@")],
    )
    rows = [{"ts": 1, "body": "rt @ana y @beto sobre el corredor"}]
    assert bridge.compute_edges(rows) == [("@ana", "@beto", 1)]
