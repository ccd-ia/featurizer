"""Unit tests for the embedding-trajectory φ (no database).

Hand-built embedding sequences with a planted outlier: an entity that repeats
itself twice and then says something orthogonal must score novelty/drift/
volatility ≈ 1 on the outlier and ≈ 0 on the repeat. History is strictly
per-entity and strictly prior — the first event is NULL, and another entity's
vectors never leak in.
"""

from __future__ import annotations

import pytest

from featurizer.bridge import EmbeddingTrajectoryBridge
from featurizer.bridge.trajectory import _parse_vector


def _bridge() -> EmbeddingTrajectoryBridge:
    return EmbeddingTrajectoryBridge(
        pk_col="eid", fk_col="owner", order_col="ts", embedding_col="emb"
    )


ROWS = [
    {"eid": 1, "owner": "a", "ts": 1, "emb": [1.0, 0.0]},
    {"eid": 2, "owner": "a", "ts": 2, "emb": [1.0, 0.0]},
    {"eid": 3, "owner": "a", "ts": 3, "emb": [0.0, 1.0]},  # planted outlier
    {"eid": 4, "owner": "b", "ts": 1, "emb": [0.0, 1.0]},  # other entity
]


def test_planted_outlier_is_recovered():
    phi = _bridge().compute(ROWS, fit_rows=ROWS)
    # First event: no history -> NULLs (not "maximally novel").
    assert phi[1] == {"novelty": None, "drift": None, "volatility": None}
    # Exact repeat: zero everywhere.
    assert phi[2]["novelty"] == pytest.approx(0.0)
    assert phi[2]["drift"] == pytest.approx(0.0)
    assert phi[2]["volatility"] == pytest.approx(0.0)
    # Orthogonal outlier: max distance on all three columns.
    assert phi[3]["novelty"] == pytest.approx(1.0)
    assert phi[3]["drift"] == pytest.approx(1.0)
    assert phi[3]["volatility"] == pytest.approx(1.0)


def test_history_is_per_entity():
    """Entity b's first event has no history even though entity a already
    said the identical vector — cross-entity history must never leak."""
    phi = _bridge().compute(ROWS, fit_rows=ROWS)
    assert phi[4] == {"novelty": None, "drift": None, "volatility": None}


def test_history_is_strictly_prior_not_symmetric():
    """The repeat of a vector scores 0 only for the LATER event; reversing
    the order moves the novelty to the other row."""
    rows = [
        {"eid": 1, "owner": "a", "ts": 2, "emb": [1.0, 0.0]},  # later
        {"eid": 2, "owner": "a", "ts": 1, "emb": [0.0, 1.0]},  # earlier
    ]
    phi = _bridge().compute(rows, fit_rows=rows)
    assert phi[2] == {"novelty": None, "drift": None, "volatility": None}
    assert phi[1]["novelty"] == pytest.approx(1.0)


def test_drift_uses_the_centroid_not_the_nearest():
    """Half the history matches exactly, half is orthogonal: novelty (max
    cosine) is 0 but drift (centroid distance) is strictly positive."""
    rows = [
        {"eid": 1, "owner": "a", "ts": 1, "emb": [1.0, 0.0]},
        {"eid": 2, "owner": "a", "ts": 2, "emb": [0.0, 1.0]},
        {"eid": 3, "owner": "a", "ts": 3, "emb": [1.0, 0.0]},
    ]
    phi = _bridge().compute(rows, fit_rows=rows)
    assert phi[3]["novelty"] == pytest.approx(0.0)
    # centroid = [0.5, 0.5]; cos([1,0], centroid) = 1/sqrt(2).
    assert phi[3]["drift"] == pytest.approx(1.0 - 2**-0.5)
    # volatility is the step to the immediately previous (orthogonal) event.
    assert phi[3]["volatility"] == pytest.approx(1.0)


def test_missing_embeddings_are_null_and_skip_history():
    rows = [
        {"eid": 1, "owner": "a", "ts": 1, "emb": None},
        {"eid": 2, "owner": "a", "ts": 2, "emb": [1.0, 0.0]},
    ]
    phi = _bridge().compute(rows, fit_rows=rows)
    assert phi[1] == {"novelty": None, "drift": None, "volatility": None}
    # The unparseable row contributed no history: eid 2 is still "first".
    assert phi[2] == {"novelty": None, "drift": None, "volatility": None}


def test_numeric_order_is_not_string_order():
    """Orders 2 and 10: string sorting would reverse them ("10" < "2")."""
    rows = [
        {"eid": 1, "owner": "a", "ts": 2, "emb": [1.0, 0.0]},
        {"eid": 2, "owner": "a", "ts": 10, "emb": [0.0, 1.0]},
    ]
    phi = _bridge().compute(rows, fit_rows=rows)
    assert phi[1]["novelty"] is None  # ts=2 really is first
    assert phi[2]["novelty"] == pytest.approx(1.0)


def test_parse_vector_accepts_pgvector_text_and_arrays():
    assert _parse_vector("[1.0, 2.0]").tolist() == [1.0, 2.0]
    assert _parse_vector((3.0, 4.0)).tolist() == [3.0, 4.0]
    assert _parse_vector(None) is None
    assert _parse_vector("[]") is None


def test_emit_yaml_declares_the_three_columns():
    fragment = _bridge().emit_yaml(
        output_table="t",
        pk="eid",
        parent_alias="owners",
        parent_key="owner",
        fk="owner",
        temporal_ix="ts",
    )
    assert fragment["entity"]["variables"] == {
        "novelty": {"type": "numeric"},
        "drift": {"type": "numeric"},
        "volatility": {"type": "numeric"},
    }
