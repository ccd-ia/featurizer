# coding: utf-8

"""Embedding-trajectory φ: novelty, drift, volatility per event (Path 3).

Given per-event embeddings (typically produced by
:class:`~featurizer.bridge.embeddings.SentenceEmbeddingBridge`, but any vector
column works), :class:`EmbeddingTrajectoryBridge` scores each event against
the entity's **own strictly-prior history** — the "out of character?" signal:

- ``novelty`` — ``1 − max cosine`` to any prior embedding of the same entity
  (distance of the new message from everything the entity has said before).
- ``drift`` — cosine distance to the centroid of the prior embeddings (how far
  the entity has moved from its own reference point).
- ``volatility`` — cosine distance to the *immediately previous* embedding
  (the per-event step size; the spine's rolling std/mean over it is the
  semantic-volatility trend).

All three are NULL for an entity's first event (no history ≠ maximally novel).
Causality is by construction: each event's φ reads only events strictly
*before* it in ``order_col``, so the value is knowable the moment the event
is — the spine's normal ``<= as_of_date`` bound does the rest. No model is
fitted (``fit_rows`` is unused).

Embeddings are accepted as Python sequences, PostgreSQL arrays, or pgvector's
``"[0.1, 0.2, …]"`` text form — so the output of a materialized
``SentenceEmbeddingBridge`` table can be read back directly. numpy only (a
hard transitive dependency); no ``[bridge]`` extra needed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .base import MultiColumnBridge


def _parse_vector(value: Any) -> Optional[np.ndarray]:
    """Best-effort embedding parse: sequence, PG array, or pgvector text."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().strip("[]{}")
        if not stripped:
            return None
        value = [float(part) for part in stripped.split(",")]
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        return None
    return array


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmbeddingTrajectoryBridge(MultiColumnBridge):
    """Per-event novelty / drift / volatility over an embedding stream.

    ``compute()`` orders each entity's rows by ``(order_col, pk_col)`` and
    scores every event against the strictly-prior ones; rows with missing or
    unparseable embeddings get NULLs and do not enter any history.
    """

    VALUE_COLS = ("novelty", "drift", "volatility")

    def __init__(
        self,
        *,
        pk_col: str,
        fk_col: str,
        order_col: str,
        embedding_col: str,
        name: str = "embedding_trajectory",
    ) -> None:
        super().__init__(name=name, value_cols=list(self.VALUE_COLS))
        self.pk_col = pk_col
        self.fk_col = fk_col
        self.order_col = order_col
        self.embedding_col = embedding_col

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Dict[str, Any]]:
        by_entity: Dict[Any, List[Dict[str, Any]]] = {}
        for row in rows:
            by_entity.setdefault(row.get(self.fk_col), []).append(row)

        out: Dict[Any, Dict[str, Any]] = {}
        for entity_rows in by_entity.values():
            ordered = sorted(
                entity_rows,
                key=lambda r: (
                    r.get(self.order_col) is None,  # missing order sorts last
                    r.get(self.order_col),
                    str(r.get(self.pk_col)),
                ),
            )
            history: List[np.ndarray] = []
            for row in ordered:
                embedding = _parse_vector(row.get(self.embedding_col))
                if embedding is None:
                    out[row[self.pk_col]] = dict.fromkeys(self.VALUE_COLS)
                    continue
                if not history:
                    out[row[self.pk_col]] = dict.fromkeys(self.VALUE_COLS)
                else:
                    centroid = np.mean(np.stack(history), axis=0)
                    out[row[self.pk_col]] = {
                        "novelty": 1.0 - max(_cosine(embedding, h) for h in history),
                        "drift": 1.0 - _cosine(embedding, centroid),
                        "volatility": 1.0 - _cosine(embedding, history[-1]),
                    }
                history.append(embedding)
        return out

    # Convenience mirror of the base signature for readers of the cookbook;
    # the base implementation already does the right thing.
    def score_sequence(
        self, embeddings: Sequence[Any]
    ) -> List[Dict[str, Optional[float]]]:
        """Score an in-memory embedding sequence (one entity, given order)."""
        rows = [
            {"pk": i, "fk": 0, "order": i, "emb": e} for i, e in enumerate(embeddings)
        ]
        bridge = EmbeddingTrajectoryBridge(
            pk_col="pk", fk_col="fk", order_col="order", embedding_col="emb"
        )
        values = bridge.compute(rows, fit_rows=rows)
        return [values[i] for i in range(len(embeddings))]
