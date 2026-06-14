# coding: utf-8

"""Embeddings φ-bridge exemplar: sentence embeddings → pgvector ([bridge] extra).

φ per row = a sentence-transformer embedding of the row's text, materialized into
a pgvector column for downstream similarity / drift / novelty features. The model
is pretrained (no fit on this data), and the per-row transform reads only the
row's own text, so it is causally safe by construction; the SQL spine handles the
temporal cut at aggregation time.

Both sentence-transformers and the PostgreSQL ``pgvector`` extension are optional:
:meth:`compute` raises if the package is missing, and materializing a
``value_type="vector"`` column requires ``create extension vector`` on the target
database (the integration test skips when it is absent).
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BridgeComputer


class SentenceEmbeddingBridge(BridgeComputer):
    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        model_name: str = "all-MiniLM-L6-v2",
        name: str = "sentence_embedding",
        value_col: str = "sentence_embedding",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="vector")
        self.pk_col = pk_col
        self.text_col = text_col
        self.model_name = model_name

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Any]:
        try:
            from sentence_transformers import (  # pyright: ignore[reportMissingImports]
                SentenceTransformer,
            )
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "SentenceEmbeddingBridge needs sentence-transformers: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        model = SentenceTransformer(self.model_name)
        texts = [str(r.get(self.text_col) or "") for r in rows]
        vectors = model.encode(texts, normalize_embeddings=True)
        # pgvector accepts the '[1,2,3]' text representation on insert.
        return {
            row[self.pk_col]: "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
            for row, vec in zip(rows, vectors)
        }
