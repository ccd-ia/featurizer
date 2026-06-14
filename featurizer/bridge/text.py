# coding: utf-8

"""Text φ-bridge exemplar: a fitted topic share (scikit-learn, [bridge] extra).

φ per row = the row's loading on the first latent topic of a TF-IDF + Truncated
SVD model **fit on the pre-t₀ documents only** — so the learned topic space never
sees the future. The per-row transform reads only that row's own text; the SQL
spine then aggregates the share over the parent entity with its causal bound.

scikit-learn is an optional dependency (``pip install 'featurizer[bridge]'``);
:meth:`compute` raises a clear error if it is missing.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BridgeComputer


class TfidfTopicShareBridge(BridgeComputer):
    def __init__(
        self,
        *,
        pk_col: str,
        text_col: str,
        name: str = "tfidf_topic_share",
        value_col: str = "tfidf_topic_share",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="numeric")
        self.pk_col = pk_col
        self.text_col = text_col

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, Any]:
        try:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.pipeline import make_pipeline
        except ImportError as exc:  # pragma: no cover - exercised only without extra
            raise ImportError(
                "TfidfTopicShareBridge needs scikit-learn: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        fit_text = [str(r.get(self.text_col) or "") for r in fit_rows]
        if not any(fit_text):
            return {r[self.pk_col]: None for r in rows}

        model = make_pipeline(
            TfidfVectorizer(min_df=1, stop_words="english"),
            TruncatedSVD(n_components=1, random_state=0),
        )
        model.fit(fit_text)
        all_text = [str(r.get(self.text_col) or "") for r in rows]
        loadings = model.transform(all_text)[:, 0]
        return {row[self.pk_col]: float(load) for row, load in zip(rows, loadings)}
