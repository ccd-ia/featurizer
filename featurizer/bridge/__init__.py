# coding: utf-8

"""The φ-bridge: precompute companion for non-SQL feature families.

The SQL spine handles everything expressible as point-in-time-correct SQL. The
φ-bridge covers the rest — features that need heavy Python — by precomputing a
value per source row (a model, when needed, fit on pre-t₀ rows only),
materializing it back into PostgreSQL as a column, and emitting a config fragment
so the spine aggregates it like any other ``Variable``. See :mod:`.base` for the
contract and the causal boundary (ADR-0001); the bridge is an orchestration-
agnostic library (ADR-0003) — wire it upstream of the SQL run in Dagster/Snakemake.

Four exemplars ship, one per modality:

- :class:`~featurizer.bridge.sequence.MarkovSurprisalBridge` — pure-Python Markov
  surprisal; the end-to-end reference (no optional deps).
- :class:`~featurizer.bridge.text.TfidfTopicShareBridge` — fitted TF-IDF/SVD topic
  share (scikit-learn).
- :class:`~featurizer.bridge.graph.PageRankBridge` — PageRank centrality
  (networkx).
- :class:`~featurizer.bridge.embeddings.SentenceEmbeddingBridge` — sentence
  embeddings → pgvector (sentence-transformers + the pgvector extension).

The optional dependencies live in the ``[bridge]`` extra
(``pip install 'featurizer[bridge]'``).

Remaining heavy families ship as *documented abstract interfaces only* — each is
a :class:`~featurizer.bridge.base.BridgeComputer` subclass implementing
``compute(rows, *, fit_rows) -> {pk: φ}``, following the four exemplars above:

  Text/NLP      named-entity counts, POS/dependency stats, sentiment, toxicity,
                language id, readability, keyphrase rates, coreference density.
  Embeddings    document/user/image embeddings, drift vs a reference centroid,
                novelty (1 - max cosine to history), cluster assignment, outlier
                score, nearest-prototype distance.
  Graph         betweenness / eigenvector / closeness centrality, community id
                and modularity, k-core number, triangle count, label propagation,
                temporal-motif counts, embedding (node2vec).
  Sequence      HMM state posterior, change-point score, motif/n-gram surprisal,
                edit distance to a prototype, survival/hazard estimates,
                periodicity (FFT peak), trend (STL) components.
  Geospatial    road-network travel time, isochrone population, POI density by
                category, trajectory stay-points, map-matched route features.

Implement one by subclassing ``BridgeComputer`` and adding its dependency to the
``[bridge]`` extra; no engine change is needed — the spine consumes the
materialized column.
"""

from __future__ import annotations

from .base import BridgeComputer, assert_pre_t0
from .embeddings import SentenceEmbeddingBridge
from .graph import PageRankBridge
from .sequence import MarkovSurprisalBridge
from .text import TfidfTopicShareBridge

__all__ = [
    "BridgeComputer",
    "assert_pre_t0",
    "MarkovSurprisalBridge",
    "TfidfTopicShareBridge",
    "PageRankBridge",
    "SentenceEmbeddingBridge",
]
