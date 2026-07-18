# coding: utf-8

"""The φ-bridge: precompute companion for non-SQL feature families.

The SQL spine handles everything expressible as point-in-time-correct SQL. The
φ-bridge covers the rest — features that need heavy Python — by precomputing a
value per source row (a model, when needed, fit on pre-t₀ rows only),
materializing it back into PostgreSQL as a column, and emitting a config fragment
so the spine aggregates it like any other ``Variable``. See :mod:`.base` for the
contract and the causal boundary (ADR-0001); the bridge is an orchestration-
agnostic library (ADR-0003) — wire it upstream of the SQL run in Dagster/Snakemake.

Shipped bridges by modality:

- :class:`~featurizer.bridge.sequence.MarkovSurprisalBridge` — pure-Python Markov
  surprisal; the end-to-end reference (no optional deps).
- :class:`~featurizer.bridge.text.TfidfTopicShareBridge` — fitted TF-IDF/SVD topic
  share (scikit-learn).
- Text Path 1 reductions (:mod:`.nlp`): :class:`SentimentBridge`,
  :class:`ReadabilityBridge`, :class:`LanguageIdBridge` (all dependency-free)
  and :class:`NERCountsBridge` (spaCy, multi-column).
- :class:`~featurizer.bridge.graph.PageRankBridge` — PageRank centrality
  (networkx).
- :class:`~featurizer.bridge.centrality.CentralityBridge` — multi-metric
  centrality profile, cheap tier default / heavy opt-in, snapshot-aware
  (networkx).
- :class:`~featurizer.bridge.community.CommunityBridge` — Louvain membership
  (categorical) + modularity (python-louvain).
- :class:`~featurizer.bridge.embeddings.SentenceEmbeddingBridge` — sentence
  embeddings → pgvector (sentence-transformers + the pgvector extension).
- :class:`~featurizer.bridge.trajectory.EmbeddingTrajectoryBridge` — per-event
  novelty / drift / volatility over an embedding stream (numpy only).
- Sequence extensions (:mod:`.changepoint`): :class:`ChangePointBridge` and
  :class:`PeriodicityBridge` — per-entity mean-shift and FFT-peak scores over
  the pre-t₀ event series (numpy only, snapshot-aware).
- Text-induced edges (:mod:`.edges`, Path 2):
  :class:`NearDuplicateEdgeBridge` (MinHash/LSH, datasketch) and
  :class:`CoMentionEdgeBridge` — emit an ``(src, dst, ts)`` edge table that
  feeds the graph bridges or the native ``graph_relationships`` stage.

The optional dependencies live in the ``[bridge]`` extra
(``pip install 'featurizer[bridge]'``).

Remaining heavy families ship as *documented abstract interfaces only* — each is
a :class:`~featurizer.bridge.base.BridgeComputer` (or
:class:`~featurizer.bridge.base.MultiColumnBridge`) subclass implementing
``compute(rows, *, fit_rows)``, following the shipped bridges above:

  Text/NLP      POS/dependency stats, toxicity, keyphrase rates, coreference
                density, LLM structured extraction.
  Embeddings    cluster assignment, outlier score, nearest-prototype distance.
  Graph         SBM block membership / MDL surprise (graph-tool; deliberately
                deferred — not pip-installable), temporal-motif counts,
                embedding (node2vec).
  Sequence      HMM state posterior, motif/n-gram surprisal, edit distance to
                a prototype, survival/hazard estimates, trend (STL) components.
  Geospatial    road-network travel time, isochrone population, POI density by
                category, trajectory stay-points, map-matched route features.

Implement one by subclassing ``BridgeComputer`` and adding its dependency to the
``[bridge]`` extra; no engine change is needed — the spine consumes the
materialized column.
"""

from __future__ import annotations

from .base import BridgeComputer, MultiColumnBridge, assert_pre_t0
from .centrality import CentralityBridge
from .changepoint import ChangePointBridge, PeriodicityBridge
from .community import CommunityBridge
from .edges import CoMentionEdgeBridge, EdgeBridge, NearDuplicateEdgeBridge
from .embeddings import SentenceEmbeddingBridge
from .graph import PageRankBridge
from .nlp import (
    LanguageIdBridge,
    NERCountsBridge,
    ReadabilityBridge,
    SentimentBridge,
)
from .sequence import MarkovSurprisalBridge
from .text import TfidfTopicShareBridge
from .trajectory import EmbeddingTrajectoryBridge

__all__ = [
    "BridgeComputer",
    "MultiColumnBridge",
    "assert_pre_t0",
    "MarkovSurprisalBridge",
    "TfidfTopicShareBridge",
    "SentimentBridge",
    "ReadabilityBridge",
    "LanguageIdBridge",
    "NERCountsBridge",
    "PageRankBridge",
    "CentralityBridge",
    "CommunityBridge",
    "SentenceEmbeddingBridge",
    "EmbeddingTrajectoryBridge",
    "ChangePointBridge",
    "PeriodicityBridge",
    "EdgeBridge",
    "NearDuplicateEdgeBridge",
    "CoMentionEdgeBridge",
]
