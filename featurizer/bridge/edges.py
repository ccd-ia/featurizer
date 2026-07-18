# coding: utf-8

"""Path 2 — text induces the graph: edge builders feeding the graph bridges.

These bridges do not emit per-row φ values; they emit an **edge table**
``(src, dst, ts)`` that is the *input* to
:class:`~featurizer.bridge.centrality.CentralityBridge` /
:class:`~featurizer.bridge.community.CommunityBridge` (or the native
``graph_relationships`` planner stage). The two-stage wiring:

1. ``NearDuplicateEdgeBridge(...).materialize_edges(conn, source_table="docs",
   output_table="text_edges", ...)``
2. ``CentralityBridge(source_col="src", target_col="dst").materialize_snapshots(
   conn, source_table="text_edges", causal_col="ts", ...)``

Causality lives on the edge timestamp: an induced edge carries the moment it
became knowable (for a near-duplicate pair, the *later* document's timestamp
— the pair does not exist until the copy appears), and every downstream
consumer applies its own ``<= as_of`` bound to it. No model is fitted here.

- :class:`NearDuplicateEdgeBridge` — MinHash/LSH (datasketch, ``[bridge]``
  extra) over word shingles: an edge between the *entities* of two documents
  whose estimated Jaccard similarity clears ``threshold`` — the copy-paste
  signature of coordination. Same-entity duplicates are skipped (self-copies
  are not coordination); repeated duplicate pairs yield repeated edges (the
  graph bridges count them as weight).
- :class:`CoMentionEdgeBridge` — an edge between every pair of names
  mentioned together in one document. The default extractor is a deliberately
  naive capitalized-sequence heuristic (dependency-free, sentence-initial
  words included); pass ``extract=`` for a real NER-based extractor.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .base import create_table_sql, load_rows, value_sql_type
from .nlp import word_tokens

#: An induced edge: (src, dst, knowable-at timestamp).
Edge = Tuple[Any, Any, Any]


class EdgeBridge(ABC):
    """Base for bridges whose output is an edge table, not a value column."""

    def __init__(self, *, name: str) -> None:
        self.name = name

    @abstractmethod
    def compute_edges(self, rows: List[Dict[str, Any]]) -> List[Edge]:
        """Return the induced ``(src, dst, ts)`` edges for the source rows."""

    def materialize_edges(
        self,
        conn: Any,
        *,
        source_table: str,
        output_table: str,
        content_cols: Sequence[str],
        src_col: str = "src",
        dst_col: str = "dst",
        ts_col: str = "ts",
        persist: bool = False,
    ) -> str:
        """Read ``content_cols`` from ``source_table``, write the edge table.

        The output is ``(src_col, dst_col, ts_col)`` — exactly the shape the
        graph bridges and the native ``graph_relationships`` block consume.
        ``persist=True`` writes a real table (ADR-0003/0014 semantics).
        """
        rows = load_rows(conn, source_table, list(content_cols))
        edges = sorted(self.compute_edges(rows), key=lambda e: tuple(map(str, e)))

        src_type = value_sql_type([e[0] for e in edges])
        dst_type = value_sql_type([e[1] for e in edges])
        ts_type = value_sql_type([e[2] for e in edges])
        with conn.cursor() as cur:
            cur.execute(
                create_table_sql(
                    output_table,
                    f"{src_col} {src_type}, {dst_col} {dst_type}, "
                    f"{ts_col} {ts_type}",
                    persist,
                )
            )
            cur.executemany(f"insert into {output_table} values (%s, %s, %s)", edges)
        return output_table


class NearDuplicateEdgeBridge(EdgeBridge):
    """Entity pairs sharing near-duplicate text (MinHash/LSH).

    Documents are shingled into word ``shingle_size``-grams; a MinHash
    signature per document goes into an LSH index at ``threshold``; candidate
    pairs are verified against the estimated Jaccard before an edge is
    emitted. Deterministic (fixed MinHash seed).
    """

    def __init__(
        self,
        *,
        pk_col: str,
        entity_col: str,
        text_col: str,
        ts_col: str,
        threshold: float = 0.8,
        num_perm: int = 128,
        shingle_size: int = 3,
        name: str = "near_duplicate_edges",
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"{name}: threshold must be in (0, 1]")
        super().__init__(name=name)
        self.pk_col = pk_col
        self.entity_col = entity_col
        self.text_col = text_col
        self.ts_col = ts_col
        self.threshold = threshold
        self.num_perm = num_perm
        self.shingle_size = shingle_size

    def _shingles(self, text: str) -> set[str]:
        tokens = word_tokens(text)
        if not tokens:
            return set()
        if len(tokens) < self.shingle_size:
            return {" ".join(tokens)}
        return {
            " ".join(tokens[i : i + self.shingle_size])
            for i in range(len(tokens) - self.shingle_size + 1)
        }

    def compute_edges(self, rows: List[Dict[str, Any]]) -> List[Edge]:
        try:
            from datasketch import (  # pyright: ignore[reportMissingImports]
                MinHash,
                MinHashLSH,
            )
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise ImportError(
                "NearDuplicateEdgeBridge needs datasketch: "
                "install with `pip install 'featurizer[bridge]'`."
            ) from exc

        docs: List[Tuple[Dict[str, Any], Any]] = []
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        for i, row in enumerate(rows):
            shingles = self._shingles(str(row.get(self.text_col) or ""))
            if not shingles:
                continue
            mh = MinHash(num_perm=self.num_perm)
            for shingle in shingles:
                mh.update(shingle.encode("utf8"))
            key = str(len(docs))
            lsh.insert(key, mh)
            docs.append((row, mh))

        edges: List[Edge] = []
        for j, (row_j, mh_j) in enumerate(docs):
            for key in lsh.query(mh_j):
                i = int(str(key))
                if i >= j:  # each unordered doc pair once
                    continue
                row_i, mh_i = docs[i]
                if mh_i.jaccard(mh_j) < self.threshold:
                    continue  # LSH candidate that fails verification
                a, b = row_i.get(self.entity_col), row_j.get(self.entity_col)
                if a is None or b is None or a == b:
                    continue  # self-copies are not coordination
                src, dst = sorted((a, b), key=str)
                ts_i, ts_j = row_i.get(self.ts_col), row_j.get(self.ts_col)
                # Knowable when the LATER of the pair appears.
                ts = max(
                    (t for t in (ts_i, ts_j) if t is not None),
                    default=None,
                )
                edges.append((src, dst, ts))
        return edges


#: Naive mention heuristic: runs of capitalized words (Spanish letters
#: included). Sentence-initial words match too — a documented trade-off of
#: staying dependency-free; pass ``extract=`` for a real extractor.
_MENTION_RE = re.compile(
    r"(?:[A-ZÁÉÍÓÚÜÑ][\wáéíóúüñ]+)(?:\s+[A-ZÁÉÍÓÚÜÑ][\wáéíóúüñ]+)*"
)


def _default_extract(text: str) -> List[str]:
    return _MENTION_RE.findall(text)


class CoMentionEdgeBridge(EdgeBridge):
    """An edge between every pair of names co-mentioned in one document."""

    def __init__(
        self,
        *,
        text_col: str,
        ts_col: str,
        extract: Optional[Callable[[str], List[str]]] = None,
        name: str = "co_mention_edges",
    ) -> None:
        super().__init__(name=name)
        self.text_col = text_col
        self.ts_col = ts_col
        self.extract = extract or _default_extract

    def compute_edges(self, rows: List[Dict[str, Any]]) -> List[Edge]:
        edges: List[Edge] = []
        for row in rows:
            mentions = sorted(
                set(self.extract(str(row.get(self.text_col) or ""))), key=str
            )
            ts = row.get(self.ts_col)
            for i, a in enumerate(mentions):
                for b in mentions[i + 1 :]:
                    edges.append((a, b, ts))
        return edges
