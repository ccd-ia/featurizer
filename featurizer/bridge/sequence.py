# coding: utf-8

"""Sequence φ-bridge exemplar: Markov surprisal (pure Python, no extra deps).

For each event row, φ = the information content ``-ln P(state | previous state)``
under a first-order Markov model fit on the pre-t₀ sequences (add-one smoothed).
The first event of a sequence uses the unigram ``-ln P(state)``. High surprisal
flags a state transition the model rarely saw — a cheap, leakage-safe novelty
signal the SQL spine then aggregates (mean/max) over the parent entity.

This exemplar needs no optional dependency, so it is the end-to-end test of the
bridge contract on a plain PostgreSQL.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Dict, List

from .base import BridgeComputer


class MarkovSurprisalBridge(BridgeComputer):
    def __init__(
        self,
        *,
        pk_col: str,
        fk_col: str,
        order_col: str,
        state_col: str,
        name: str = "markov_surprisal",
        value_col: str = "markov_surprisal",
    ) -> None:
        super().__init__(name=name, value_col=value_col, value_type="numeric")
        self.pk_col = pk_col
        self.fk_col = fk_col
        self.order_col = order_col
        self.state_col = state_col

    def _sequences(self, rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
        seqs: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for row in rows:
            seqs[row[self.fk_col]].append(row)
        for owner in seqs:
            seqs[owner].sort(key=lambda r: r[self.order_col])
        return seqs

    def compute(
        self, rows: List[Dict[str, Any]], *, fit_rows: List[Dict[str, Any]]
    ) -> Dict[Any, float]:
        # Fit transition and unigram counts on the pre-t₀ sequences.
        transitions: Dict[Any, Counter[Any]] = defaultdict(Counter)
        unigram: Counter[Any] = Counter()
        vocab: set[Any] = set()
        for seq in self._sequences(fit_rows).values():
            states = [r[self.state_col] for r in seq]
            for state in states:
                vocab.add(state)
                unigram[state] += 1
            for prev, cur in zip(states, states[1:]):
                transitions[prev][cur] += 1

        v = max(len(vocab), 1)
        total_unigram = sum(unigram.values())

        def p_unigram(state: Any) -> float:
            return (unigram.get(state, 0) + 1) / (total_unigram + v)

        def p_transition(prev: Any, cur: Any) -> float:
            row = transitions.get(prev)
            denom = (sum(row.values()) if row else 0) + v
            num = (row.get(cur, 0) if row else 0) + 1
            return num / denom

        out: Dict[Any, float] = {}
        for seq in self._sequences(rows).values():
            prev: Any = None
            for row in seq:
                state = row[self.state_col]
                prob = p_unigram(state) if prev is None else p_transition(prev, state)
                out[row[self.pk_col]] = -math.log(prob)
                prev = state
        return out
