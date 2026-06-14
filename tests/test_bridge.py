"""Unit tests for the φ-bridge contract (no database).

Covers the causal guard and the pure-Python Markov-surprisal exemplar's φ values
against a hand computation. The end-to-end materialize → SQL-spine handoff is in
``tests/integration/test_bridge.py``.
"""

from __future__ import annotations

import math
from datetime import date

import pytest

from featurizer.bridge import MarkovSurprisalBridge, assert_pre_t0


def test_assert_pre_t0_rejects_future_rows():
    rows = [
        {"id": 1, "ts": date(2020, 1, 1)},
        {"id": 2, "ts": date(2020, 6, 1)},
        {"id": 3, "ts": date(2021, 1, 1)},  # after the cutoff -> leak
    ]
    with pytest.raises(ValueError, match="causal boundary violated"):
        assert_pre_t0(rows, date(2020, 6, 1), "ts")


def test_assert_pre_t0_allows_knowable_rows():
    rows = [
        {"id": 1, "ts": date(2020, 1, 1)},
        {"id": 2, "ts": date(2020, 6, 1)},  # == cutoff is knowable (inclusive)
        {"id": 3, "ts": None},  # unknown timestamp is not a violation
    ]
    assert_pre_t0(rows, date(2020, 6, 1), "ts")  # does not raise


def test_markov_surprisal_matches_hand_computation():
    """Sequence A,B,A,C (one owner). Add-one smoothed, vocab size 3, 4 unigrams."""
    rows = [
        {"eid": 1, "owner": 1, "ts": 1, "state": "A"},
        {"eid": 2, "owner": 1, "ts": 2, "state": "B"},
        {"eid": 3, "owner": 1, "ts": 3, "state": "A"},
        {"eid": 4, "owner": 1, "ts": 4, "state": "C"},
    ]
    bridge = MarkovSurprisalBridge(
        pk_col="eid", fk_col="owner", order_col="ts", state_col="state"
    )
    phi = bridge.compute(rows, fit_rows=rows)

    # unigram counts A:2 B:1 C:1 (total 4), V=3; transitions A->{B,C}, B->{A}.
    assert math.isclose(phi[1], -math.log(3 / 7))  # first: unigram P(A)=(2+1)/(4+3)
    assert math.isclose(phi[2], -math.log(2 / 5))  # P(B|A)=(1+1)/(2+3)
    assert math.isclose(phi[3], -math.log(2 / 4))  # P(A|B)=(1+1)/(1+3)
    assert math.isclose(phi[4], -math.log(2 / 5))  # P(C|A)=(1+1)/(2+3)


def test_markov_surprisal_fit_excludes_future():
    """φ for the same scored rows differs when the model is fit on a smaller
    (pre-t₀) window — proving the fit set, not the scored set, drives φ."""
    rows = [
        {"eid": 1, "owner": 1, "ts": 1, "state": "A"},
        {"eid": 2, "owner": 1, "ts": 2, "state": "B"},
        {"eid": 3, "owner": 1, "ts": 3, "state": "A"},
        {"eid": 4, "owner": 1, "ts": 4, "state": "C"},
    ]
    bridge = MarkovSurprisalBridge(
        pk_col="eid", fk_col="owner", order_col="ts", state_col="state"
    )
    full = bridge.compute(rows, fit_rows=rows)
    partial = bridge.compute(rows, fit_rows=rows[:2])  # only A->B seen
    # The A->C transition is unseen in the partial fit, so e4 is more surprising.
    assert partial[4] > full[4]
