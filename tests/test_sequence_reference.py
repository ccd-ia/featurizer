"""The ADR-0020 reference is right on sequences small enough to check by hand.

``tests/_sequence_reference.py`` is the oracle the NULL-state sweep compares
the engine with, so it is checked here first, DB-free, against values written
out by hand. N is a NULL value.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import pytest

from tests._sequence_reference import REFERENCE, Series

AS_OF = date(2024, 2, 1)
DAY = [date(2024, 1, 1) + timedelta(days=10 * i) for i in range(6)]  # 01-01, 01-11…
N = None


def _series(*values: Optional[str]) -> Series:
    return list(zip(DAY, values))


def _check(name: str, rows: Series, expected: object) -> None:
    actual = REFERENCE[name](rows, AS_OF)
    if isinstance(expected, float):
        assert actual is not None and math.isclose(actual, expected), (name, actual)
    else:
        assert actual == expected, (name, actual)


# The days since AS_OF of each DAY: 31, 21, 11, 1, -9, -19.
CASES = {
    # a -> N and N -> b are both transitions.
    "a,N,b": (
        _series("a", N, "b"),
        {
            "sequence_entropy": math.log(2),
            "markov_conditional_entropy": 0.0,
            "max_transition_prob": 1.0,
            "state_volatility": 2,
            "transition_matrix_summary": 2,
            "rework_count": 0,
            "ngram_2_freq": 1,
            "ngram_3_freq": 1,
            "longest_streak": 1,
            "time_in_current_state": 11,
            "recurrence_interval": None,
        },
    ),
    # N -> N repeats a state; the issue's example (rework 1, volatility 2).
    "a,N,N,a": (
        _series("a", N, N, "a"),
        {
            "sequence_entropy": math.log(3),
            "markov_conditional_entropy": (2 / 3) * math.log(2),
            "max_transition_prob": 1.0,
            "state_volatility": 2,
            "transition_matrix_summary": 3,
            "rework_count": 1,
            "ngram_2_freq": 1,
            "ngram_3_freq": 1,
            "longest_streak": 2,
            "time_in_current_state": 1,
            "recurrence_interval": 20.0,  # N: 10 days, a: 30 days
        },
    ),
    # One state throughout: a defined dwell, entropy and streak.
    "N,N,N": (
        _series(N, N, N),
        {
            "sequence_entropy": 0.0,
            "markov_conditional_entropy": 0.0,
            "max_transition_prob": 1.0,
            "state_volatility": 0,
            "transition_matrix_summary": 1,
            "rework_count": 2,
            "ngram_2_freq": 2,
            "ngram_3_freq": 1,
            "longest_streak": 3,
            "time_in_current_state": 31,
            "recurrence_interval": 10.0,
        },
    ),
    # One row: no transition, a run of one.
    "a": (
        _series("a"),
        {
            "sequence_entropy": None,
            "markov_conditional_entropy": None,
            "max_transition_prob": None,
            "state_volatility": 0,
            "transition_matrix_summary": 0,
            "rework_count": 0,
            "ngram_2_freq": None,
            "ngram_3_freq": None,
            "longest_streak": 1,
            "time_in_current_state": 31,
            "recurrence_interval": None,
        },
    ),
    # Two rows hold no trigram.
    "a,b": (
        _series("a", "b"),
        {"ngram_2_freq": 1, "ngram_3_freq": None, "state_volatility": 1},
    ),
    # A NULL run at the start, then at the end.
    "N,N,a,b": (
        _series(N, N, "a", "b"),
        {
            "state_volatility": 2,
            "rework_count": 1,
            "transition_matrix_summary": 3,
            "longest_streak": 2,
            "time_in_current_state": 1,
            "ngram_3_freq": 1,
        },
    ),
    "a,b,N,N": (
        _series("a", "b", N, N),
        {
            "state_volatility": 2,
            "rework_count": 1,
            "transition_matrix_summary": 3,
            "longest_streak": 2,
            "time_in_current_state": 11,
            "max_transition_prob": 1.0,
        },
    ),
    # No rows at all: nothing to reduce.
    "empty": (
        [],
        {name: None for name in REFERENCE},
    ),
}


@pytest.mark.parametrize(
    ("case", "name"),
    [(case, name) for case, (_, expected) in CASES.items() for name in expected],
)
def test_the_reference_gives_the_hand_checked_value(case: str, name: str) -> None:
    rows, expected = CASES[case]
    _check(name, rows, expected[name])


def test_the_reference_covers_the_eleven_sequence_primitives() -> None:
    assert len(REFERENCE) == 11
