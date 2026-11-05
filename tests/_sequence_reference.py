"""A pure-Python reference of the 11 sequence primitives under ADR-0020.

ADR-0020: a NULL is a state. Every function here reads a series that is
already windowed (the rows the as-of bound and the interval let through) and
already in timeline order (the temporal index, then the ADR-0018 tiebreak), as
a list of ``(day, value)`` pairs where ``value`` may be ``None``. A transition
is a consecutive pair from the second row on, a ``None`` on either side
included: "no previous row" is the position, never a ``None`` value.

The functions return what the SQL returns on the same rows, including the
empty cases the rule does not touch: ``None`` where the SQL aggregate has no
input (an empty transition matrix, no complete n-gram, no rows), and a count
of 0 where the engine keeps a row with no transitions.

``tests/integration/test_null_state.py`` compares the engine with these
functions cell by cell; ``tests/test_sequence_reference.py`` pins them on hand
cases.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from datetime import date
from typing import Optional

Series = list[tuple[date, Optional[str]]]


def _values(rows: Series) -> list[str | None]:
    return [value for _, value in rows]


def _pairs(rows: Series) -> list[tuple[str | None, str | None]]:
    values = _values(rows)
    return list(zip(values, values[1:]))


def _matrix(rows: Series) -> tuple[Counter, Counter, int]:
    """Pair counts, row totals (by the previous state) and the joint total."""
    pairs = Counter(_pairs(rows))
    row_totals: Counter = Counter()
    for (prev, _), freq in pairs.items():
        row_totals[prev] += freq
    return pairs, row_totals, sum(pairs.values())


def sequence_entropy(rows: Series, as_of: date) -> float | None:
    """Joint entropy of the (prev, curr) pairs, in nats."""
    pairs, _, total = _matrix(rows)
    if not total:
        return None
    return -sum((f / total) * math.log(f / total) for f in pairs.values())


def markov_conditional_entropy(rows: Series, as_of: date) -> float | None:
    """H(X_t | X_{t-1}): ``-sum p(i,j) ln p(j|i)``, in nats."""
    pairs, row_totals, total = _matrix(rows)
    if not total:
        return None
    return -sum(
        (f / total) * math.log(f / row_totals[prev]) for (prev, _), f in pairs.items()
    )


def max_transition_prob(rows: Series, as_of: date) -> float | None:
    """The largest conditional transition probability ``p(j|i)``."""
    pairs, row_totals, total = _matrix(rows)
    if not total:
        return None
    return max(f / row_totals[prev] for (prev, _), f in pairs.items())


def state_volatility(rows: Series, as_of: date) -> int | None:
    """Pairs whose two states differ, ``None`` and a value differing."""
    if not rows:
        return None
    return sum(1 for prev, curr in _pairs(rows) if prev != curr)


def transition_matrix_summary(rows: Series, as_of: date) -> int | None:
    """Distinct (prev, curr) pairs, ``None`` equal to ``None``."""
    if not rows:
        return None
    return len(set(_pairs(rows)))


def rework_count(rows: Series, as_of: date) -> int | None:
    """Pairs that repeat a state, a ``None`` after a ``None`` included."""
    if not rows:
        return None
    return sum(1 for prev, curr in _pairs(rows) if prev == curr)


def _ngram_freq(rows: Series, n: int) -> int | None:
    values = _values(rows)
    grams = Counter(tuple(values[i : i + n]) for i in range(len(values) - n + 1))
    return max(grams.values()) if grams else None


def ngram_2_freq(rows: Series, as_of: date) -> int | None:
    """The count of the most common complete bigram."""
    return _ngram_freq(rows, 2)


def ngram_3_freq(rows: Series, as_of: date) -> int | None:
    """The count of the most common complete trigram: a series of two rows
    has none."""
    return _ngram_freq(rows, 3)


def longest_streak(rows: Series, as_of: date) -> int | None:
    """The longest run of one state, a run of ``None`` included."""
    if not rows:
        return None
    best = run = 0
    previous: object = object()  # equal to no value, ``None`` included
    for value in _values(rows):
        run = run + 1 if value == previous else 1
        best = max(best, run)
        previous = value
    return best


def time_in_current_state(rows: Series, as_of: date) -> int | None:
    """Days from the start of the current run to the as-of date. The first
    row starts a run, so an all-``None`` window dwells since its first row."""
    if not rows:
        return None
    starts = [
        day for i, (day, value) in enumerate(rows) if i == 0 or value != rows[i - 1][1]
    ]
    return (as_of - max(starts)).days


def recurrence_interval(rows: Series, as_of: date) -> float | None:
    """Mean day gap between consecutive rows of the same state, ``None`` a
    state."""
    last: dict[str | None, date] = {}
    gaps: list[int] = []
    for day, value in rows:
        if value in last:
            gaps.append((day - last[value]).days)
        last[value] = day
    return sum(gaps) / len(gaps) if gaps else None


REFERENCE: dict[str, Callable[[Series, date], float | int | None]] = {
    "sequence_entropy": sequence_entropy,
    "markov_conditional_entropy": markov_conditional_entropy,
    "max_transition_prob": max_transition_prob,
    "state_volatility": state_volatility,
    "transition_matrix_summary": transition_matrix_summary,
    "rework_count": rework_count,
    "ngram_2_freq": ngram_2_freq,
    "ngram_3_freq": ngram_3_freq,
    "longest_streak": longest_streak,
    "time_in_current_state": time_in_current_state,
    "recurrence_interval": recurrence_interval,
}
