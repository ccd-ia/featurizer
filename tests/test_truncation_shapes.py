"""DB-free guard on the truncation-shape conclusion (63-byte identifier cap).

Tail-preserving truncation gets re-suggested every time someone meets a
``…kw_rod~978dcf98`` column, on the intuition that keeping the innermost
``alias.variable`` fragment is obviously friendlier to glob patterns. It is
not: measured, it trades ~200 lost variable names for ~1,200 lost *operator*
names and makes visual ambiguity worse.

These tests pin the ORDERING between the shapes, not the exact counts — the
counts move whenever the sample config's primitive set changes, but the
conclusion is a property of name structure and must not silently invert. If one
of these fails, the trade-off genuinely changed and
``.out-of-scope/tail-preserving-truncation.md`` needs revisiting rather than
the assertion needing relaxing.

Uses the same functions as ``benchmarks.truncation_shapes`` so the guard and
the report never drift. No database required.
"""

from __future__ import annotations

import pytest

from benchmarks.truncation_shapes import (
    both_ends,
    head_keep,
    score,
    tail_keep,
    truncating_labels,
)


@pytest.fixture(scope="module")
def labels() -> list[str]:
    out = truncating_labels()
    assert out, "sample config no longer produces any truncating names"
    return out


def test_truncation_is_not_an_edge_case(labels: list[str]) -> None:
    """The sample config truncates enough names for the comparison to mean something."""
    assert len(labels) > 100


def test_tail_preserving_is_a_regression(labels: list[str]) -> None:
    """The proposed shape recovers variable names by erasing operator names."""
    head = score(labels, head_keep)
    tail = score(labels, tail_keep)

    # It does what it claims: the innermost variable always survives.
    assert tail["var_lost"] == 0
    assert head["var_lost"] > 0

    # But it erases the operator stack instead, and far more often than
    # head-keeping ever erased variables. This is the whole objection.
    assert tail["op_lost"] > head["var_lost"], (
        "tail-preserving no longer trades more operator names than the variable "
        "names it recovers — the core objection may not hold any more"
    )
    assert head["op_lost"] == 0

    # And it makes the glob-friendliness it was proposed to improve *worse*.
    assert tail["ambiguous"] > head["ambiguous"]
    assert tail["worst"] > head["worst"]


def test_both_ends_dominates_but_does_not_retire_the_manifest(
    labels: list[str],
) -> None:
    """The steelman beats the status quo on every axis — and still isn't enough."""
    head = score(labels, head_keep)
    both = score(labels, both_ends)

    assert both["var_lost"] == 0 and both["op_lost"] == 0
    assert both["ambiguous"] < head["ambiguous"]
    assert both["worst"] <= head["worst"]

    # The reason it still doesn't justify a universal rename: physical names
    # remain an unsafe parse surface even under the best shape.
    assert both["ambiguous"] > 0, (
        "both-ends now leaves physical names unambiguous — if that holds on real "
        "configs, the rejection in .out-of-scope/ is worth re-opening"
    )


def test_every_shape_stays_within_the_identifier_cap(labels: list[str]) -> None:
    """Whatever the shape, the output must still be a legal PG identifier."""
    for shape in (head_keep, tail_keep, both_ends):
        for raw in labels[:200]:
            assert len(shape(raw).encode()) <= 63, f"{shape.__name__} exceeded 63 bytes"
