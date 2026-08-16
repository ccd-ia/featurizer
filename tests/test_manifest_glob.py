"""DB-free tests for manifest label globbing (``columns_matching`` and friends).

Driven by the real sample config rather than hand-written toys: 95% of genuine
labels contain ``_`` and 58% are hash-truncated, and both facts are exactly what
the implementation has to get right.
"""

from __future__ import annotations

import pytest

from featurizer import Featurizer
from featurizer.manifest import (
    LIKE_ESCAPE,
    filter_manifest,
    glob_to_like,
)

CONFIG = "featurizer/featurizer.yaml"


@pytest.fixture(scope="module")
def featurizer() -> Featurizer:
    return Featurizer(CONFIG, validate=False)


@pytest.fixture(scope="module")
def entries(featurizer: Featurizer):
    manifest = featurizer.feature_manifest
    assert manifest, "sample config produced an empty manifest"
    return manifest


# --------------------------------------------------------------------------- #
# The motivating case
# --------------------------------------------------------------------------- #


def test_the_motivating_case(featurizer: Featurizer, entries) -> None:
    """Globbing the label finds columns globbing the physical name cannot.

    The numbers are the ones recorded in the FAQ and in
    ``.out-of-scope/tail-preserving-truncation.md``. If this test fails, the
    helper has stopped solving the thing it was built for.
    """
    from fnmatch import fnmatchcase

    pattern = "*frecuencia_cardiaca*"
    by_label = featurizer.columns_matching(pattern)
    by_column = [e.column for e in entries if fnmatchcase(e.column, pattern)]

    assert len(by_label) == 672, f"expected 672 label matches, got {len(by_label)}"
    assert len(by_column) == 198, f"expected 198 column matches, got {len(by_column)}"
    # Everything the physical-name glob finds, the label glob also finds.
    assert set(by_column) <= set(by_label)


def test_truncated_column_is_reachable(featurizer: Featurizer) -> None:
    """A column whose readable tail was capped is still selectable by label."""
    columns = featurizer.columns_matching("*frecuencia_cardiaca*")
    assert any("~" in c for c in columns), "no truncated column in the result"


# --------------------------------------------------------------------------- #
# Glob semantics
# --------------------------------------------------------------------------- #


def test_order_follows_the_manifest(featurizer: Featurizer, entries) -> None:
    """Results keep output order — callers build select lists from them."""
    matched = featurizer.columns_matching("*plan_score*")
    manifest_order = [e.column for e in entries if e.column in set(matched)]
    assert matched == manifest_order


def test_literal_underscore(entries) -> None:
    """``_`` is a literal in a glob — the 95%-of-labels trap.

    ``care_plans`` must not be matched by a pattern that only differs in the
    underscore position, which is what would happen if the pattern leaked into
    a context treating ``_`` as a wildcard.
    """
    real = filter_manifest(entries, "*care_plans.*")
    assert real, "expected matches for the care_plans alias"
    fake = filter_manifest(entries, "*careXplans.*", allow_empty=True)
    assert fake == []


def test_matching_is_case_sensitive(entries) -> None:
    """fnmatchcase, not fnmatch — no platform-dependent case folding."""
    assert filter_manifest(entries, "*MEAN(*")
    assert filter_manifest(entries, "*mean(*", allow_empty=True) == []


def test_question_mark_is_single_char(entries) -> None:
    hits = filter_manifest(entries, "P?W_unmatched_sentinel", allow_empty=True)
    assert hits == []
    # A single-character wildcard against a known interval token.
    assert filter_manifest(entries, "*|interval=P?W)*", allow_empty=True)


def test_columns_matching_projects_manifest_matching(featurizer: Featurizer) -> None:
    """One matching path, not two."""
    pattern = "*plan_score*"
    assert featurizer.columns_matching(pattern) == [
        e.column for e in featurizer.manifest_matching(pattern)
    ]


def test_manifest_matching_returns_entries(featurizer: Featurizer) -> None:
    entries = featurizer.manifest_matching("*plan_score*")
    assert entries and all(e.label for e in entries)
    assert all("plan_score" in e.label for e in entries)


# --------------------------------------------------------------------------- #
# translator
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("glob", "expected"),
    [
        # `_` is the whole point: literal in glob, wildcard in LIKE.
        ("kw_rodent", r"kw\_rodent"),
        ("*(inspections.kw_*", r"%(inspections.kw\_%"),
        # `%` in the input is data, not a wildcard.
        ("100%_sure", r"100\%\_sure"),
        # the escape character itself
        ("back\\slash", r"back\\slash"),
        # wildcards map only after literals are escaped
        ("*", "%"),
        ("?", "_"),
        ("a*b?c_d", r"a%b_c\_d"),
        # no wildcards at all
        ("MEAN(x.y)", "MEAN(x.y)"),
    ],
)
def test_translator_escaping(glob: str, expected: str) -> None:
    like, escape = glob_to_like(glob)
    assert like == expected
    assert escape == LIKE_ESCAPE


def test_translator_order_is_not_self_escaping() -> None:
    """A freshly-introduced wildcard must not itself get escaped.

    The failure this guards: escaping `_` *after* mapping `?`→`_` would turn
    ``?`` into a literal underscore, silently changing the pattern's meaning.
    """
    like, _escape = glob_to_like("a?b")
    assert like == "a_b"  # a wildcard, not r"a\_b"
    like, _escape = glob_to_like("a*b")
    assert like == "a%b"


# --------------------------------------------------------------------------- #
# zero-match diagnostics
# --------------------------------------------------------------------------- #


def test_zero_match_raises_with_suggestion(featurizer: Featurizer) -> None:
    """A near-miss typo raises and the message names a real label."""
    with pytest.raises(LookupError) as excinfo:
        featurizer.columns_matching("*frecuencia_cardiaco*")
    message = str(excinfo.value)
    assert "matched 0 of" in message
    assert "Did you mean" in message
    assert "frecuencia_cardiaca" in message


def test_zero_match_mentions_allow_empty(featurizer: Featurizer) -> None:
    """The escape hatch has to be discoverable from the error itself."""
    with pytest.raises(LookupError, match="allow_empty=True"):
        featurizer.columns_matching("*definitely_absent_xyzzy*")


def test_no_suggestion_when_nothing_is_close(featurizer: Featurizer) -> None:
    """Still raises, invents nothing, and falls back to the truncation hint."""
    with pytest.raises(LookupError) as excinfo:
        featurizer.columns_matching("*qqqzzzwww*")
    message = str(excinfo.value)
    assert "matched 0 of" in message
    assert "Did you mean" not in message
    # With no suggestion to show, the caller gets the one piece of orienting
    # context available instead.
    assert "hash-truncated" in message


def test_truncation_hint_is_suppressed_when_redundant(featurizer: Featurizer) -> None:
    """Suggestions already show full labels — no need to also lecture."""
    with pytest.raises(LookupError) as excinfo:
        featurizer.columns_matching("*frecuencia_cardiaco*")
    message = str(excinfo.value)
    assert "Did you mean" in message
    assert "hash-truncated" not in message


def test_physical_name_pattern_explains_itself(featurizer: Featurizer, entries) -> None:
    """A pattern aimed at a truncated physical name says why it failed.

    This is the inverse mix-up: the caller holds a column name they can see in
    the output, globs it, and gets nothing because the helper matches labels.
    """
    truncated = next(e for e in entries if e.truncated)
    with pytest.raises(LookupError) as excinfo:
        featurizer.columns_matching(truncated.column)
    message = str(excinfo.value)
    assert "physical COLUMN name" in message
    assert "Glob the label instead" in message


def test_allow_empty_is_silent(featurizer: Featurizer) -> None:
    assert featurizer.columns_matching("*nothing_here*", allow_empty=True) == []
    assert featurizer.manifest_matching("*nothing_here*", allow_empty=True) == []
