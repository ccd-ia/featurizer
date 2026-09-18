# coding: utf-8

"""The synth CTE projects declared names as delimited identifiers (issue #13).

A target's own variables reach the planner exactly as the config wrote them,
while generated features (aggregates, as-of pulls) arrive already quoted.
Emitting the first group bare is invisible for as long as every declared name
is a plain identifier — and a syntax error the moment one is not, which is
what a wide relation of *featurizer's own output* is.

The reported failure, from a target whose columns are a previous run's
feature names::

    syntax error at or near "("
    LINE 21: ... , MEAN(away_goalie_games.ev_save_pct) as MEAN(away_goal...

PostgreSQL reads ``MEAN(...)`` as an aggregate call over a column that does
not exist, then chokes on the alias.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from featurizer import Featurizer
from featurizer.primitives.abstractions import pg_identifier, quote_if_bare


def _config(tmp_path: Path, variable: str) -> Path:
    """A one-entity config whose single declared variable is ``variable``."""
    path = tmp_path / "config.yaml"
    path.write_text(
        "target: wide\n"
        "max_depth: 1\n"
        "intervals: []\n"
        "aggregations: []\n"
        "transformations:\n"
        "  - identity\n"
        "entities:\n"
        "  - alias: wide\n"
        "    id: entity_id\n"
        "    table: wide\n"
        "    temporal_ix: as_of\n"
        "    variables:\n"
        f"      {json.dumps(variable)}:\n"
        "        type: numeric\n"
    )
    return path


# --------------------------------------------------------------- the helper


@pytest.mark.parametrize(
    "raw",
    [
        "age",
        "MEAN(x.y|interval=P30D)",
        "a b",
        "select",
        "Mixed_Case",
    ],
)
def test_bare_names_are_delimited(raw: str) -> None:
    assert quote_if_bare(raw) == f'"{raw}"'


def test_already_delimited_names_pass_through(raw: str = '"ABS(care.risk)"') -> None:
    """Generated features arrive quoted; re-quoting them must be a no-op."""
    assert quote_if_bare(raw) == raw


def test_embedded_quotes_are_doubled_not_stripped() -> None:
    """The database owns a declared name, so it survives byte for byte.

    ``pg_identifier`` drops the quote instead, which is correct for a name
    featurizer generates and wrong for one it was handed.
    """
    assert quote_if_bare('he"llo') == '"he""llo"'
    assert pg_identifier('he"llo') == '"hello"'


def test_long_declared_names_are_not_hash_capped() -> None:
    """PostgreSQL truncates at 63 bytes; it does not hash.

    ``pg_identifier`` caps a long *generated* name with a stable hash suffix so
    two long names cannot collide. Doing that to a declared name would produce
    an identifier naming no column at all.
    """
    long = "x" * 70
    assert quote_if_bare(long) == f'"{long}"'
    assert "~" in pg_identifier(long)


# ------------------------------------------------------------- the rendering


def test_declared_name_that_looks_like_an_aggregate_call_renders(tmp_path) -> None:
    """The regression: this raises a SyntaxError-shaped SQL string before the fix."""
    config = _config(tmp_path, "MEAN(games.goals|interval=P30D)")
    query = Featurizer(str(config)).query
    assert '"MEAN(games.goals|interval=P30D)"' in query
    # The bare form must not survive anywhere in the projection: that spelling
    # is what PostgreSQL parses as a function call.
    assert " MEAN(games.goals|interval=P30D)" not in query


def test_ordinary_names_are_quoted_without_changing_output_names(tmp_path) -> None:
    """Quoting changes how a column is written, never what it is called.

    ``select t.age`` and ``select t."age"`` both yield a column named ``age``,
    so no existing feature matrix moves.
    """
    config = _config(tmp_path, "age")
    query = Featurizer(str(config)).query
    # The transform aliases the column to itself, so the output column is
    # still called `age` — quoting is invisible downstream.
    assert '"age" as "age"' in query
    assert " age," not in query and " age " not in query, "no bare reference left"


def test_identifier_columns_are_quoted(tmp_path) -> None:
    """The id and temporal index are declared names too, in both CTEs."""
    config = _config(tmp_path, "age")
    query = Featurizer(str(config)).query
    # synth reads them off the base table ...
    assert 'wide."entity_id"' in query
    assert 'wide."as_of"' in query
    # ... and transform projects them on out of synth.
    assert '"entity_id", "as_of"' in query
