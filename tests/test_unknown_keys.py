# coding: utf-8

"""A key nothing reads is rejected, at all three levels (issues #7, #12).

The config is a contract with the planner, and until now three quarters of it
was unenforced: only keys inside a ``temporal:`` block were checked. A key
misspelt anywhere else was dropped in silence and the run continued with
different behaviour than its author had written down.

The silence has one cause in three places — the config dict is *cherry-picked*
rather than splatted, so no ``TypeError`` ever fires:

- top level: the loader reads the ten keys it knows
- relationship: ``ERGraph.__init__`` reads ``r["parent"]``, ``r.get("temporal")``…
- variable: ``Entity.__init__`` reads ``description["type"]``, ``.get("role")``…

Unlike ``Entity(**e)``, which has always raised on an unknown *entity* key.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from featurizer import validate_config

BASE = """
target: users
max_depth: 1
intervals: [P1M]
entities:
  - alias: users
    id: user_id
    table: users
    temporal_ix: created_at
    variables:
      age:
        type: numeric
"""


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text).lstrip())
    return str(path)


def _unknown(result) -> list[str]:
    return [e.message for e in result.errors if "Unknown key" in e.message]


# ------------------------------------------------------------- top level


def test_unknown_top_level_key_is_rejected(tmp_path) -> None:
    """The shipped example advertised `whitelist:` for months (issue #7)."""
    result = validate_config(
        _write(tmp_path, BASE + "whitelist:\n  transformations: [identity]\n")
    )
    assert not result.is_valid
    assert any("'whitelist'" in m for m in _unknown(result))


def test_every_documented_top_level_key_is_accepted(tmp_path) -> None:
    """The guard must not reject the surface it is guarding."""
    result = validate_config(
        _write(
            tmp_path,
            BASE + "aggregations: [count]\ntransformations: [identity]\n"
            "as_of_boundary: exclusive\nrelationships: []\n",
        )
    )
    assert _unknown(result) == []


# ---------------------------------------------------------- relationship


def test_flat_parent_key_spelling_is_rejected_with_the_nested_form(tmp_path) -> None:
    """`parent_key:` is what the SQL and Relationship.__init__ call it.

    The config nests it as ``parent: {entity, key}``, so writing the flat form
    used to yield a silently ignored key followed by a bare ``KeyError: 'key'``.
    """
    result = validate_config(
        _write(
            tmp_path,
            BASE + "relationships:\n  - parent_key: user_id\n    child_key: user_id\n",
        )
    )
    assert not result.is_valid
    messages = _unknown(result)
    assert any("'parent_key'" in m for m in messages)
    assert any("'child_key'" in m for m in messages)
    suggestions = " ".join(e.suggestion or "" for e in result.errors)
    assert "parent" in suggestions and "child" in suggestions


def test_known_relationship_keys_are_accepted(tmp_path) -> None:
    result = validate_config(
        _write(
            tmp_path,
            BASE + "relationships:\n"
            "  - name: r\n"
            "    parent: {entity: users, key: user_id}\n"
            "    child: {entity: users, key: user_id}\n"
            "    temporal: {mode: as_of}\n",
        )
    )
    assert _unknown(result) == []


# -------------------------------------------------------------- variable


def test_per_variable_intervals_is_rejected(tmp_path) -> None:
    """Documented once, never implemented, silently dropped (issue #12).

    ``Variable.__init__`` takes no ``intervals`` argument and ``Entity`` never
    passes one, so a config asking for per-variable windows got the global
    ones and no diagnostic.
    """
    result = validate_config(
        _write(
            tmp_path,
            """
            target: users
            max_depth: 1
            intervals: [P1M]
            entities:
              - alias: users
                id: user_id
                table: users
                temporal_ix: created_at
                variables:
                  age:
                    type: numeric
                    intervals: [P7D, P1Y]
            """,
        )
    )
    assert not result.is_valid
    assert any("'intervals'" in m for m in _unknown(result))


@pytest.mark.parametrize("key", ["predicates", "role", "vocabulary"])
def test_known_variable_keys_are_accepted(tmp_path, key: str) -> None:
    values = {
        "predicates": "{active: 'status = 1'}",
        "role": "categorical",
        "vocabulary": "[a, b]",
    }
    result = validate_config(
        _write(
            tmp_path,
            f"""
            target: users
            max_depth: 1
            intervals: [P1M]
            entities:
              - alias: users
                id: user_id
                table: users
                temporal_ix: created_at
                variables:
                  age:
                    type: categorical
                    {key}: {values[key]}
            """,
        )
    )
    assert _unknown(result) == []


# ------------------------------------------------------------- the fleet


def test_every_shipped_example_still_validates() -> None:
    """The regression net: the guard must not break what ships.

    ``examples/04-custom-primitives`` is excluded because it names primitives
    the tutorial registers at runtime, so it has never validated standalone —
    a pre-existing condition, unrelated to unknown keys.
    """
    repo = Path(__file__).resolve().parent.parent
    configs = sorted(repo.glob("examples/*/config.yaml"))
    configs = [c for c in configs if "04-custom-primitives" not in str(c)]
    configs.append(repo / "featurizer" / "featurizer.yaml")
    assert len(configs) >= 6

    for config in configs:
        result = validate_config(str(config))
        assert _unknown(result) == [], f"{config.name}: {_unknown(result)}"
        assert result.is_valid, f"{config.name}: {[e.message for e in result.errors]}"
