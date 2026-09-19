"""A ``temporal:`` block on an aggregation-direction relationship warns (issue #9).

``temporal_mode`` is read on exactly one traversal path — the forward
(parent -> child) transfer in ``_build_direct``. Declared on a relationship the
planner walks the other way, the block is discarded: no ``_asof_for_`` lateral
renders and ``grace`` is never applied. Validation used to say nothing, and the
example named after temporal joins shipped that inert orientation.

Nothing leaks either way (the aggregation path bounds the child stream on the
as-of date regardless), so this is a warning and ``is_valid`` stays ``True``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from featurizer import Featurizer, validate_config
from featurizer.validation import ConfigValidator

REPO = Path(__file__).resolve().parent.parent

# The issue's minimal repro, verbatim in shape: ``games`` is the target and the
# parent, so ``teams`` is aggregated onto it and the block is never read.
ISSUE_REPRO = {
    "target": "games",
    "max_depth": 2,
    "intervals": ["P30D"],
    "aggregations": ["sum"],
    "transformations": ["identity"],
    "entities": [
        {
            "alias": "games",
            "id": "game_id",
            "table": "games",
            "temporal_ix": "game_date",
            "variables": {"goals": {"type": "numeric"}},
        },
        {
            "alias": "teams",
            "id": "row_id",
            "table": "teams",
            "temporal_ix": "as_of",
            "variables": {"v0": {"type": "numeric"}},
        },
    ],
    "relationships": [
        {
            "name": "home",
            "parent": {"entity": "games", "key": "home_team"},
            "child": {"entity": "teams", "key": "team"},
            "temporal": {"mode": "as_of", "grace": "P7D"},
        }
    ],
}


def _inert(result) -> list:
    return [w for w in result.warnings if "has no effect" in w.message]


def _render(config: dict, tmp_path: Path) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return Featurizer(str(path)).query


def test_issue_repro_warns_once_and_stays_valid() -> None:
    result = ConfigValidator().validate(copy.deepcopy(ISSUE_REPRO))

    assert result.is_valid
    warnings = _inert(result)
    assert len(warnings) == 1
    assert warnings[0].location == "relationships[0].temporal"


def test_the_warning_names_the_fix_and_says_nothing_leaked() -> None:
    (warning,) = _inert(ConfigValidator().validate(copy.deepcopy(ISSUE_REPRO)))

    assert "'home'" in warning.message
    assert "as 'parent'" in warning.message and "as 'child'" in warning.message
    assert "bounded by as_of_date" in warning.message
    assert "'grace' is not applied" in warning.message


def test_the_warned_config_really_renders_no_asof_lateral(tmp_path) -> None:
    """The warning describes the planner, so pin the planner too."""
    query = _render(copy.deepcopy(ISSUE_REPRO), tmp_path)

    assert "_asof_for_" not in query
    assert "P7D" not in query


def test_the_working_orientation_is_silent_and_renders_the_lateral(tmp_path) -> None:
    config = copy.deepcopy(ISSUE_REPRO)
    config["relationships"][0]["parent"] = {"entity": "teams", "key": "team"}
    config["relationships"][0]["child"] = {"entity": "games", "key": "home_team"}

    assert _inert(ConfigValidator().validate(copy.deepcopy(config))) == []
    query = _render(config, tmp_path)
    assert "home_asof_for_games" in query
    assert "interval 'P7D'" in query


def test_a_relationship_without_a_temporal_block_is_silent() -> None:
    config = copy.deepcopy(ISSUE_REPRO)
    del config["relationships"][0]["temporal"]

    assert _inert(ConfigValidator().validate(config)) == []


def test_direction_is_judged_from_the_target_not_from_the_declaration() -> None:
    """Two hops out, the entity nearer the target is the one traversed from.

    ``seasons <- games <- teams`` with target ``seasons``: the block sits on
    ``games -> teams``, where ``games`` (1 hop) is nearer than ``teams`` (2), so
    ``teams`` is aggregated onto ``games`` and the block is inert.
    """
    config = copy.deepcopy(ISSUE_REPRO)
    config["target"] = "seasons"
    config["max_depth"] = 3
    config["entities"].append(
        {
            "alias": "seasons",
            "id": "season_id",
            "table": "seasons",
            "variables": {"year": {"type": "numeric"}},
        }
    )
    config["entities"][0]["variables"]["season_id"] = {"type": "numeric"}
    config["relationships"].insert(
        0,
        {
            "parent": {"entity": "seasons", "key": "season_id"},
            "child": {"entity": "games", "key": "season_id"},
        },
    )

    warnings = _inert(ConfigValidator().validate(config))
    assert [w.location for w in warnings] == ["relationships[1].temporal"]


def test_example_02_renders_an_asof_lateral() -> None:
    """The test that would have caught this on every version since it shipped.

    Example 02 is the tutorial for as-of joins. Its config carried a
    ``temporal:`` block for years and rendered no lateral at all.
    """
    config = REPO / "examples" / "02-temporal-joins" / "config.yaml"
    declared = yaml.safe_load(config.read_text())
    blocks = [r for r in declared["relationships"] if "temporal" in r]
    assert blocks, "example 02 is the as-of tutorial; it must declare a temporal block"

    query = Featurizer(str(config)).query
    assert query.count("_asof_for_") >= 1
    for rel in blocks:
        grace = rel["temporal"].get("grace")
        if grace:
            assert f"interval '{grace}'" in query


def test_no_shipped_config_carries_an_inert_block() -> None:
    configs = sorted(REPO.glob("examples/*/config.yaml"))
    configs.append(REPO / "featurizer" / "featurizer.yaml")
    assert len(configs) >= 7

    for config in configs:
        assert _inert(validate_config(str(config))) == [], config
