"""The ``featurizer-dfs`` Claude skill must describe the engine that ships with it.

``.claude/skills/featurizer-dfs/SKILL.md`` is the canonical copy of the skill;
``~/.claude/skills/`` and the claude-tips catalog hold vendored snapshots of its
body. The skill quotes facts that change with releases — the version, the
primitive counts, the curated default set, the public ``Featurizer`` methods —
and every one of them drifted between 0.9.1 and 1.1.0 without anything
noticing. These checks make that drift a failed test (and, through the
``release.yml`` version guard, a failed release) instead of a discovery weeks
later. Same idea as the count-parity check on the generated primitives
reference: the prose may not disagree with the registry.

DB-free.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from featurizer import Featurizer
from featurizer.featurizer import DEFAULT_AGGREGATIONS, DEFAULT_TRANSFORMATIONS
from featurizer.primitives.utils import get_aggregations, get_transformers

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_PATH = REPO_ROOT / ".claude" / "skills" / "featurizer-dfs" / "SKILL.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def skill_text() -> str:
    assert SKILL_PATH.exists(), (
        f"{SKILL_PATH.relative_to(REPO_ROOT)} is missing — it is the canonical "
        "copy of the featurizer-dfs skill and must be committed with the engine"
    )
    return SKILL_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def project_version() -> str:
    # ``tomllib`` is 3.11+, and the project floor is 3.10; the version line is
    # the only field needed, and ``release.yml`` pins the same string.
    match = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT_PATH.read_text(), re.MULTILINE
    )
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_skill_heading_names_the_current_version(skill_text, project_version):
    heading = re.search(r"^# Featurizer .*?— v(\S+)\s*$", skill_text, re.MULTILINE)
    assert heading, "skill has no '# Featurizer … — vX.Y.Z' heading"
    assert heading.group(1) == project_version, (
        f"skill heading says v{heading.group(1)} but pyproject.toml is "
        f"{project_version} — update the skill body as part of the release"
    )


def test_skill_pin_line_names_the_current_tag(skill_text, project_version):
    pins = re.findall(r"github\.com/ccd-ia/featurizer(?:\.git)?@v(\S+?)`", skill_text)
    assert pins, "skill has no `git+https://github.com/ccd-ia/featurizer@vX.Y.Z` pin"
    assert set(pins) == {project_version}, (
        f"skill pins {sorted(set(pins))} but the current version is {project_version}"
    )


def test_skill_primitive_counts_match_the_registry(skill_text):
    n_agg = len(get_aggregations())
    n_tx = len(get_transformers())
    stated = re.search(
        r"(\d+) aggregations \+ (\d+) transformers \((\d+)\s*primitives\)",
        skill_text,
    )
    assert stated, "skill has no 'N aggregations + M transformers (T primitives)'"
    assert tuple(int(g) for g in stated.groups()) == (n_agg, n_tx, n_agg + n_tx), (
        f"skill states {stated.group(0)!r}; registry has {n_agg} aggregations "
        f"and {n_tx} transformers"
    )
    # Every other place the total is quoted must agree too (the anti-patterns
    # section repeats it).
    totals = {int(t) for t in re.findall(r"all (\d+) primitives", skill_text)}
    assert totals <= {n_agg + n_tx}, f"stale primitive totals in skill: {totals}"


def test_skill_lists_the_curated_default_aggregations(skill_text):
    # The skill tells the reader what "omit `aggregations:`" buys them. That
    # sentence must name exactly the engine's curated set, or the reader plans
    # around the wrong baseline.
    match = re.search(
        r"curated active set \(([^)]+?)\s*\+ \d+ default transformers\)",
        skill_text,
    )
    assert match, "skill has no 'curated active set (…) + N default transformers'"
    # The sentence lives in a YAML comment that wraps: drop whitespace, the
    # continuation-line `#`, and any backticks from each token.
    listed = {re.sub(r"[\s#`]", "", name) for name in match.group(1).split(",")}
    assert listed == set(DEFAULT_AGGREGATIONS), (
        f"skill lists {sorted(listed)}; engine defaults are "
        f"{sorted(DEFAULT_AGGREGATIONS)}"
    )
    stated_tx = re.search(r"\+ (\d+) default transformers\)", skill_text)
    assert stated_tx and int(stated_tx.group(1)) == len(DEFAULT_TRANSFORMATIONS)


def test_every_featurizer_attribute_the_skill_names_exists(skill_text):
    # ``f.to_arrow(``, ``f.columns_matching(``, ``f.query`` … — a renamed or
    # removed method would otherwise survive in the prose indefinitely.
    named = set(re.findall(r"\bf\.([a-z_]+)", skill_text))
    assert named, "skill names no `f.<attribute>` at all — regex out of date?"
    missing = sorted(n for n in named if not hasattr(Featurizer, n))
    assert not missing, (
        f"skill references Featurizer attributes that do not exist: {missing}"
    )


def test_skill_documents_the_post_0_9_surface(skill_text):
    # The three things that drifted last time, pinned by name so the next
    # re-sync cannot quietly drop them: the 1.1.0 label-globbing helpers, the
    # 1.0.1 empty-list semantics, and the 1.0.0 freeze.
    for needle in (
        "columns_matching",
        "manifest_matching",
        "glob_to_like",
        "transformations: []",
        "aggregations: []",
        "ADR-0015",
    ):
        assert needle in skill_text, f"skill does not mention {needle!r}"
