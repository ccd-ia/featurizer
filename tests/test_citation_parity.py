"""``CITATION.cff`` must describe the release it ships with.

Since 1.1.1 every release is archived on Zenodo, and Zenodo builds the archived
record's metadata from this file — title, abstract, version, license, authors.
That makes ``CITATION.cff`` release-critical in a way it was not before, and
its failure mode is silent: nothing in CI or in ``release.yml`` notices a stale
``version:``, so the tag succeeds and Zenodo simply mints a permanent DOI whose
metadata names the wrong version. A minted DOI cannot be un-minted.

These checks turn that into a failed fast tier at bump time, the same way
``test_skill_parity.py`` does for the ``featurizer-dfs`` skill. Both exist
because a file that quotes release facts drifts unless something pins it.

DB-free.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CITATION_PATH = REPO_ROOT / "CITATION.cff"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CHANGELOG_PATH = REPO_ROOT / "CHANGELOG.md"

# The subset of the CFF schema Zenodo actually implements, per
# help.zenodo.org/docs/github/describe-software/citation-file/. Anything absent
# here degrades the archived record — a missing `abstract`, for instance, makes
# Zenodo fall back to the GitHub repository blurb.
ZENODO_CONSUMED_FIELDS = (
    "cff-version",
    "title",
    "abstract",
    "version",
    "type",
    "license",
    "message",
    "authors",
    "keywords",
)


@pytest.fixture(scope="module")
def citation() -> dict:
    assert CITATION_PATH.exists(), (
        "CITATION.cff is missing — it is the sole source of the metadata Zenodo "
        "attaches to every archived release"
    )
    return yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def project_version() -> str:
    # ``tomllib`` is 3.11+ and the project floor is 3.10; the version line is
    # the only field needed, and ``release.yml`` pins the same string.
    match = re.search(
        r'^version\s*=\s*"([^"]+)"', PYPROJECT_PATH.read_text(), re.MULTILINE
    )
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_citation_version_matches_pyproject(citation, project_version):
    # ``str()`` because YAML types a two-component version like `2.0` as a
    # float; every release so far is X.Y.Z, which parses as a string.
    stated = str(citation.get("version", ""))
    assert stated == project_version, (
        f"CITATION.cff says version {stated!r} but pyproject.toml is "
        f"{project_version!r} — bump both in the release commit, or Zenodo "
        f"archives the next tag under the wrong version"
    )


def test_citation_date_matches_the_changelog_section(citation, project_version):
    released = citation.get("date-released")
    assert isinstance(released, dt.date), (
        f"CITATION.cff date-released is {released!r}; it must be a plain "
        "YYYY-MM-DD date"
    )
    match = re.search(
        rf"^## \[{re.escape(project_version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})",
        CHANGELOG_PATH.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match, (
        f"CHANGELOG.md has no '## [{project_version}] - YYYY-MM-DD' section — "
        "release.yml's guard would fail this tag too"
    )
    assert released.isoformat() == match.group(1), (
        f"CITATION.cff date-released is {released.isoformat()} but the "
        f"CHANGELOG section for {project_version} is dated {match.group(1)}"
    )


def test_citation_carries_every_field_zenodo_consumes(citation):
    missing = [f for f in ZENODO_CONSUMED_FIELDS if not citation.get(f)]
    assert not missing, (
        f"CITATION.cff is missing fields Zenodo reads: {missing}. The archived "
        "record degrades silently without them."
    )


def test_authors_carry_an_orcid(citation):
    # The ORCID is what connects an archived release to a person rather than to
    # a name string; Zenodo passes it straight through to the record.
    authors = citation["authors"]
    without = [a for a in authors if not a.get("orcid")]
    assert not without, f"CITATION.cff authors without an orcid: {without}"


def test_no_zenodo_json_shadows_the_citation_file():
    # Recorded in CONTRIBUTING: when a repository carries both, Zenodo uses
    # .zenodo.json and ignores CITATION.cff *entirely*, which would leave two
    # metadata sources to drift apart and silently demote this file to the
    # GitHub citation widget. Adding one must be a deliberate act, not a
    # drive-by that quietly disables everything above.
    assert not (REPO_ROOT / ".zenodo.json").exists(), (
        ".zenodo.json exists — Zenodo will ignore CITATION.cff completely. "
        "Delete it, or move the citation metadata into it and retire this test."
    )
