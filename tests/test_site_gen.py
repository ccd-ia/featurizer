# coding: utf-8

"""DB-free tests for the docs-site generator (site/gen.py).

Skipped wholesale when the ``docs`` dependency group (nbconvert) is not
installed — the pages workflow installs it and runs the generator for real.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("nbconvert")

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("site_gen", REPO / "site" / "gen.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["site_gen"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def converted(gen) -> list[str]:
    return gen.convert_notebooks()


def test_every_example_notebook_converts(converted: list[str]) -> None:
    example_notebooks = sorted(
        d.name
        for d in (REPO / "examples").iterdir()
        if (d / "tutorial.ipynb").is_file()
    )
    assert converted == [f"{name}.md" for name in example_notebooks]
    assert len(converted) == 6


def test_converted_pages_have_frontmatter_and_outputs(converted: list[str]) -> None:
    notebooks_dir = REPO / "src" / "content" / "docs" / "notebooks"
    for name in converted:
        text = (notebooks_dir / name).read_text()
        assert text.startswith("---\ntitle:"), name
        assert "View on GitHub" in text, name
    # The executed-output contract: notebook 05 carries real DataFrame tables.
    assert (notebooks_dir / "05-categoricals-output.md").read_text().count(
        "<table"
    ) >= 1


def test_referenced_notebook_assets_exist(converted: list[str]) -> None:
    notebooks_dir = REPO / "src" / "content" / "docs" / "notebooks"
    assets_root = REPO / "public" / "notebook-assets"
    for name in converted:
        text = (notebooks_dir / name).read_text()
        for ref in re.findall(r"\]\(/featurizer/notebook-assets/([^)]+)\)", text):
            assert (assets_root / ref).is_file(), f"{name}: missing asset {ref}"


def test_primitives_page_row_count_matches_registry(gen) -> None:
    """The drift alarm: one table row per registered primitive, counts in header."""
    import sys as _sys

    _sys.path.insert(0, str(REPO))
    from featurizer.primitives.utils import list_aggregations, list_transformations

    page = gen.generate_primitives().read_text()
    n_aggs = len(list(list_aggregations()))
    n_transforms = len(list(list_transformations()))
    rows = re.findall(r"^\| `", page, flags=re.M)
    assert len(rows) == n_aggs + n_transforms
    assert f"## Aggregations ({n_aggs})" in page
    assert f"## Transformers ({n_transforms})" in page


def test_configuration_yaml_snippets_parse() -> None:
    """Every fenced YAML block in the configuration reference must load."""
    import yaml

    text = (REPO / "src/content/docs/reference/configuration.md").read_text()
    blocks = re.findall(r"```yaml\n(.*?)```", text, flags=re.S)
    assert blocks, "configuration.md should contain yaml examples"
    for block in blocks:
        yaml.safe_load(block)


def test_explorable_is_self_contained() -> None:
    """The explorable must load nothing from external hosts (CSP-clean):
    no external scripts/styles/fonts/images; only plain <a href> links out."""
    html = (REPO / "site/explorables/phi-dfs.html").read_text()
    assert re.search(r"<script\s+[^>]*src=", html) is None
    assert re.search(r"<link\s", html) is None
    assert "@import" not in html and "fonts." not in html
    externals = re.findall(r'(\w+)="https?://[^"]+"', html)
    assert set(externals) <= {"href"}, f"non-anchor external refs: {externals}"


def test_explorable_copied_to_public(gen) -> None:
    gen.copy_passthrough()
    assert (REPO / "public/explorables/phi-dfs.html").is_file()


def test_explorer_generates_from_registry(gen) -> None:
    """The explorer embeds one record per registered primitive, counts wired."""
    gen.copy_passthrough()  # populates public/explorables/ before the explorer joins
    out = gen.generate_explorer()
    html = out.read_text()
    records, n_agg, n_trans = gen.registry_records()
    assert len(records) == n_agg + n_trans
    assert html.count('"name":') == len(records)
    assert f"N_AGG = {n_agg}, N_TRANS = {n_trans}" in html
    # No template placeholder survives substitution.
    for token in ("__DATA__", "__N_AGG__", "__N_TRANS__"):
        assert token not in html, token


def test_explorer_is_self_contained(gen) -> None:
    """The generated explorer is CSP-clean — only <a href> links leave the page."""
    gen.copy_passthrough()
    html = gen.generate_explorer().read_text()
    assert re.search(r"<script\s+[^>]*src=", html) is None
    assert re.search(r"<link\s", html) is None
    assert "@import" not in html and "fonts." not in html
    externals = re.findall(r'(\w+)="https?://[^"]+"', html)
    assert set(externals) <= {"href"}, f"non-anchor external refs: {externals}"


def test_api_reference_generates_curated_modules(gen) -> None:
    """pdoc emits one page per curated module into public/api/, index included."""
    pytest.importorskip("pdoc")
    api_out = gen.generate_api()
    pages = {p.relative_to(api_out).as_posix() for p in api_out.rglob("*.html")}
    assert "index.html" in pages
    # The SQL spine must be documented — a stand-in for the whole curated list.
    for expected in (
        "featurizer.html",
        "featurizer/planner.html",
        "featurizer/sql.html",
    ):
        assert expected in pages, f"missing api page: {expected}"
    assert len(pages) == len(gen.API_MODULES) + 1  # +1 for the index redirect


def test_api_reference_is_self_contained(gen) -> None:
    """No external resource loads from the pdoc tree — only pdoc's footer link."""
    pytest.importorskip("pdoc")
    api_out = gen.generate_api()
    for page in api_out.rglob("*.html"):
        html = page.read_text()
        assert re.search(r'<script\s+[^>]*src="https?://', html) is None, page.name
        assert re.search(r'<link\s+[^>]*href="https?://', html) is None, page.name
