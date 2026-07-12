# coding: utf-8

"""Python pre-build seam for the Starlight docs site.

Run BEFORE ``npm run build``:

    uv run python site/gen.py

Responsibilities (grown phase by phase — see specs/github-pages-docs-hub.html):

1. Pass-through copies: the self-contained validation artifacts (``specs/``)
   and repo images (``docs/images/``) into ``public/`` — they are lab reports,
   not docs pages, and keep their own identity.
2. Notebook conversion: each ``examples/*/tutorial.ipynb`` (its **committed**
   state — the outputs validated against a live database) converts to a
   markdown page inside the Starlight content collection, with image outputs
   extracted to ``public/notebook-assets/``. Rendered, never executed: the
   Pages workflow has no database, and committed outputs are the truth.

Everything this script writes is gitignored: CI regenerates it on every
deploy, so generated content cannot drift from its sources.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "public"
DOCS = REPO / "src" / "content" / "docs"
BASE = "/featurizer"
GITHUB = "https://github.com/ccd-ia/featurizer"


def copy_passthrough() -> None:
    """specs/ and docs/images/ → public/, verbatim (single sources stay put)."""
    for source, dest in [
        (REPO / "specs", PUBLIC / "specs"),
        (REPO / "docs" / "images", PUBLIC / "images"),
    ]:
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)
        n_files = sum(1 for p in dest.rglob("*") if p.is_file())
        print(
            f"copied {source.relative_to(REPO)} -> {dest.relative_to(REPO)} ({n_files} files)"
        )


def committed_bytes(repo_path: str) -> bytes:
    """A file's content at HEAD (never the working tree — see module docs)."""
    return subprocess.run(
        ["git", "show", f"HEAD:{repo_path}"],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout


def convert_notebooks() -> list[str]:
    """examples/*/tutorial.ipynb (committed) -> src/content/docs/notebooks/*.md."""
    import nbformat
    from nbconvert import MarkdownExporter

    out_dir = DOCS / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.md"):
        if stale.name != "index.md":
            stale.unlink()
    assets_root = PUBLIC / "notebook-assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)

    exporter = MarkdownExporter()
    written: list[str] = []
    example_dirs = sorted(
        d for d in (REPO / "examples").iterdir() if (d / "tutorial.ipynb").is_file()
    )
    for order, example_dir in enumerate(example_dirs, start=1):
        slug = example_dir.name  # e.g. 01-basic-aggregations
        rel = f"examples/{slug}/tutorial.ipynb"
        nb = nbformat.reads(committed_bytes(rel).decode("utf-8"), as_version=4)
        body, resources = exporter.from_notebook_node(nb)

        # Extracted outputs (matplotlib PNGs etc.) -> public/, refs rewritten.
        outputs = resources.get("outputs") or {}
        if outputs:
            asset_dir = assets_root / slug
            asset_dir.mkdir(parents=True, exist_ok=True)
            for filename, data in outputs.items():
                (asset_dir / filename).write_bytes(data)
                body = body.replace(
                    f"]({filename})", f"]({BASE}/notebook-assets/{slug}/{filename})"
                )

        # Page title = the notebook's own H1; the body drops it (the frontmatter
        # title renders as the page heading).
        h1 = re.search(r"^# (.+)$", body, flags=re.M)
        title = h1.group(1).strip() if h1 else slug
        title = title.removeprefix("Featurizer Tutorial: ")
        if h1:
            body = body.replace(h1.group(0), "", 1).lstrip()

        header = (
            f'<p><a href="{GITHUB}/blob/master/{rel}">View on GitHub</a> · '
            f'<a href="https://raw.githubusercontent.com/ccd-ia/featurizer/master/{rel}">'
            f"Download .ipynb</a></p>\n\n"
        )
        frontmatter = (
            "---\n"
            f'title: "{order:02d} · {title}"\n'
            f'description: "Tutorial notebook {slug}, rendered from its committed, executed outputs."\n'
            f"sidebar:\n  order: {order}\n"
            "---\n\n"
        )
        page = out_dir / f"{slug}.md"
        page.write_text(frontmatter + header + body)
        written.append(page.name)
        print(f"notebook {rel} -> {page.relative_to(REPO)}")
    return written


def main() -> None:
    copy_passthrough()
    convert_notebooks()


if __name__ == "__main__":
    main()
