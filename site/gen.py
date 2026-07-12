# coding: utf-8

"""Python pre-build seam for the Starlight docs site.

Run BEFORE ``npm run build``:

    uv run python site/gen.py

Responsibilities (grown phase by phase — see specs/github-pages-docs-hub.html):

1. Pass-through copies: the self-contained validation artifacts (``specs/``)
   and repo images (``docs/images/``) into ``public/`` — they are lab reports,
   not docs pages, and keep their own identity.

Everything this script writes is gitignored: CI regenerates it on every
deploy, so generated content cannot drift from its sources.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "public"


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


def main() -> None:
    copy_passthrough()


if __name__ == "__main__":
    main()
