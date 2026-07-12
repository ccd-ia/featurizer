# coding: utf-8

"""Internal link/asset checker for the built docs site.

    uv run python site/check_links.py [dist]

Walks every HTML file under the build directory, collects href/src values,
and verifies each *internal* target exists in the tree. External (http/https,
mailto) links are skipped. Exits 1 listing every broken target — CI runs this
between ``astro build`` and deploy.

The site is served under the ``/featurizer/`` base path (GitHub project
pages), so absolute paths are resolved by stripping that prefix.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

BASE = "/featurizer"


class RefCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.refs.append(value)


def resolve(ref: str, page: Path, root: Path) -> Path | None:
    """The filesystem target an internal ref points at, or None if external."""
    parsed = urlparse(ref)
    if parsed.scheme or ref.startswith("//"):
        return None  # external
    path = unquote(parsed.path)
    if not path:
        return None  # pure fragment
    if path.startswith(BASE + "/") or path == BASE:
        rel = path[len(BASE) :].lstrip("/")
        target = root / rel
    elif path.startswith("/"):
        target = root / path.lstrip("/")
    else:
        target = (page.parent / path).resolve()
    if path.endswith("/") or target.is_dir():
        target = target / "index.html"
    return target


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "dist").resolve()
    if not root.is_dir():
        print(f"build directory not found: {root}")
        return 1
    broken: list[tuple[Path, str]] = []
    n_pages = n_refs = 0
    for page in sorted(root.rglob("*.html")):
        n_pages += 1
        collector = RefCollector()
        collector.feed(page.read_text(errors="replace"))
        for ref in collector.refs:
            target = resolve(ref, page, root)
            if target is None:
                continue
            n_refs += 1
            if not target.exists():
                broken.append((page.relative_to(root), ref))
    if broken:
        print(f"BROKEN internal links ({len(broken)}):")
        for page_rel, ref in broken:
            print(f"  {page_rel}: {ref}")
        return 1
    print(f"link check ok: {n_pages} pages, {n_refs} internal refs, 0 broken")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
