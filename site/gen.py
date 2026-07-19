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

import json
import re
import shutil
import subprocess
import sys
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
        (REPO / "site" / "explorables", PUBLIC / "explorables"),
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


def generate_primitives() -> Path:
    """The primitives reference, generated from the live registry.

    The registry (``list_aggregations`` / ``list_transformations``) is the
    source of truth for *what exists*; ``featurizer.cli``'s DOCS dicts supply
    the human metadata. A primitive missing from the DOCS renders as a stub
    row rather than crashing — the count-parity test in tests/test_site_gen.py
    is the drift alarm.
    """
    sys.path.insert(0, str(REPO))
    from featurizer.cli import AGGREGATION_DOCS, TRANSFORMATION_DOCS
    from featurizer.primitives.utils import list_aggregations, list_transformations

    def esc(text: str) -> str:
        return text.replace("|", "\\|")

    def section(kind: str, names: list[str], docs: dict) -> list[str]:
        by_category: dict[str, list[str]] = {}
        for name in names:
            meta = docs.get(name, {})
            by_category.setdefault(meta.get("category", "general"), []).append(name)
        lines = [f"## {kind} ({len(names)})", ""]
        for category in sorted(by_category):
            lines += [f"### {category}", ""]
            lines += ["| primitive | description | SQL example |", "|---|---|---|"]
            for name in sorted(by_category[category]):
                meta = docs.get(name, {})
                description = esc(meta.get("description", "*(no metadata registered)*"))
                sql = meta.get("sql_example", "")
                sql_cell = f"`{esc(sql)}`" if sql else "—"
                lines.append(f"| `{name}` | {description} | {sql_cell} |")
            lines.append("")
        return lines

    aggs = sorted(list_aggregations())
    transforms = sorted(list_transformations())
    lines = [
        "---",
        "title: Primitives reference",
        "description: >-",
        f"  Every registered primitive — {len(aggs)} aggregations and",
        f"  {len(transforms)} transformers — generated from the live registry.",
        "sidebar:",
        "  order: 1",
        "---",
        "",
        f"featurizer registers **{len(aggs)} aggregations** (applied across",
        "backward relationships, parent ← child) and",
        f"**{len(transforms)} transformers** (applied to features within an",
        "entity). This page is generated from the registry at build time, so it",
        "cannot drift from the code. Select primitives per config with the",
        "`aggregations:` / `transformations:` keys — see the",
        "[configuration reference](/featurizer/reference/configuration/).",
        "",
        ":::tip[Prefer to browse interactively?]",
        "Open the [**primitives explorer**](/featurizer/explorables/primitives.html)"
        " — faceted filter by type and category, live search, and the SQL each"
        " primitive emits.",
        ":::",
        "",
        ":::note",
        "Peer-group features (`peer_groups`), spatial second-table features",
        "(`spatial_relationships`), and the φ-bridge companion are **planner",
        "passes** driven by their own config blocks — deliberately not registry",
        "primitives, so they are not listed here.",
        ":::",
        "",
        "Discover the same information from the CLI:",
        "",
        "```bash",
        "uv run python -m featurizer list-primitives --type agg --show-sql",
        "```",
        "",
    ]
    lines += section("Aggregations", aggs, AGGREGATION_DOCS)
    lines += section("Transformers", transforms, TRANSFORMATION_DOCS)

    out = DOCS / "reference" / "primitives.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(
        f"primitives registry ({len(aggs)}+{len(transforms)}) -> {out.relative_to(REPO)}"
    )
    return out


def registry_records() -> tuple[list[dict[str, str]], int, int]:
    """Flat primitive records from the live registry — the explorer's data.

    Same source of truth as ``generate_primitives`` (the registry for *what
    exists*, ``featurizer.cli``'s DOCS for the human metadata), so the explorer
    cannot drift from the reference table.
    """
    sys.path.insert(0, str(REPO))
    from featurizer.cli import AGGREGATION_DOCS, TRANSFORMATION_DOCS
    from featurizer.primitives.utils import list_aggregations, list_transformations

    aggs = sorted(list_aggregations())
    transforms = sorted(list_transformations())
    records: list[dict[str, str]] = []
    for kind, names, docs in (
        ("aggregation", aggs, AGGREGATION_DOCS),
        ("transformer", transforms, TRANSFORMATION_DOCS),
    ):
        for name in names:
            meta = docs.get(name, {})
            records.append(
                {
                    "name": name,
                    "type": kind,
                    "category": meta.get("category", "general"),
                    "description": meta.get("description", ""),
                    "sql": meta.get("sql_example", ""),
                }
            )
    return records, len(aggs), len(transforms)


# Self-contained (CSP-clean) interactive explorer: faceted filter + text search
# + SQL preview over the registry JSON, embedded inline. Palette mirrors the
# hand-authored explorable (site/explorables/phi-dfs.html) so it reads as native
# to the site. __DATA__/__N_AGG__/__N_TRANS__ are string-substituted, never
# .format()'d — the CSS/JS braces must survive verbatim.
EXPLORER_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Primitives explorer — featurizer</title>
<style>
:root{
  --bg:#fafaf8;--surface:#fff;--fg:#1a2332;--muted:#5b6472;
  --accent:#0f766e;--accent-soft:#ccfbf1;--accent-2:#b45309;--accent-2-soft:#fef3c7;
  --border:#e2e4e0;--code-bg:#f1f3f0;
  --mono:"SF Mono",Menlo,Consolas,monospace;
  --sans:-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root{--bg:#15181e;--surface:#1e232b;--fg:#e7eaf0;--muted:#98a1af;
  --accent:#2dd4bf;--accent-soft:#134e4a;--accent-2:#f5b45e;--accent-2-soft:#4a3008;
  --border:#333a45;--code-bg:#262c36}
}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.5;margin:0;padding:1rem}
main{max-width:920px;margin:0 auto}
h1{font-size:1.2rem;margin:.2rem 0 .1rem}
p{margin:.35rem 0;font-size:.86rem}
.muted{color:var(--muted)}
code{font-family:var(--mono);font-size:.82em;background:var(--code-bg);padding:.05em .3em;border-radius:4px}
a{color:var(--accent)}
.controls{position:sticky;top:0;background:var(--bg);z-index:2;display:flex;flex-wrap:wrap;gap:.7rem;
  align-items:center;padding:.7rem 0;border-bottom:1px solid var(--border);margin-bottom:.6rem}
.controls label{font-size:.72rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em}
input[type=search],select{font:inherit;font-size:.85rem;background:var(--surface);color:var(--fg);
  border:1px solid var(--border);border-radius:7px;padding:.32rem .55rem}
input[type=search]{min-width:min(260px,80vw);flex:1 1 200px}
input[type=search]:focus,select:focus{outline:2px solid var(--accent);outline-offset:0}
.count{font-size:.78rem;color:var(--muted);margin:.2rem 0 .8rem}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:.7rem}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:.7rem .85rem;
  display:flex;flex-direction:column;gap:.35rem}
.card .name{font-family:var(--mono);font-weight:700;font-size:.9rem;word-break:break-word}
.badges{display:flex;flex-wrap:wrap;gap:.35rem;align-items:center}
.badge{font-size:.66rem;font-weight:700;text-transform:uppercase;letter-spacing:.03em;
  padding:.1rem .4rem;border-radius:999px;white-space:nowrap}
.badge.agg{background:var(--accent-soft);color:var(--accent)}
.badge.transformer{background:var(--accent-2-soft);color:var(--accent-2)}
.badge.cat{background:var(--code-bg);color:var(--muted)}
.desc{font-size:.8rem;color:var(--fg)}
.desc.none{color:var(--muted);font-style:italic}
.sql{font-family:var(--mono);font-size:.72rem;background:var(--code-bg);border:1px solid var(--border);
  border-radius:7px;padding:.4rem .55rem;overflow-x:auto;white-space:pre}
mark{background:var(--accent-2-soft);color:inherit;border-radius:3px;padding:0 .05em}
.empty{color:var(--muted);font-size:.85rem;padding:1.5rem 0;text-align:center}
footer{font-size:.72rem;color:var(--muted);margin-top:1.4rem;border-top:1px solid var(--border);padding-top:.6rem}
</style>
</head>
<body>
<main>
<h1>Primitives explorer</h1>
<p class="muted">Every registered primitive — <strong id="nagg"></strong> aggregations
(parent&nbsp;&larr;&nbsp;child) and <strong id="ntrans"></strong> transformers (within an entity).
Generated from the live registry at build time, so it cannot drift from the code.
Filter, search, and preview the SQL each one emits.</p>

<div class="controls">
  <input type="search" id="q" placeholder="Search name or description…" autocomplete="off" aria-label="Search primitives">
  <span><label for="type">Type</label>
    <select id="type" aria-label="Filter by type">
      <option value="all">All</option>
      <option value="aggregation">Aggregations</option>
      <option value="transformer">Transformers</option>
    </select></span>
  <span><label for="cat">Category</label>
    <select id="cat" aria-label="Filter by category"></select></span>
</div>
<p class="count" id="count"></p>
<div class="grid" id="grid"></div>
<p class="empty" id="empty" hidden>No primitives match — clear the search or widen the filters.</p>

<footer>
Back to the <a href="../reference/primitives/">primitives reference</a> ·
Discover the same from the CLI: <code>python -m featurizer list-primitives --type agg --show-sql</code>
</footer>
</main>
<script>
const DATA = __DATA__;
const N_AGG = __N_AGG__, N_TRANS = __N_TRANS__;
const $ = (id) => document.getElementById(id);
$("nagg").textContent = N_AGG; $("ntrans").textContent = N_TRANS;

// Category dropdown reflects the current type filter (only reachable categories).
function refreshCategories() {
  const type = $("type").value;
  const cats = [...new Set(DATA.filter(d => type === "all" || d.type === type)
    .map(d => d.category))].sort();
  const prev = $("cat").value;
  $("cat").innerHTML = '<option value="all">All</option>' +
    cats.map(c => `<option value="${c}">${c}</option>`).join("");
  $("cat").value = cats.includes(prev) ? prev : "all";
}

function esc(s) {
  return s.replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
}
function highlight(text, q) {
  const safe = esc(text);
  if (!q) return safe;
  const re = new RegExp("(" + q.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&") + ")", "ig");
  return safe.replace(re, "<mark>$1</mark>");
}

function render() {
  const q = $("q").value.trim().toLowerCase();
  const type = $("type").value, cat = $("cat").value;
  const rows = DATA.filter(d =>
    (type === "all" || d.type === type) &&
    (cat === "all" || d.category === cat) &&
    (!q || d.name.toLowerCase().includes(q) || (d.description || "").toLowerCase().includes(q)));
  const grid = $("grid");
  grid.innerHTML = rows.map(d => {
    const badge = d.type === "aggregation"
      ? '<span class="badge agg">agg</span>'
      : '<span class="badge transformer">transform</span>';
    const desc = d.description
      ? `<div class="desc">${highlight(d.description, q)}</div>`
      : '<div class="desc none">no metadata registered</div>';
    const sql = d.sql ? `<div class="sql">${esc(d.sql)}</div>` : "";
    return `<div class="card"><div class="name">${highlight(d.name, q)}</div>` +
      `<div class="badges">${badge}<span class="badge cat">${esc(d.category)}</span></div>` +
      `${desc}${sql}</div>`;
  }).join("");
  $("empty").hidden = rows.length > 0;
  const nAgg = rows.filter(d => d.type === "aggregation").length;
  $("count").textContent =
    `Showing ${rows.length} of ${DATA.length} — ${nAgg} aggregations, ${rows.length - nAgg} transformers`;
}

$("q").addEventListener("input", render);
$("type").addEventListener("change", () => { refreshCategories(); render(); });
$("cat").addEventListener("change", render);
refreshCategories();
render();
</script>
</body>
</html>
"""


def generate_explorer() -> Path:
    """The interactive primitives explorer -> public/explorables/primitives.html.

    Runs AFTER copy_passthrough (which populates public/explorables/ from the
    hand-authored sources) so the generated page joins them without being wiped.
    """
    records, n_agg, n_trans = registry_records()
    # Escape ``</`` so the inline JSON can never terminate the <script> early.
    data = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    html = (
        EXPLORER_TEMPLATE.replace("__DATA__", data)
        .replace("__N_AGG__", str(n_agg))
        .replace("__N_TRANS__", str(n_trans))
    )
    out = PUBLIC / "explorables" / "primitives.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"primitives explorer ({n_agg}+{n_trans}) -> {out.relative_to(REPO)}")
    return out


# The SQL-spine public API, documented by pdoc into public/api/. The optional
# viz/bridge extras are deliberately omitted (heavy deps absent in the docs
# build) — pdoc still emits harmless return-type warnings for the FeaturizerViz
# methods re-exported at the top level. Extend this list when a new core module
# joins the public surface.
API_MODULES: list[str] = [
    "featurizer",
    "featurizer.planner",
    "featurizer.sql",
    "featurizer.executor",
    "featurizer.validation",
    "featurizer.manifest",
    "featurizer.categoricals",
    "featurizer.imputation",
    "featurizer.sharding",
    "featurizer.boundary",
    "featurizer.arrow",
    "featurizer.primitives.abstractions",
    "featurizer.primitives.aggregations",
    "featurizer.primitives.transformations",
    "featurizer.primitives.utils",
    "featurizer.primitives.preagg",
]


def generate_api() -> Path:
    """Auto-generated API reference (pdoc) -> public/api/, self-contained HTML.

    pdoc renders the Google-style docstrings already in the source to a set of
    static, CSP-clean pages (all CSS/JS inlined; the only external reference is
    pdoc's own footer link). ``--edit-url`` wires each page back to its module
    on GitHub. Served under /featurizer/api/ via pdoc's relative links.
    """
    api_out = PUBLIC / "api"
    if api_out.exists():
        shutil.rmtree(api_out)
    api_out.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdoc",
            *API_MODULES,
            "--docformat",
            "google",
            "--edit-url",
            f"featurizer={GITHUB}/blob/master/featurizer/",
            "-o",
            str(api_out),
        ],
        cwd=REPO,
        check=True,
    )
    pages = sorted(p.relative_to(api_out) for p in api_out.rglob("*.html"))
    print(f"api reference (pdoc): {len(pages)} pages -> {api_out.relative_to(REPO)}")
    return api_out


# ADR index grouping. New ADRs land in "other" until themed here — the index
# generator prints a reminder when that bucket is non-empty.
ADR_THEMES: dict[str, list[str]] = {
    "Sharding & performance": ["0005", "0006", "0009", "0010", "0012", "0013"],
    "Correctness & leakage": ["0001", "0008", "0014"],
    "Feature families": ["0002", "0004", "0007", "0011"],
    "Operations & boundaries": ["0003", "0015"],
}


def ingest_engineering() -> None:
    """docs/adr/*.md + CHANGELOG.md -> engineering/ pages (canonical homes stay).

    Only the *generated* content is cleaned (adr/ + changelog.md) — the
    engineering/ directory also holds authored pages (internals.md).
    """
    eng = DOCS / "engineering"
    adr_out = eng / "adr"
    if adr_out.exists():
        shutil.rmtree(adr_out)
    (eng / "changelog.md").unlink(missing_ok=True)
    adr_out.mkdir(parents=True)

    def fm(title: str, description: str, order: int | None = None) -> str:
        lines = ["---", f'title: "{title}"', f'description: "{description}"']
        if order is not None:
            lines += ["sidebar:", f"  order: {order}"]
        return "\n".join(lines) + "\n---\n\n"

    entries: dict[str, tuple[str, str]] = {}  # number -> (slug, title)
    for adr in sorted((REPO / "docs" / "adr").glob("[0-9]*.md")):
        text = adr.read_text()
        h1 = re.search(r"^# (.+)$", text, flags=re.M)
        title = h1.group(1).strip() if h1 else adr.stem
        body = text.replace(h1.group(0), "", 1).lstrip() if h1 else text
        number = adr.stem.split("-")[0]
        entries[number] = (adr.stem, title)
        # Source-relative links: sibling ADRs become site routes; anything
        # else under docs/ points at GitHub (org docs stay canonical in-repo).
        body = re.sub(
            r"\]\((\d{4}-[\w-]+)\.md\)",
            r"](/featurizer/engineering/adr/\1/)",
            body,
        )
        body = re.sub(
            r"\]\(\.\./([\w./-]+)\)",
            rf"]({GITHUB}/blob/master/docs/\1)",
            body,
        )
        source_note = (
            f"\n\n---\n\n*Canonical file: "
            f"[`docs/adr/{adr.name}`]({GITHUB}/blob/master/docs/adr/{adr.name})*\n"
        )
        (adr_out / f"{adr.stem}.md").write_text(
            fm(title, f"Architecture decision record {number}.") + body + source_note
        )

    themed = {n for numbers in ADR_THEMES.values() for n in numbers}
    other = [n for n in sorted(entries) if n not in themed]
    if other:
        print(f"NOTE: untitled ADR theme bucket gets: {other} — extend ADR_THEMES")
    index_lines = [
        fm("Architecture decisions", "The ADR index, grouped by theme.", 0)
        + "Short records of hard-to-reverse decisions and the trade-offs behind\n"
        "them. Canonical home:\n"
        f"[`docs/adr/`]({GITHUB}/tree/master/docs/adr).\n"
    ]
    groups = list(ADR_THEMES.items()) + ([("Other", other)] if other else [])
    for theme, numbers in groups:
        index_lines.append(f"\n## {theme}\n")
        for number in numbers:
            if number not in entries:
                continue
            slug, title = entries[number]
            index_lines.append(f"- [{title}](/featurizer/engineering/adr/{slug}/)")
    (adr_out / "index.md").write_text("\n".join(index_lines) + "\n")

    changelog = (REPO / "CHANGELOG.md").read_text()
    changelog = re.sub(r"^# Changelog\n", "", changelog, count=1)
    changelog = re.sub(
        r"\]\(docs/adr/(\d{4}-[\w-]+)\.md\)",
        r"](/featurizer/engineering/adr/\1/)",
        changelog,
    )
    changelog = re.sub(
        r"\]\((docs/[\w./-]+|specs/[\w./-]+|tests/[\w./-]+)\)",
        rf"]({GITHUB}/blob/master/\1)",
        changelog,
    )
    (eng / "changelog.md").write_text(
        fm("Changelog", "All notable changes, per release.", 1) + changelog
    )
    print(f"engineering pages: {len(entries)} ADRs + index + changelog")


def main() -> None:
    copy_passthrough()
    convert_notebooks()
    generate_primitives()
    generate_explorer()
    generate_api()
    ingest_engineering()


if __name__ == "__main__":
    main()
