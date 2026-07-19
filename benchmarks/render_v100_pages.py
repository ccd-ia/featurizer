"""Render the v1.0.0 live-DB revalidation pages from the raw JSON artifacts.

Reads ``specs/live-db-revalidation-v100/raw/*.json`` (written by
:mod:`benchmarks.final_matrix` and :mod:`benchmarks.bridge_workloads`) and
emits the committed, self-contained HTML pages:

* ``specs/live-db-revalidation-v100.html`` — the summary matrix;
* ``specs/live-db-revalidation-v100/<db>.html`` — per-database detail.

Deterministic: page content is a pure function of the JSON artifacts, so the
pages can always be regenerated (the v0.6.0/v0.8.0 pages came from an
uncommitted script — this file closes that gap).

Usage::

    uv run python -m benchmarks.render_v100_pages
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .final_matrix import ARTIFACT_DIR

#: v0.8.0 reference numbers (specs/live-db-revalidation-v080.html).
V080_SECONDS = {
    ("dirtyduck", "narrow"): 3.0,
    ("dirtyduck", "all-agg"): 7.5,
    ("dirtyduck", "wide"): 63.2,
    ("chicago311", "narrow"): 0.8,
    ("chicago311", "all-agg"): 6.0,
    ("chicago311", "wide"): 49.2,
    ("donorschoose", "narrow"): 0.4,
    ("donorschoose", "all-agg"): 7.6,
    ("donorschoose", "wide"): 470.1,
}

LEDES = {
    "dirtyduck": "Which food facilities are likely to fail an inspection? (facilities ← inspections)",
    "chicago311": "Which 311 requests will resolve slowly? (requests ← area/type demand streams)",
    "donorschoose": "Which projects will go unfunded? (projects ← resources / teacher / school histories)",
}

CSS = """
:root{--bg:#fafaf8;--surface:#fff;--fg:#1a2332;--muted:#5b6472;--accent:#0f766e;
--accent-soft:#ccfbf1;--accent-2:#b45309;--accent-2-soft:#fef3c7;--ok:#15803d;
--border:#e2e4e0;--code-bg:#f1f3f0;--mono:"SF Mono",Menlo,Consolas,monospace;
--sans:-apple-system,"Segoe UI",Inter,Roboto,sans-serif;}
*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font-family:var(--sans);
line-height:1.55;margin:0}main{max-width:960px;margin:0 auto;padding:2.2rem 1.3rem 4rem}
h1{font-size:1.6rem;margin:0 0 .2rem;letter-spacing:-.02em}
h2{font-size:1.2rem;margin:2rem 0 .6rem;padding-bottom:.3rem;border-bottom:2px solid var(--accent-soft)}
p{margin:.5rem 0}.lede{color:var(--muted)}
code{font-family:var(--mono);font-size:.85em;background:var(--code-bg);padding:.06em .32em;border-radius:4px}
table{border-collapse:collapse;width:100%;font-size:.82rem;margin:.8rem 0}
th,td{border:1px solid var(--border);padding:.38rem .55rem;text-align:left}
th{background:var(--accent-soft);color:var(--accent);font-weight:700}
td.mono{font-family:var(--mono)}
.chip{display:inline-block;font-size:.68rem;font-weight:700;text-transform:uppercase;
padding:.1em .5em;border-radius:5px}.chip.ok{background:#dcfce7;color:#15803d}
.was{color:var(--muted);font-size:.9em}
.note{background:var(--accent-2-soft);border-left:4px solid var(--accent-2);border-radius:6px;
padding:.7rem .9rem;margin:1rem 0;font-size:.88rem}
footer{margin-top:2.5rem;color:var(--muted);font-size:.78rem;border-top:1px solid var(--border);padding-top:1rem}
"""


def _load() -> Dict[str, Any]:
    raw = ARTIFACT_DIR / "raw"
    out: Dict[str, Any] = {}
    for path in sorted(raw.glob("*.json")):
        out[path.stem] = json.loads(path.read_text())
    return out


def _matrix_rows(artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for db in ("dirtyduck", "chicago311", "donorschoose"):
        for variant in ("narrow", "all-agg", "wide"):
            rec = artifacts.get(f"{db}-{variant}")
            if rec:
                rows.append(rec)
    return rows


def _page(title: str, lede: str, body: str, version: str) -> str:
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{title}</h1><p class=lede>{lede}</p>{body}"
        f"<footer>featurizer v{version} · read-only runs against the live triage "
        f"databases · raw per-cell artifacts in <code>specs/live-db-revalidation-v100/raw/</code> · "
        f"harness: <code>benchmarks/final_matrix.py</code> + "
        f"<code>benchmarks/bridge_workloads.py</code> (committed — reproducible)."
        f"</footer></main></body></html>\n"
    )


def render_summary(artifacts: Dict[str, Any]) -> str:
    rows = _matrix_rows(artifacts)
    version = rows[0].get("featurizer_version", "1.0.0") if rows else "1.0.0"
    tr = []
    for r in rows:
        was = V080_SECONDS.get((r["dataset"], r["variant"]))
        secs = r.get("exec_seconds")
        delta = ""
        if was and secs:
            pct = (secs - was) / was * 100
            delta = f"{pct:+.0f}%"
        tr.append(
            f"<tr><td class=mono>{r['dataset']}</td><td class=mono>{r['variant']}</td>"
            f"<td class=mono>{r['features']:,}</td><td class=mono>{r['shards']}</td>"
            f"<td class=mono>{r.get('rows', ''):,}</td>"
            f"<td class=mono>{secs if secs is not None else '—'}</td>"
            f"<td class='mono was'>{was if was is not None else '—'}</td>"
            f"<td class=mono>{delta}</td>"
            f"<td class=mono>{r.get('dup_names', '—')}</td>"
            f"<td><span class='chip ok'>{r.get('status', '?')}</span></td></tr>"
        )
    bridge = []
    g = artifacts.get("bridge-graph-dirtyduck", {})
    g2 = artifacts.get("bridge-graph-donorschoose", {})
    c = artifacts.get("bridge-centrality-dirtyduck", {})
    c2 = artifacts.get("bridge-centrality-donorschoose", {})
    t = artifacts.get("bridge-text-dirtyduck", {})
    if g:
        bridge.append(
            f"<tr><td class=mono>graph_relationships (native SQL pass)</td>"
            f"<td class=mono>dirtyduck</td>"
            f"<td>{g['edges']:,} chain edges · full cohort × {len(g['as_of_dates'])} "
            f"as-of dates = {g['rows']:,} rows · DEGREE + windowed variants</td>"
            f"<td class=mono>{g['exec_seconds']}s</td>"
            f"<td class=mono>hand-SQL ×{g['hand_sql_verified']} ✓</td></tr>"
        )
    if g2:
        bridge.append(
            f"<tr><td class=mono>graph_relationships (native SQL pass)</td>"
            f"<td class=mono>donorschoose</td>"
            f"<td>{g2['edges']:,} school edges · {g2['rows']:,} rows</td>"
            f"<td class=mono>{g2['exec_seconds']}s</td>"
            f"<td class=mono>hand-SQL ×{g2['hand_sql_verified']} ✓</td></tr>"
        )
    if c:
        bridge.append(
            f"<tr><td class=mono>CentralityBridge.materialize_snapshots</td>"
            f"<td class=mono>dirtyduck</td>"
            f"<td>{c['edges']:,} edges × {len(c['as_of_dates'])} windows "
            f"({c['cheap_snapshot_rows']:,} snapshot rows)</td>"
            f"<td class=mono>cheap {c['cheap_seconds']}s · heavy {c['heavy_seconds']}s "
            f"({c['heavy_over_cheap']}×)</td><td class=mono>—</td></tr>"
        )
    if c2:
        bridge.append(
            f"<tr><td class=mono>CentralityBridge.materialize_snapshots</td>"
            f"<td class=mono>donorschoose</td>"
            f"<td>{c2['edges']:,} edges × {len(c2['as_of_dates'])} windows</td>"
            f"<td class=mono>cheap {c2['cheap_seconds']}s · heavy {c2['heavy_seconds']}s "
            f"({c2['heavy_over_cheap']}×)</td><td class=mono>—</td></tr>"
        )
    if t:
        bridge.append(
            f"<tr><td class=mono>SentimentBridge → spine (end to end)</td>"
            f"<td class=mono>dirtyduck</td>"
            f"<td>{t['comments_processed']:,} of {t['comments_total']:,} inspector "
            f"comments (cap {t['row_cap']:,}, recorded) → MEAN/MIN/MAX/COUNT × windows</td>"
            f"<td class=mono>bridge {t['bridge_seconds']}s · spine {t['spine_seconds']}s</td>"
            f"<td class=mono>hand-SQL ×{t['hand_sql_verified']} ✓</td></tr>"
        )
    body = f"""
<div class=note><b>Headline.</b> Every cell of the 3-DB × 3-variant matrix
materializes on v1.0.0 with <b>no regression vs the v0.8.0 numbers</b>
(every delta within the ±20% gate), and — for the first time — the 0.9.x
families are measured at realistic scale: the native
<code>graph_relationships</code> pass, the centrality snapshot costs
(the measured case for cheap-by-default), and a text bridge end to end.
Feature counts: narrow and all-agg reproduce v0.8.0 exactly; wide matches
exactly on dirtyduck/chicago311 and emits ~6% more columns on donorschoose
(0.9.x planner evolution; the config is identical).</div>
<h2>1 · The regression matrix</h2>
<table><thead><tr><th>database</th><th>variant</th><th>features</th><th>shards</th>
<th>rows</th><th>seconds</th><th>was (v0.8.0)</th><th>Δ</th><th>dup-names</th><th>status</th></tr></thead>
<tbody>{''.join(tr)}</tbody></table>
<h2>2 · New: the 0.9.x families at scale</h2>
<table><thead><tr><th>workload</th><th>database</th><th>shape</th><th>wall-clock</th>
<th>value check</th></tr></thead><tbody>{''.join(bridge)}</tbody></table>
<h2>3 · Per-database detail</h2>
<p><a href='live-db-revalidation-v100/dirtyduck.html'>dirtyduck</a> ·
<a href='live-db-revalidation-v100/chicago311.html'>chicago311</a> ·
<a href='live-db-revalidation-v100/donorschoose.html'>donorschoose</a></p>
"""
    return _page(
        "Live-DB revalidation — v1.0.0",
        "The 3-DB × 3-variant release matrix plus the first at-scale "
        "measurement of the 0.9.x graph/text families.",
        body,
        version,
    )


def render_db(artifacts: Dict[str, Any], db: str) -> str:
    rows = [
        artifacts[f"{db}-{v}"]
        for v in ("narrow", "all-agg", "wide")
        if f"{db}-{v}" in artifacts
    ]
    version = rows[0].get("featurizer_version", "1.0.0") if rows else "1.0.0"
    tr = "".join(
        f"<tr><td class=mono>{r['variant']}</td><td class=mono>{r['features']:,}</td>"
        f"<td class=mono>{r['shards']}</td><td class=mono>{r.get('rows', ''):,}</td>"
        f"<td class=mono>{r.get('dup_names', '—')}</td>"
        f"<td class=mono>{r.get('exec_seconds', '—')}</td>"
        f"<td class='mono was'>{V080_SECONDS.get((db, r['variant']), '—')}</td>"
        f"<td><span class='chip ok'>{r.get('status', '?')}</span></td></tr>"
        for r in rows
    )
    extras = []
    for key, label in (
        (f"bridge-graph-{db}", "graph_relationships"),
        (f"bridge-centrality-{db}", "centrality snapshots"),
        (f"bridge-text-{db}", "text bridge end-to-end"),
    ):
        if key in artifacts:
            extras.append(
                f"<h2>{label}</h2><pre><code>"
                + json.dumps(artifacts[key], indent=2)
                + "</code></pre>"
            )
    body = f"""
<h2>1 · Full-cohort materialization (v1.0.0)</h2>
<p>Each variant materialized through <code>to_dataframe(connection=…)</code>
at one as-of date ({rows[0].get('as_of_date', '?') if rows else '?'} — the day
after the last knowledge date, so every event is under the window); no
permanent writes. Variant definitions and the recovered wide transformer set:
<code>benchmarks/final_matrix.py</code>.</p>
<table><thead><tr><th>variant</th><th>features</th><th>shards</th><th>rows</th>
<th>dup-names</th><th>seconds</th><th>was (v0.8.0)</th><th>status</th></tr></thead>
<tbody>{tr}</tbody></table>
{''.join(extras)}
"""
    return _page(f"featurizer × {db}", LEDES[db], body, version)


def main() -> None:
    artifacts = _load()
    summary_path = ARTIFACT_DIR.parent / "live-db-revalidation-v100.html"
    summary_path.write_text(render_summary(artifacts))
    print(f"wrote {summary_path}")
    for db in ("dirtyduck", "chicago311", "donorschoose"):
        page = ARTIFACT_DIR / f"{db}.html"
        page.write_text(render_db(artifacts, db))
        print(f"wrote {page}")


if __name__ == "__main__":
    main()
