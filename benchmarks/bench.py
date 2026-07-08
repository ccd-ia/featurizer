"""Synthetic scaling benchmark for the subquery-aggregator tier.

Measures the ADR-0009 cliff on a controlled dataset (parents T ∈ {100, 1k, 10k},
~20 children each) so the set-based rewrite has a before/after curve. Three
tiers per scale: the narrow default set (control — must stay fast), each
subquery family in isolation, and the all-agg set. Per-run wall-clock is
recorded; a run exceeding ``--timeout`` is marked censored (not crashed).

Output: ``specs/correlated-subquery-aggregator-scaling/<label>.json`` plus a
readable ``<label>.html`` scaling table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from . import _db, preagg_cases

ARTIFACT_DIR = (
    Path(__file__).resolve().parent.parent
    / "specs"
    / "correlated-subquery-aggregator-scaling"
)

SCALES: Dict[str, int] = {"100": 100, "1k": 1000, "10k": 10000}

# The curated default-active set used by the fast tier (kept small + index-free
# of correlated subqueries). Mirrors a realistic narrow config.
_NARROW = ["count", "sum", "mean", "min", "max", "stddev", "nunique"]


def _synth_rows(n_parents: int, children_each: int = 20) -> List[Tuple[Any, ...]]:
    """Deterministic synthetic child stream, tie-free per parent."""
    rows: List[Tuple[Any, ...]] = []
    cats = ("a", "b", "c", "d", "e")
    for pid in range(1, n_parents + 1):
        day = 0
        for j in range(children_each):
            day += 1 + ((pid + j) % 4)
            month = 1 + (day // 28) % 12
            dom = 1 + (day % 28)
            ts = f"2022-{month:02d}-{dom:02d}"
            num = float(((pid * 3 + j * 7) % 101) - 30)
            cat = cats[(pid + j) % len(cats)]
            rows.append((pid, ts, num, cat))
    return rows


def _seed_scale(conn: Any, n_parents: int, ts_type: str) -> None:
    with conn.cursor() as cur:
        cur.execute("drop table if exists p, c, as_of_dates")
    with conn.cursor() as cur:
        cur.execute("create temp table p (pid int) on commit drop")
        cur.executemany(
            "insert into p values (%s)", [(k,) for k in range(1, n_parents + 1)]
        )
        cur.execute(
            f"create temp table c (pid int, ts {ts_type}, num numeric, cat text) "
            "on commit drop"
        )
        cur.executemany("insert into c values (%s, %s, %s, %s)", _synth_rows(n_parents))
        cur.execute("create temp table as_of_dates (as_of_date date) on commit drop")
        cur.execute("insert into as_of_dates values ('2023-01-01')")


def _config(aggs: List[str], interval: str | None) -> Dict[str, Any]:
    cfg = preagg_cases.config(aggs[0], "date", interval)
    cfg["aggregations"] = aggs
    return cfg


def _time_run(
    conn: Any, aggs: List[str], interval: str | None, timeout_s: float
) -> Dict[str, Any]:
    """Time one config. Uses a statement_timeout so a runaway run is censored."""
    cfg = _config(aggs, interval)
    with conn.cursor() as cur:
        cur.execute(f"set local statement_timeout = {int(timeout_s * 1000)}")
    start = time.perf_counter()
    try:
        rows = _db.run_config(conn, cfg)
        elapsed = time.perf_counter() - start
        return {"seconds": round(elapsed, 3), "rows": len(rows), "censored": False}
    except Exception as exc:  # noqa: BLE001 - timeout surfaces as an error we tag
        conn.rollback()
        return {
            "seconds": round(time.perf_counter() - start, 3),
            "censored": True,
            "error": str(exc).splitlines()[0][:200],
        }


def explain(conn: Any, aggs: List[str], interval: str | None) -> Any:
    """EXPLAIN (ANALYZE, FORMAT JSON) for one config, or an error string."""
    cfg = _config(aggs, interval)
    import tempfile

    import yaml

    from featurizer import Featurizer

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as h:
        yaml.safe_dump(cfg, h)
        path = h.name
    sql = Featurizer(path, validate=False).query
    with conn.cursor() as cur:
        try:
            cur.execute(f"explain (analyze, format json) {sql}")
            return cur.fetchone()[0]
        except Exception as exc:  # noqa: BLE001
            conn.rollback()
            return {"error": str(exc).splitlines()[0][:200]}


def run(scale_label: str, timeout_s: float = 300.0) -> Dict[str, Any]:
    n_parents = SCALES[scale_label]
    families = preagg_cases.migratable_aggregators()
    conn = _db.connect()
    result: Dict[str, Any] = {
        "scale": scale_label,
        "n_parents": n_parents,
        "timeout_s": timeout_s,
        "narrow": None,
        "all_agg": None,
        "per_family": {},
    }
    try:
        _seed_scale(conn, n_parents, "date")
        print(f"[{scale_label}] narrow tier ({len(_NARROW)} aggs)...", flush=True)
        result["narrow"] = _time_run(conn, _NARROW, None, timeout_s)
        print(f"  narrow: {result['narrow']}", flush=True)

        for fam in families:
            r = _time_run(conn, [fam, "count"], None, timeout_s)
            result["per_family"][fam] = r
            print(f"  {fam}: {r}", flush=True)

        print(f"[{scale_label}] all-agg tier...", flush=True)
        result["all_agg"] = _time_run(conn, families + ["count"], None, timeout_s)
        print(f"  all_agg: {result['all_agg']}", flush=True)
    finally:
        conn.rollback()
        conn.close()
    return result


def _fmt(run_result: Dict[str, Any] | None) -> str:
    if not run_result:
        return "—"
    if run_result.get("censored"):
        return f'&gt;{run_result["seconds"]}s (censored)'
    return f'{run_result["seconds"]}s'


def render_html(document: Dict[str, Any], label: str) -> str:
    scales = document["scales"]
    families = sorted({f for s in scales.values() for f in s.get("per_family", {})})
    scale_labels = list(scales.keys())
    head = "".join(f"<th>{s}</th>" for s in scale_labels)

    def _row(label_html: str, pick) -> str:
        cells = "".join(f"<td>{_fmt(pick(scales[s]))}</td>" for s in scale_labels)
        return f"<tr><td>{label_html}</td>{cells}</tr>"

    rows = [
        _row("narrow (control)", lambda s: s["narrow"]),
        _row("<strong>all-agg</strong>", lambda s: s["all_agg"]),
    ]
    for fam in families:
        rows.append(_row(fam, lambda s, f=fam: s.get("per_family", {}).get(f)))
    body = "\n".join(rows)
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Subquery aggregator scaling — {label}</title>
<style>body{{font-family:-apple-system,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1a2332}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}}th,td{{border:1px solid #e2e4e0;padding:.35rem .5rem;text-align:left}}
th{{background:#ccfbf1;color:#0f766e}}td:first-child{{font-family:monospace}}h1{{font-size:1.4rem}}</style></head>
<body><h1>Subquery aggregator scaling — {label}</h1>
<p>Wall-clock per config at each parent-count scale (~20 children/parent). Timeout {document.get('timeout_s', 300)}s.</p>
<table><tr><th>config</th>{head}</tr>
{body}
</table></body></html>
"""


def write_artifacts(document: Dict[str, Any], label: str) -> Tuple[Path, Path]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = ARTIFACT_DIR / f"{label}.json"
    html_path = ARTIFACT_DIR / f"{label}.html"
    json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    html_path.write_text(render_html(document, label))
    return json_path, html_path
