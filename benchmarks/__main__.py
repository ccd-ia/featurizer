"""CLI for the benchmark + golden-capture tooling.

    uv run python -m benchmarks inventory
    uv run python -m benchmarks capture-golden
    uv run python -m benchmarks bench --scale 1k [--timeout 300] [--label baseline]
    uv run python -m benchmarks explain --scale 1k --family gap_mean

Needs DATABASE_URL / PG* for every subcommand except ``inventory``.
"""

from __future__ import annotations

import argparse
import json

from . import bench, capture_golden, preagg_cases


def _cmd_inventory(_: argparse.Namespace) -> None:
    subq = preagg_cases.subquery_aggregators()
    migratable = preagg_cases.migratable_aggregators()
    special = sorted(preagg_cases.NEEDS_SPECIAL_CONFIG & set(subq))
    print(f"subquery aggregators ({len(subq)}):")
    for name in subq:
        tag = " [special-config]" if name in preagg_cases.NEEDS_SPECIAL_CONFIG else ""
        print(f"  {name}{tag}")
    print(f"\nmigratable (rewrite scope): {len(migratable)}")
    print(f"stays correlated (special-config subq): {len(special)} -> {special}")
    print(f"golden case count: {len(preagg_cases.cases())}")


def _cmd_capture_golden(_: argparse.Namespace) -> None:
    capture_golden.main()


def _cmd_bench(args: argparse.Namespace) -> None:
    label = args.label or f"scale-{args.scale}"
    document = {
        "timeout_s": args.timeout,
        "scales": {args.scale: bench.run(args.scale, timeout_s=args.timeout)},
    }
    json_path, html_path = bench.write_artifacts(document, label)
    print(f"\nwrote {json_path}\nwrote {html_path}")


def _cmd_explain(args: argparse.Namespace) -> None:
    conn = bench._db.connect()
    try:
        bench._seed_scale(conn, bench.SCALES[args.scale], "date")
        plan = bench.explain(conn, [args.family, "count"], None)
    finally:
        conn.rollback()
        conn.close()
    print(json.dumps(plan, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("inventory", help="list subquery aggregators (no DB)").set_defaults(
        func=_cmd_inventory
    )
    sub.add_parser(
        "capture-golden", help="freeze v0.5.2 values to the golden JSON"
    ).set_defaults(func=_cmd_capture_golden)

    p_bench = sub.add_parser("bench", help="run the scaling benchmark")
    p_bench.add_argument("--scale", choices=sorted(bench.SCALES), default="1k")
    p_bench.add_argument("--timeout", type=float, default=300.0)
    p_bench.add_argument("--label", default=None)
    p_bench.set_defaults(func=_cmd_bench)

    p_explain = sub.add_parser("explain", help="EXPLAIN ANALYZE one family")
    p_explain.add_argument("--scale", choices=sorted(bench.SCALES), default="1k")
    p_explain.add_argument("--family", required=True)
    p_explain.set_defaults(func=_cmd_explain)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
