"""Seeding CLI for the realistic-dataset integration tier.

Usage (normally via just)::

    DATABASE_URL=... uv run python -m tests.integration.datasets seed [all|food|donorschoose]
"""

from __future__ import annotations

import argparse
import sys

from . import CACHE_DIR
from ._db import connect_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.integration.datasets",
        description="Download (cached) and seed realistic test datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seed_parser = subparsers.add_parser(
        "seed", help="Seed dataset(s) into the test database"
    )
    seed_parser.add_argument(
        "dataset",
        nargs="?",
        default="all",
        choices=["all", "food", "donorschoose"],
        help="Which dataset to seed (default: all)",
    )
    args = parser.parse_args(argv)

    # Imported lazily so `--help` works without a configured database.
    from . import donorschoose, food_inspections

    targets = {
        "food": [food_inspections],
        "donorschoose": [donorschoose],
        "all": [food_inspections, donorschoose],
    }[args.dataset]

    conn = connect_from_env()
    try:
        for module in targets:
            print(f"== seeding {module.SCHEMA} (v{module.SEED_VERSION}) ==")
            module.seed(conn, CACHE_DIR)
            conn.commit()
            print(f"== {module.SCHEMA} done ==")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
