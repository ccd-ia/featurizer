"""Seed metadata: a one-row bookkeeping table per seeded schema.

``<schema>.seed_meta`` lets the test fixtures detect (a) whether a dataset is
seeded at all and (b) whether it was seeded by the version of the seeder the
tests expect. ``subset_sha256`` fingerprints the cached raw subset so tests
with frozen hand-verified constants can skip themselves when the upstream
portal has amended history (Chicago does).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import psycopg


@dataclass(frozen=True)
class SeedStatus:
    dataset: str
    seed_version: int
    row_counts: dict[str, int]
    subset_sha256: str


def mark_seeded(
    conn: Any,
    schema: str,
    *,
    dataset: str,
    version: int,
    row_counts: dict[str, int],
    subset_sha256: str,
) -> None:
    """Write the single bookkeeping row for ``schema`` (replacing any prior)."""
    with conn.cursor() as cur:
        cur.execute(f"""
            create table if not exists {schema}.seed_meta (
                dataset text not null,
                seed_version integer not null,
                seeded_at timestamptz not null default now(),
                row_counts jsonb not null,
                subset_sha256 text not null
            )
            """)
        cur.execute(f"delete from {schema}.seed_meta")
        cur.execute(
            f"insert into {schema}.seed_meta "
            "(dataset, seed_version, row_counts, subset_sha256) "
            "values (%s, %s, %s, %s)",
            (dataset, version, json.dumps(row_counts), subset_sha256),
        )


def seed_status(conn: Any, schema: str) -> SeedStatus | None:
    """Return the seed status for ``schema``, or None if it is not seeded."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"select dataset, seed_version, row_counts, subset_sha256 "
                f"from {schema}.seed_meta"
            )
            row = cur.fetchone()
    except (psycopg.errors.UndefinedTable, psycopg.errors.InvalidSchemaName):
        conn.rollback()
        return None
    if row is None:
        return None
    dataset, version, row_counts, subset_sha256 = row
    return SeedStatus(
        dataset=dataset,
        seed_version=version,
        row_counts=dict(row_counts),
        subset_sha256=subset_sha256,
    )
