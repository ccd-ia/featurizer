"""Chicago Food Inspections + Business Licenses seeder.

The dataset behind the dirtyduck/Triage tutorial. A deterministic historical
subset (2014-2016, ordered by unique id, capped) is downloaded from the
Chicago Open Data portal and normalized into the ``food_inspections`` schema:

- ``facilities``   — one row per license number (entity), derived from
  inspections; ``first_seen`` = earliest inspection date.
- ``inspections``  — the event stream (results / inspection_type categorical,
  derived ``violation_count`` numeric).
- ``licenses``     — second event stream for as-of joins, filtered to license
  numbers present in ``facilities``.
- ``chain_edges``  — derived edge table (facilities sharing a normalized
  dba_name, chain size 2..50), with a causal ``knowable_at`` timestamp;
  one row per unordered pair (least, greatest).

Idempotent: drop schema cascade + recreate from the cached files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ._download import (
    cached_download as _cached_download,
)  # noqa: F401  (re-export for tests)
from ._download import download_socrata_csv, sha256_of
from ._meta import mark_seeded

SCHEMA = "food_inspections"
SEED_VERSION = 1

INSPECTIONS_URL = "https://data.cityofchicago.org/resource/4ijn-s7e5.csv"
LICENSES_URL = "https://data.cityofchicago.org/resource/r5kz-chrr.csv"

# Closed historical window so re-downloads are near-stable (the portal can
# still amend old rows — seed_meta.subset_sha256 guards frozen test constants).
INSPECTIONS_WHERE = (
    "inspection_date between '2014-01-01T00:00:00' and '2016-12-31T23:59:59' "
    "and license_ is not null"
)
INSPECTIONS_SELECT = (
    "inspection_id,dba_name,license_,facility_type,risk,address,zip,"
    "inspection_date,inspection_type,results,violations,latitude,longitude"
)
INSPECTIONS_MAX_ROWS = 120_000

# Wider start so the as-of grace window (P2Y) has license history before 2014.
LICENSES_WHERE = (
    "license_start_date between '2012-01-01T00:00:00' and '2016-12-31T23:59:59'"
)
LICENSES_SELECT = (
    "id,license_number,application_type,license_status,license_start_date,"
    "license_description"
)
LICENSES_MAX_ROWS = 200_000


def download(cache_dir: Path) -> dict[str, Path]:
    """Fetch the two CSV subsets into the cache; return their paths."""
    dataset_dir = cache_dir / "food_inspections"
    inspections = download_socrata_csv(
        INSPECTIONS_URL,
        dataset_dir / "inspections_2014_2016.csv",
        select=INSPECTIONS_SELECT,
        where=INSPECTIONS_WHERE,
        order="inspection_id",
        max_rows=INSPECTIONS_MAX_ROWS,
    )
    licenses = download_socrata_csv(
        LICENSES_URL,
        dataset_dir / "licenses_2012_2016.csv",
        select=LICENSES_SELECT,
        where=LICENSES_WHERE,
        order="id",
        max_rows=LICENSES_MAX_ROWS,
    )
    return {"inspections": inspections, "licenses": licenses}


def _copy_csv(conn: Any, table: str, columns: str, path: Path) -> None:
    with conn.cursor() as cur, path.open("rb") as handle:
        with cur.copy(
            f"copy {table} ({columns}) from stdin with (format csv, header true)"
        ) as copy:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                copy.write(chunk)


def seed(conn: Any, cache_dir: Path) -> None:
    """Load the cached subset into the ``food_inspections`` schema."""
    files = download(cache_dir)
    with conn.cursor() as cur:
        cur.execute(f"drop schema if exists {SCHEMA} cascade")
        cur.execute(f"create schema {SCHEMA}")

        # -- staging (temp, all-text: CSV lands as-is, casts happen in SQL) --
        cur.execute("""
            create temp table _stage_inspections (
                inspection_id text, dba_name text, license_ text,
                facility_type text, risk text, address text, zip text,
                inspection_date text, inspection_type text, results text,
                violations text, latitude text, longitude text
            ) on commit drop
            """)
        cur.execute("""
            create temp table _stage_licenses (
                id text, license_number text, application_type text,
                license_status text, license_start_date text,
                license_description text
            ) on commit drop
            """)
    _copy_csv(
        conn,
        "_stage_inspections",
        "inspection_id, dba_name, license_, facility_type, risk, address, zip, "
        "inspection_date, inspection_type, results, violations, latitude, longitude",
        files["inspections"],
    )
    _copy_csv(
        conn,
        "_stage_licenses",
        "id, license_number, application_type, license_status, license_start_date, "
        "license_description",
        files["licenses"],
    )

    with conn.cursor() as cur:
        # -- inspections: typed event stream with derived violation_count --
        cur.execute(f"""
            create table {SCHEMA}.inspections as
            select distinct on (inspection_id::bigint)
                inspection_id::bigint as inspection_id,
                license_::bigint as license_no,
                (inspection_date::timestamp)::date as inspection_date,
                inspection_type,
                results,
                coalesce(array_length(string_to_array(violations, ' | '), 1), 0)
                    as violation_count,
                nullif(latitude, '')::double precision as latitude,
                nullif(longitude, '')::double precision as longitude
            from _stage_inspections
            where license_ ~ '^[0-9]+$' and license_ <> '0'
            order by inspection_id::bigint
            """)
        cur.execute(f"alter table {SCHEMA}.inspections add primary key (inspection_id)")

        # -- facilities: one row per license_no; attributes from the earliest
        #    inspection record; first_seen = min(inspection_date) --
        cur.execute(f"""
            create table {SCHEMA}.facilities as
            with attrs as (
                select distinct on (license_::bigint)
                    license_::bigint as license_no,
                    dba_name, facility_type, risk, address, zip,
                    nullif(latitude, '')::double precision as latitude,
                    nullif(longitude, '')::double precision as longitude
                from _stage_inspections
                where license_ ~ '^[0-9]+$' and license_ <> '0'
                order by license_::bigint, inspection_date, inspection_id::bigint
            ),
            seen as (
                select license_no, min(inspection_date) as first_seen
                from {SCHEMA}.inspections
                group by license_no
            )
            select attrs.license_no, attrs.dba_name, attrs.facility_type,
                   attrs.risk, attrs.address, attrs.zip,
                   attrs.latitude, attrs.longitude, seen.first_seen
            from attrs
            join seen using (license_no)
            """)
        cur.execute(f"alter table {SCHEMA}.facilities add primary key (license_no)")

        # -- licenses: second stream, only for facilities we actually have --
        cur.execute(f"""
            create table {SCHEMA}.licenses as
            select distinct on (id)
                id as license_record_id,
                license_number::bigint as license_no,
                application_type,
                license_status,
                (license_start_date::timestamp)::date as license_start_date,
                license_description
            from _stage_licenses
            where license_number ~ '^[0-9]+$'
              and license_start_date <> ''
              and license_number::bigint in
                  (select license_no from {SCHEMA}.facilities)
            order by id
            """)
        cur.execute(
            f"alter table {SCHEMA}.licenses add primary key (license_record_id)"
        )

        # -- chain_edges: facilities sharing a normalized dba_name (2..50),
        #    one row per unordered pair, knowable once both exist --
        cur.execute(f"""
            create table {SCHEMA}.chain_edges as
            with named as (
                select license_no,
                       lower(trim(dba_name)) as chain_name,
                       first_seen
                from {SCHEMA}.facilities
                where dba_name is not null and trim(dba_name) <> ''
            ),
            chains as (
                select chain_name
                from named
                group by chain_name
                having count(*) between 2 and 50
            )
            select a.license_no as source_license,
                   b.license_no as target_license,
                   greatest(a.first_seen, b.first_seen) as knowable_at
            from named a
            join named b using (chain_name)
            join chains using (chain_name)
            where a.license_no < b.license_no
            """)

        # -- indexes for the join patterns the generated SQL uses --
        cur.execute(
            f"create index on {SCHEMA}.inspections (license_no, inspection_date)"
        )
        cur.execute(
            f"create index on {SCHEMA}.licenses (license_no, license_start_date)"
        )
        cur.execute(f"create index on {SCHEMA}.chain_edges (source_license)")
        cur.execute(f"create index on {SCHEMA}.chain_edges (target_license)")

        counts: dict[str, int] = {}
        for table in ("facilities", "inspections", "licenses", "chain_edges"):
            cur.execute(f"select count(*) from {SCHEMA}.{table}")
            counts[table] = cur.fetchone()[0]
        print(f"  row counts: {counts}")

    combined = hashlib.sha256(
        (sha256_of(files["inspections"]) + sha256_of(files["licenses"])).encode()
    ).hexdigest()
    mark_seeded(
        conn,
        SCHEMA,
        dataset="chicago-food-inspections",
        version=SEED_VERSION,
        row_counts=counts,
        subset_sha256=combined,
    )
    with conn.cursor() as cur:
        cur.execute(f"analyze {SCHEMA}.facilities")
        cur.execute(f"analyze {SCHEMA}.inspections")
        cur.execute(f"analyze {SCHEMA}.licenses")
        cur.execute(f"analyze {SCHEMA}.chain_edges")
