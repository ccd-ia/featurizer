"""DonorsChoose (KDD Cup 2014) seeder — DSSG pre-sampled Triage dump.

Source: the prepared PostgreSQL dump used by the Triage colab quickstart
(~25 MB, custom format, already sampled to ~16k projects). No Kaggle account
needed; this is by construction the same data Triage's own tutorial runs on.

The dump restores a raw ``data`` schema (projects/donations/essays/outcomes/
resources/time_series_features). We derive the normalized ``donorschoose``
schema from it and drop ``data`` afterwards:

- ``projects``      — one row per project (``projectid`` = the dump's integer
  ``entity_id``); ``date_posted`` kept as **timestamp** so the temporal_ix
  path differs from the food dataset's date columns.
- ``donations``     — the event stream (amounts, donor ids, payment method).
- ``resources``     — static child (declared without temporal_ix in configs).
- ``schools``       — derived entity for the depth-3 config
  (schools -> projects -> donations); ``first_seen`` = min(date_posted).
- ``project_edges`` — co-donation edges: projects sharing a donor (donors
  with 2..20 distinct projects), one row per unordered pair,
  ``knowable_at`` = the earliest moment any shared donor had donated to both.

Idempotent: schemas are dropped and rebuilt from the cached dump.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._db import conninfo_from_env
from ._download import cached_download, run_pg_restore, sha256_of
from ._meta import mark_seeded

SCHEMA = "donorschoose"
RAW_SCHEMA = "data"  # created by the dump; dropped after derivation
SEED_VERSION = 1

DUMP_URL = (
    "https://dsapp-public-data-migrated.s3.us-west-2.amazonaws.com/"
    "donors_sampled_20210920_v3.dmp"
)


def download(cache_dir: Path) -> dict[str, Path]:
    dataset_dir = cache_dir / "donorschoose"
    dump = cached_download(DUMP_URL, dataset_dir / "donors_sampled_20210920_v3.dmp")
    return {"dump": dump}


def seed(conn: Any, cache_dir: Path) -> None:
    """Restore the dump and derive the normalized ``donorschoose`` schema."""
    files = download(cache_dir)

    # pg_restore runs as a separate process, so the schema drops must be
    # committed before it starts.
    with conn.cursor() as cur:
        cur.execute(f"drop schema if exists {RAW_SCHEMA} cascade")
        cur.execute(f"drop schema if exists {SCHEMA} cascade")
    conn.commit()
    run_pg_restore(files["dump"], conninfo_from_env())

    with conn.cursor() as cur:
        cur.execute(f"create schema {SCHEMA}")

        cur.execute(f"""
            create table {SCHEMA}.projects as
            select
                entity_id as projectid,
                projectid_str,
                schoolid,
                teacher_acctid,
                school_state,
                school_latitude,
                school_longitude,
                poverty_level,
                grade_level,
                resource_type,
                total_asking_price,
                total_price_including_optional_support,
                students_reached,
                date_posted
            from {RAW_SCHEMA}.projects
            where date_posted is not null
            """)
        cur.execute(f"alter table {SCHEMA}.projects add primary key (projectid)")

        cur.execute(f"""
            create table {SCHEMA}.donations as
            select distinct on (donationid)
                donationid,
                entity_id as projectid,
                donor_acctid,
                donation_timestamp,
                donation_to_project,
                donation_total,
                payment_method,
                is_teacher_acct
            from {RAW_SCHEMA}.donations
            where entity_id in (select projectid from {SCHEMA}.projects)
              and donation_timestamp is not null
            order by donationid
            """)
        cur.execute(f"alter table {SCHEMA}.donations add primary key (donationid)")

        cur.execute(f"""
            create table {SCHEMA}.resources as
            select distinct on (resourceid)
                resourceid,
                entity_id as projectid,
                vendorid,
                project_resource_type,
                item_unit_price,
                item_quantity
            from {RAW_SCHEMA}.resources
            where entity_id in (select projectid from {SCHEMA}.projects)
              and resourceid is not null
            order by resourceid
            """)
        cur.execute(f"alter table {SCHEMA}.resources add primary key (resourceid)")

        cur.execute(f"""
            create table {SCHEMA}.schools as
            select
                schoolid,
                min(school_state) as school_state,
                min(date_posted) as first_seen,
                count(*) as n_projects
            from {SCHEMA}.projects
            where schoolid is not null
            group by schoolid
            """)
        cur.execute(f"alter table {SCHEMA}.schools add primary key (schoolid)")

        # Co-donation edges. A pair becomes knowable the first time a shared
        # donor has donated to both projects: per donor the pair is knowable
        # at greatest(t_a, t_b) of the earliest donations; across donors take
        # the min.
        cur.execute(f"""
            create table {SCHEMA}.project_edges as
            with eligible as (
                select donor_acctid
                from {SCHEMA}.donations
                where donor_acctid is not null
                group by donor_acctid
                having count(distinct projectid) between 2 and 20
            ),
            first_gift as (
                select d.donor_acctid, d.projectid,
                       min(d.donation_timestamp) as first_at
                from {SCHEMA}.donations d
                join eligible using (donor_acctid)
                group by d.donor_acctid, d.projectid
            )
            select a.projectid as source_project,
                   b.projectid as target_project,
                   min(greatest(a.first_at, b.first_at)) as knowable_at
            from first_gift a
            join first_gift b
              on a.donor_acctid = b.donor_acctid
             and a.projectid < b.projectid
            group by a.projectid, b.projectid
            """)

        cur.execute(
            f"create index on {SCHEMA}.donations (projectid, donation_timestamp)"
        )
        cur.execute(f"create index on {SCHEMA}.resources (projectid)")
        cur.execute(f"create index on {SCHEMA}.projects (schoolid)")
        cur.execute(f"create index on {SCHEMA}.project_edges (source_project)")
        cur.execute(f"create index on {SCHEMA}.project_edges (target_project)")

        counts: dict[str, int] = {}
        for table in ("projects", "donations", "resources", "schools", "project_edges"):
            cur.execute(f"select count(*) from {SCHEMA}.{table}")
            counts[table] = cur.fetchone()[0]
        print(f"  row counts: {counts}")

        cur.execute(f"drop schema {RAW_SCHEMA} cascade")

    mark_seeded(
        conn,
        SCHEMA,
        dataset="donorschoose-kdd2014-dssg-sample",
        version=SEED_VERSION,
        row_counts=counts,
        subset_sha256=sha256_of(files["dump"]),
    )
    with conn.cursor() as cur:
        for table in ("projects", "donations", "resources", "schools", "project_edges"):
            cur.execute(f"analyze {SCHEMA}.{table}")
