"""Realistic-dataset seeding for the PostgreSQL integration harness.

This package downloads two public, Triage-compatible datasets (cached under
the gitignored ``tests/data/``) and loads them into per-dataset schemas of the
test database:

- ``food_inspections`` — Chicago Food Inspections + Business Licenses
  (the dataset used by the dirtyduck/Triage tutorial).
- ``donorschoose`` — the DSSG pre-sampled DonorsChoose KDD Cup 2014 dump
  (the dataset used by the Triage colab quickstart).

Seeding is idempotent (drop schema cascade + recreate from the cached,
deterministically subsetted files). Connection comes exclusively from
``DATABASE_URL`` / ``PG*`` env vars — there is no default; the CLI fails
loudly when unset.

Usage::

    just db-up
    just seed            # or: seed food / seed donorschoose
    just test-realistic
"""

from __future__ import annotations

from pathlib import Path

#: Repository root (…/featurizer), derived from this file's location.
REPO_ROOT = Path(__file__).resolve().parents[3]

#: Download cache for raw dataset files. Gitignored; safe to delete.
CACHE_DIR = REPO_ROOT / "tests" / "data"
