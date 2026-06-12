"""DonorsChoose (KDD Cup 2014) seeder — DSSG pre-sampled Triage dump.

Source: the prepared PostgreSQL dump used by the Triage colab quickstart
(https://dsapp-public-data-migrated.s3.us-west-2.amazonaws.com/
donors_sampled_20210920_v3.dmp, ~25 MB, custom format). No Kaggle account
needed; this is by construction the same data Triage's own tutorial runs on.

NOTE: full implementation lands in Phase 0.6 — ``seed`` currently fails
loudly so ``just seed donorschoose`` cannot silently no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SCHEMA = "donorschoose"
SEED_VERSION = 1

DUMP_URL = (
    "https://dsapp-public-data-migrated.s3.us-west-2.amazonaws.com/"
    "donors_sampled_20210920_v3.dmp"
)


def download(cache_dir: Path) -> dict[str, Path]:
    from ._download import cached_download

    dataset_dir = cache_dir / "donorschoose"
    dump = cached_download(DUMP_URL, dataset_dir / "donors_sampled_20210920_v3.dmp")
    return {"dump": dump}


def seed(conn: Any, cache_dir: Path) -> None:
    raise SystemExit(
        "donorschoose seeding is not implemented yet (Phase 0.6); "
        "run `just seed food` for now."
    )
