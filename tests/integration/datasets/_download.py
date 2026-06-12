"""Download helpers: cached HTTP fetches, Socrata CSV paging, pg_restore.

Only the standard library is used (urllib), so the seeding CLI adds no
dependencies to the project. All downloads land in the gitignored cache
directory and are reused on subsequent runs — delete the file to re-fetch.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_USER_AGENT = "featurizer-integration-tests/1.0"


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def cached_download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` unless a non-empty cached copy exists."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=600) as response,
        tmp.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    tmp.rename(dest)
    return dest


def download_socrata_csv(
    resource_url: str,
    dest: Path,
    *,
    select: str,
    where: str,
    order: str,
    max_rows: int,
    page_size: int = 50_000,
) -> Path:
    """Download a deterministic CSV subset of a Socrata resource, with paging.

    ``order`` must be a unique column so pages are stable across requests.
    The header line is written once; subsequent pages have theirs stripped.
    """
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    rows_written = 0
    with tmp.open("wb") as out:
        offset = 0
        while offset < max_rows:
            limit = min(page_size, max_rows - offset)
            params = urllib.parse.urlencode(
                {
                    "$select": select,
                    "$where": where,
                    "$order": order,
                    "$limit": str(limit),
                    "$offset": str(offset),
                }
            )
            page = _fetch(f"{resource_url}?{params}")
            lines = page.splitlines(keepends=True)
            if not lines:
                break
            header, body = lines[0], lines[1:]
            if offset == 0:
                out.write(header)
            if not body:
                break
            out.writelines(body)
            rows_written += len(body)
            print(
                f"  fetched {rows_written} rows from {resource_url.rsplit('/', 1)[-1]}"
            )
            if len(body) < limit:
                break  # final page
            offset += limit
    tmp.rename(dest)
    return dest


def run_pg_restore(dump_path: Path, conninfo: str, *, jobs: int = 4) -> None:
    """Restore a custom-format PostgreSQL dump via the ``pg_restore`` binary.

    ``--clean --if-exists`` makes re-runs idempotent (objects are dropped and
    recreated). ``--no-owner`` avoids role mismatches on the test database.
    """
    if shutil.which("pg_restore") is None:
        raise SystemExit(
            "pg_restore not found on PATH. Install the PostgreSQL client tools "
            "(e.g. `brew install libpq` or `brew install postgresql@16`), or "
            "restore manually inside the container:\n"
            f"  docker exec -i featurizer-pg pg_restore -U postgres -d featurizer_test "
            f"--clean --if-exists --no-owner < {dump_path}"
        )
    command = [
        "pg_restore",
        "--clean",
        "--if-exists",
        "--no-owner",
        "--jobs",
        str(jobs),
        "--dbname",
        conninfo if conninfo else "postgresql://",
        str(dump_path),
    ]
    print(f"  pg_restore {dump_path.name}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        # pg_restore with --clean emits ignorable "does not exist" noise only
        # with --if-exists; a non-zero exit is a real failure. Show everything.
        sys.stderr.write(result.stderr)
        raise SystemExit(f"pg_restore failed with exit code {result.returncode}")


def sha256_of(path: Path) -> str:
    """Streaming SHA-256 of a file (used to fingerprint the cached subset)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
