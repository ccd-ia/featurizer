#!/usr/bin/env python
"""Generate sample social-posts data for Example 6 (loads into PostgreSQL).

A coordination-detection scenario: six authors post short Spanish texts.
Three of them (``a1``–``a3``) paste the *same* campaign message at staggered
dates — the copy-paste signature — while ``a4``–``a6`` post distinct organic
messages. The example's bridges turn that shared text into an edge table, the
edges into per-window centrality snapshots, and the SQL spine into a feature
matrix where the coordinated cluster pops out.

Run via ``just example 06`` (which starts the throwaway database first), or
set DATABASE_URL / PG* and run directly. See ``examples/_db.py``.
"""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/ for _db
import _db

SCHEMA = "example_06"

# The pasted campaign text (negative register — the sentiment bridge scores
# it without any model or extra dependency).
CAMPAIGN = (
    "Terrible el nuevo reglamento del corredor: otro retraso pésimo y un "
    "riesgo grave para todos los transportistas locales"
)

# (post_id, author_id, posted_at, body)
POSTS = [
    # The coordinated cluster: identical text, staggered dates. The pair
    # (a1, a2) becomes knowable on 2024-02-20; a3 joins on 2024-05-10.
    (1, "a1", date(2024, 1, 15), CAMPAIGN),
    (2, "a2", date(2024, 2, 20), CAMPAIGN),
    (3, "a3", date(2024, 5, 10), CAMPAIGN),
    # Organic authors: distinct texts, mixed sentiment.
    (
        4,
        "a4",
        date(2024, 1, 20),
        "Excelente jornada en el puerto, servicio rápido y confiable",
    ),
    (
        5,
        "a5",
        date(2024, 3, 5),
        "La asamblea revisó el calendario de obras sin mayores cambios",
    ),
    (
        6,
        "a6",
        date(2024, 4, 12),
        "Buena respuesta del operador aunque persiste una falla menor",
    ),
]

# The backtest cohort dates: at the first, only the (a1, a2) pair exists;
# at the second, the full triangle does.
AS_OF_DATES = [date(2024, 3, 31), date(2024, 6, 30)]


def main() -> None:
    conn = _db.connect(SCHEMA)
    cur = conn.cursor()

    cur.execute("CREATE TABLE authors (author_id TEXT PRIMARY KEY)")
    cur.executemany(
        "INSERT INTO authors VALUES (%s)",
        [(f"a{i}",) for i in range(1, 7)],
    )

    cur.execute("""
        CREATE TABLE posts (
            post_id INTEGER PRIMARY KEY,
            author_id TEXT NOT NULL REFERENCES authors,
            posted_at DATE NOT NULL,
            body TEXT NOT NULL
        )
        """)
    cur.executemany("INSERT INTO posts VALUES (%s, %s, %s, %s)", POSTS)

    cur.execute("CREATE TABLE as_of_dates (as_of_date DATE PRIMARY KEY)")
    cur.executemany("INSERT INTO as_of_dates VALUES (%s)", [(d,) for d in AS_OF_DATES])

    conn.commit()
    conn.close()
    print(
        f"✓ Seeded schema {SCHEMA}: 6 authors, {len(POSTS)} posts, "
        f"{len(AS_OF_DATES)} as-of dates"
    )


if __name__ == "__main__":
    main()
