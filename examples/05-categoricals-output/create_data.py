#!/usr/bin/env python
"""Generate sample food-inspections data for Example 5 (loads into PostgreSQL).

A facilities target with a direct categorical (``facility_type``) and an
identifier (``name``), plus an ``inspections`` child event stream used to show
count-vs-measure imputation. The data deliberately includes an
out-of-vocabulary ``facility_type`` and a NULL one (both must one-hot to an
all-zero row) and some facilities with zero inspections (so a measure
aggregate is NULL and gets a ``__missing`` flag).

Run via ``just example 05`` (which starts the throwaway database first), or set
DATABASE_URL / PG* and run directly. See ``examples/_db.py``.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/ for _db
import _db

SCHEMA = "example_05"
NUM_FACILITIES = 30

# The DECLARED, fixed vocabulary (also lives in config.yaml). Featurizer one-hot
# encodes exactly these; it never learns the set from the data.
FACILITY_TYPE_VOCABULARY = ["Bakery", "Grocery Store", "Restaurant", "School"]
# Injected on purpose: an out-of-vocabulary value and a NULL. Both must produce
# an all-zero one-hot row (never a crash).
OUT_OF_VOCABULARY = "Food Truck"

random.seed(42)


def _facility_type(i: int) -> str | None:
    """Assign a facility_type, including the two all-zero cases."""
    if i == NUM_FACILITIES - 1:
        return None  # NULL -> all-zero one-hot
    if i == NUM_FACILITIES:
        return OUT_OF_VOCABULARY  # out-of-vocabulary -> all-zero one-hot
    return FACILITY_TYPE_VOCABULARY[i % len(FACILITY_TYPE_VOCABULARY)]


def create_database():
    """Load sample data into the ``example_05`` schema on PostgreSQL."""
    conn = _db.connect(SCHEMA)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE facilities (
            license_no    INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            facility_type TEXT,
            first_seen    DATE NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE inspections (
            inspection_id   INTEGER PRIMARY KEY,
            license_no      INTEGER NOT NULL,
            inspection_date DATE NOT NULL,
            score           DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (license_no) REFERENCES facilities(license_no)
        )
    """)
    cursor.execute("CREATE TABLE as_of_dates (as_of_date DATE PRIMARY KEY)")

    base = datetime(2023, 1, 1)
    facilities = []
    for i in range(1, NUM_FACILITIES + 1):
        first_seen = base + timedelta(days=random.randint(0, 300))
        facilities.append(
            (i, f"Facility {i:02d}", _facility_type(i), first_seen.date())
        )
    cursor.executemany("INSERT INTO facilities VALUES (%s, %s, %s, %s)", facilities)

    # Inspections: facility i gets (i % 5) inspections, so every 5th facility has
    # ZERO -> its MEAN(score) is NULL (a measure) and gets a __missing flag, while
    # its COUNT (count-like) imputes to 0.
    inspections = []
    inspection_id = 1
    for license_no in range(1, NUM_FACILITIES + 1):
        first_seen = facilities[license_no - 1][3]
        for _ in range(license_no % 5):
            d = datetime.strptime(str(first_seen), "%Y-%m-%d") + timedelta(
                days=random.randint(10, 300)
            )
            score = round(random.uniform(50.0, 100.0), 1)
            inspections.append((inspection_id, license_no, d.date(), score))
            inspection_id += 1
    cursor.executemany("INSERT INTO inspections VALUES (%s, %s, %s, %s)", inspections)

    cursor.executemany(
        "INSERT INTO as_of_dates VALUES (%s)",
        [(datetime(2024, 1, 1).date(),), (datetime(2024, 7, 1).date(),)],
    )

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM facilities")
    n_fac = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM inspections")
    n_insp = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM facilities WHERE facility_type IS NULL")
    n_null = cursor.fetchone()[0]
    conn.close()

    print("✓ Data loaded successfully!")
    print("\nStatistics:")
    print(f"  Facilities: {n_fac} (NULL facility_type: {n_null})")
    print(f"  Inspections: {n_insp}")
    print(f"  Declared vocabulary: {FACILITY_TYPE_VOCABULARY}")
    print(f"  Out-of-vocabulary value present: {OUT_OF_VOCABULARY!r}")
    print(f"\nSchema: {SCHEMA}")


if __name__ == "__main__":
    create_database()
