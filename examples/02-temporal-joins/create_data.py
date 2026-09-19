#!/usr/bin/env python
"""Generate sample healthcare data for Example 2 (Temporal Joins).

Loads into PostgreSQL. Run via ``just example 02`` (which starts the throwaway
database first), or set DATABASE_URL / PG* and run directly. See ``examples/_db.py``.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/ for _db
import _db

# Configuration
SCHEMA = "example_02"
NUM_PATIENTS = 50
MIN_PLANS_PER_PATIENT = 0
MAX_PLANS_PER_PATIENT = 10
MAX_ASSESSMENTS_PER_PATIENT = 4
# Assessments fall on both sides of admission on purpose: the as-of join must
# skip the ones dated after it, and `grace: P30D` the ones too far before it.
ASSESSMENT_DAYS_FROM_ADMISSION = (-75, 60)

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
TREATMENT_TYPES = ["medication", "therapy", "surgery", "monitoring", "rehabilitation"]

# Seed for reproducibility
random.seed(42)


def create_database():
    """Load temporal healthcare data into the ``example_02`` schema on PostgreSQL."""
    conn = _db.connect(SCHEMA)
    cursor = conn.cursor()

    # Create tables (bare names resolve via the search_path set by _db.connect)
    cursor.execute("""
        CREATE TABLE patients (
            patient_id INTEGER PRIMARY KEY,
            admission_date DATE NOT NULL,
            age INTEGER NOT NULL,
            severity_level TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE care_plans (
            plan_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            plan_date DATE NOT NULL,
            treatment_type TEXT NOT NULL,
            cost DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE risk_assessments (
            assessment_id INTEGER PRIMARY KEY,
            patient_id INTEGER NOT NULL,
            assessed_on DATE NOT NULL,
            risk_score DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
        )
    """)

    # Create as_of_dates table (required by Featurizer)
    cursor.execute("""
        CREATE TABLE as_of_dates (
            as_of_date DATE PRIMARY KEY
        )
    """)

    # Generate patients
    base_date = datetime(2023, 1, 1)
    patients = []

    for i in range(1, NUM_PATIENTS + 1):
        admission_date = base_date + timedelta(days=random.randint(0, 365))
        age = random.randint(25, 85)
        severity = random.choice(SEVERITY_LEVELS)

        patients.append((i, admission_date.date(), age, severity))

    cursor.executemany("INSERT INTO patients VALUES (%s, %s, %s, %s)", patients)

    # Generate care plans (temporal events)
    plans = []
    plan_id = 1

    for patient_id in range(1, NUM_PATIENTS + 1):
        patient_admission = patients[patient_id - 1][1]
        num_plans = random.randint(MIN_PLANS_PER_PATIENT, MAX_PLANS_PER_PATIENT)

        for plan_num in range(num_plans):
            # Plans occur after admission, spread over time
            days_after_admission = random.randint(1, 500)
            plan_date = datetime.strptime(
                str(patient_admission), "%Y-%m-%d"
            ) + timedelta(days=days_after_admission)

            treatment = random.choice(TREATMENT_TYPES)
            cost = round(random.uniform(500.0, 50000.0), 2)

            plans.append((plan_id, patient_id, plan_date.date(), treatment, cost))
            plan_id += 1

    cursor.executemany("INSERT INTO care_plans VALUES (%s, %s, %s, %s, %s)", plans)

    # Generate risk assessments (the timestamped lookup the as-of join reads).
    # Drawn after the care plans so the plan data keeps its seeded values.
    assessments = []
    assessment_id = 1

    for patient_id in range(1, NUM_PATIENTS + 1):
        patient_admission = patients[patient_id - 1][1]
        for _ in range(random.randint(0, MAX_ASSESSMENTS_PER_PATIENT)):
            offset = random.randint(*ASSESSMENT_DAYS_FROM_ADMISSION)
            assessed_on = patient_admission + timedelta(days=offset)
            risk_score = round(random.uniform(0.0, 1.0), 3)
            assessments.append((assessment_id, patient_id, assessed_on, risk_score))
            assessment_id += 1

    cursor.executemany(
        "INSERT INTO risk_assessments VALUES (%s, %s, %s, %s)", assessments
    )

    # Generate as_of_dates (quarterly snapshots for 2023-2024)
    as_of_dates = []
    for year in [2023, 2024]:
        for month in [1, 4, 7, 10]:  # Q1, Q2, Q3, Q4
            date = datetime(year, month, 1).date()
            as_of_dates.append((date,))

    cursor.executemany("INSERT INTO as_of_dates VALUES (%s)", as_of_dates)

    conn.commit()

    # Print statistics
    cursor.execute("SELECT COUNT(*) FROM patients")
    num_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM care_plans")
    num_plans = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM as_of_dates")
    num_dates = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(plan_date), MAX(plan_date) FROM care_plans")
    min_date, max_date = cursor.fetchone()

    cursor.execute("SELECT AVG(cost), MIN(cost), MAX(cost) FROM care_plans")
    avg_cost, min_cost, max_cost = cursor.fetchone()

    cursor.execute("SELECT COUNT(*) FROM risk_assessments")
    num_assessments = cursor.fetchone()[0]

    # What the as-of join will find: an assessment dated at or before the
    # admission and no more than 30 days (the config's grace) before it.
    cursor.execute("""
        SELECT COUNT(DISTINCT p.patient_id)
        FROM patients p
        JOIN risk_assessments ra ON p.patient_id = ra.patient_id
        WHERE ra.assessed_on <= p.admission_date
          AND ra.assessed_on >= p.admission_date - interval 'P30D'
    """)
    matched = cursor.fetchone()[0]

    conn.close()

    print("✓ Data loaded successfully!")
    print("\nStatistics:")
    print(f"  Patients: {num_patients}")
    print(f"  Care plans: {num_plans}")
    print(f"  As-of dates: {num_dates}")
    print(f"  Plan date range: {min_date} to {max_date}")
    print(f"  Plan costs: ${min_cost:.2f} - ${max_cost:.2f} (avg: ${avg_cost:.2f})")
    print(f"  Risk assessments: {num_assessments}")
    print(f"  Patients with an assessment in the 30 days before admission: {matched}")
    print(f"\nSchema: {SCHEMA}")


if __name__ == "__main__":
    create_database()
