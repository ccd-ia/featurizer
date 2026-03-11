#!/usr/bin/env python
"""Generate sample healthcare data for Example 2 (Temporal Joins)."""

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
NUM_PATIENTS = 50
MIN_PLANS_PER_PATIENT = 0
MAX_PLANS_PER_PATIENT = 10

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
TREATMENT_TYPES = ["medication", "therapy", "surgery", "monitoring", "rehabilitation"]

# Seed for reproducibility
random.seed(42)


def create_database():
    """Create SQLite database with temporal healthcare data."""
    db_path = Path(__file__).parent / "data.db"

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
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
            cost REAL NOT NULL,
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

    cursor.executemany(
        "INSERT INTO patients VALUES (?, ?, ?, ?)",
        patients
    )

    # Generate care plans (temporal events)
    plans = []
    plan_id = 1

    for patient_id in range(1, NUM_PATIENTS + 1):
        patient_admission = patients[patient_id - 1][1]
        num_plans = random.randint(MIN_PLANS_PER_PATIENT, MAX_PLANS_PER_PATIENT)

        for plan_num in range(num_plans):
            # Plans occur after admission, spread over time
            days_after_admission = random.randint(1, 500)
            plan_date = datetime.strptime(str(patient_admission), "%Y-%m-%d") + timedelta(days=days_after_admission)

            treatment = random.choice(TREATMENT_TYPES)
            cost = round(random.uniform(500.0, 50000.0), 2)

            plans.append((plan_id, patient_id, plan_date.date(), treatment, cost))
            plan_id += 1

    cursor.executemany(
        "INSERT INTO care_plans VALUES (?, ?, ?, ?, ?)",
        plans
    )

    # Generate as_of_dates (quarterly snapshots for 2023-2024)
    as_of_dates = []
    for year in [2023, 2024]:
        for month in [1, 4, 7, 10]:  # Q1, Q2, Q3, Q4
            date = datetime(year, month, 1).date()
            as_of_dates.append((date,))

    cursor.executemany(
        "INSERT INTO as_of_dates VALUES (?)",
        as_of_dates
    )

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

    # Example of temporal join behavior
    cursor.execute("""
        SELECT
            COUNT(DISTINCT p.patient_id) as patients_with_active_plans
        FROM patients p
        JOIN care_plans cp ON p.patient_id = cp.patient_id
        WHERE cp.plan_date <= '2024-01-01'
    """)
    active_plans = cursor.fetchone()[0]

    conn.close()

    print("✓ Database created successfully!")
    print(f"\nStatistics:")
    print(f"  Patients: {num_patients}")
    print(f"  Care plans: {num_plans}")
    print(f"  As-of dates: {num_dates}")
    print(f"  Plan date range: {min_date} to {max_date}")
    print(f"  Plan costs: ${min_cost:.2f} - ${max_cost:.2f} (avg: ${avg_cost:.2f})")
    print(f"  Patients with plans by 2024-01-01: {active_plans}")
    print(f"\nDatabase: {db_path}")


if __name__ == "__main__":
    create_database()
