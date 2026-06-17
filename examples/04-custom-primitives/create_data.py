#!/usr/bin/env python
"""Generate sample financial data for Example 4 (Custom Primitives).

Loads into PostgreSQL. Run via ``just example 04`` (which starts the throwaway
database first), or set DATABASE_URL / PG* and run directly. See ``examples/_db.py``.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/ for _db
import _db

# Configuration
SCHEMA = "example_04"
NUM_ACCOUNTS = 75
MIN_TRANSACTIONS_PER_ACCOUNT = 0
MAX_TRANSACTIONS_PER_ACCOUNT = 100

ACCOUNT_TYPES = ["checking", "savings", "credit", "investment"]
TRANSACTION_TYPES = ["deposit", "withdrawal", "transfer", "payment", "fee"]

# Seed for reproducibility
random.seed(42)


def create_database():
    """Load financial data into the ``example_04`` schema on PostgreSQL."""
    conn = _db.connect(SCHEMA)
    cursor = conn.cursor()

    # Create tables (bare names resolve via the search_path set by _db.connect)
    cursor.execute("""
        CREATE TABLE accounts (
            account_id INTEGER PRIMARY KEY,
            open_date DATE NOT NULL,
            account_type TEXT NOT NULL,
            credit_limit DOUBLE PRECISION NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE transactions (
            transaction_id INTEGER PRIMARY KEY,
            account_id INTEGER NOT NULL,
            transaction_date DATE NOT NULL,
            amount DOUBLE PRECISION NOT NULL,
            transaction_type TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE as_of_dates (
            as_of_date DATE PRIMARY KEY
        )
    """)

    # Generate accounts
    base_date = datetime(2023, 1, 1)
    accounts = []

    for i in range(1, NUM_ACCOUNTS + 1):
        open_date = base_date + timedelta(days=random.randint(0, 365))
        account_type = random.choice(ACCOUNT_TYPES)

        # Credit limit varies by account type
        if account_type == "credit":
            credit_limit = round(random.uniform(1000, 50000), 2)
        elif account_type == "investment":
            credit_limit = round(random.uniform(10000, 500000), 2)
        else:
            credit_limit = round(random.uniform(500, 10000), 2)

        accounts.append((i, open_date.date(), account_type, credit_limit))

    cursor.executemany("INSERT INTO accounts VALUES (%s, %s, %s, %s)", accounts)

    # Generate transactions
    transactions = []
    transaction_id = 1

    for account_id in range(1, NUM_ACCOUNTS + 1):
        account_open = accounts[account_id - 1][1]
        num_transactions = random.randint(
            MIN_TRANSACTIONS_PER_ACCOUNT, MAX_TRANSACTIONS_PER_ACCOUNT
        )

        for _ in range(num_transactions):
            # Transactions happen after account opening
            days_after_open = random.randint(1, 400)
            transaction_date = datetime.strptime(
                str(account_open), "%Y-%m-%d"
            ) + timedelta(days=days_after_open)

            # Amount distribution varies by transaction type
            tx_type = random.choice(TRANSACTION_TYPES)
            if tx_type == "fee":
                amount = round(random.uniform(1.0, 50.0), 2)
            elif tx_type == "withdrawal":
                amount = -round(random.uniform(10.0, 2000.0), 2)
            elif tx_type == "deposit":
                amount = round(random.uniform(50.0, 10000.0), 2)
            else:
                amount = round(random.uniform(-1000.0, 1000.0), 2)

            transactions.append(
                (transaction_id, account_id, transaction_date.date(), amount, tx_type)
            )
            transaction_id += 1

    cursor.executemany(
        "INSERT INTO transactions VALUES (%s, %s, %s, %s, %s)", transactions
    )

    # Generate as_of_dates (monthly for 2024)
    as_of_dates = []
    for month in range(1, 13):
        date = datetime(2024, month, 1).date()
        as_of_dates.append((date,))

    cursor.executemany("INSERT INTO as_of_dates VALUES (%s)", as_of_dates)

    conn.commit()

    # Print statistics
    cursor.execute("SELECT COUNT(*) FROM accounts")
    num_accounts = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    num_transactions = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM as_of_dates")
    num_dates = cursor.fetchone()[0]

    cursor.execute(
        "SELECT MIN(transaction_date), MAX(transaction_date) FROM transactions"
    )
    min_date, max_date = cursor.fetchone()

    cursor.execute("SELECT AVG(amount), MIN(amount), MAX(amount) FROM transactions")
    avg_amount, min_amount, max_amount = cursor.fetchone()

    # Median via PostgreSQL's ordered-set aggregate (the same primitive the
    # custom Median aggregation emits).
    cursor.execute(
        "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY amount) FROM transactions"
    )
    median_amount = cursor.fetchone()[0]

    conn.close()

    print("✓ Data loaded successfully!")
    print("\nStatistics:")
    print(f"  Accounts: {num_accounts}")
    print(f"  Transactions: {num_transactions}")
    print(f"  As-of dates: {num_dates}")
    print(f"  Transaction date range: {min_date} to {max_date}")
    print("  Transaction amounts:")
    print(f"    Mean: ${avg_amount:.2f}")
    print(f"    Median: ${median_amount:.2f}")
    print(f"    Range: ${min_amount:.2f} to ${max_amount:.2f}")
    print(f"\nSchema: {SCHEMA}")


if __name__ == "__main__":
    create_database()
