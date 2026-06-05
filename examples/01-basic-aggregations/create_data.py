#!/usr/bin/env python
"""Generate sample e-commerce data for Example 1."""

import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
NUM_CUSTOMERS = 100
MIN_ORDERS_PER_CUSTOMER = 0
MAX_ORDERS_PER_CUSTOMER = 20

COUNTRIES = ["US", "UK", "CA", "DE", "FR", "AU"]
STATUSES = ["completed", "pending", "cancelled"]

# Seed for reproducibility
random.seed(42)


def create_database():
    """Create SQLite database with sample data."""
    db_path = Path(__file__).parent / "data.db"

    # Remove existing database
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE customers (
            customer_id INTEGER PRIMARY KEY,
            signup_date DATE NOT NULL,
            country TEXT NOT NULL,
            age INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        )
    """)

    # Create as_of_dates table (required by Featurizer)
    cursor.execute("""
        CREATE TABLE as_of_dates (
            as_of_date DATE PRIMARY KEY
        )
    """)

    # Generate customers
    base_date = datetime(2023, 1, 1)
    customers = []

    for i in range(1, NUM_CUSTOMERS + 1):
        signup_date = base_date + timedelta(days=random.randint(0, 730))  # 2 years
        country = random.choice(COUNTRIES)
        age = random.randint(18, 75)

        customers.append((i, signup_date.date(), country, age))

    cursor.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)

    # Generate orders
    orders = []
    order_id = 1

    for customer_id in range(1, NUM_CUSTOMERS + 1):
        customer_signup = customers[customer_id - 1][1]
        num_orders = random.randint(MIN_ORDERS_PER_CUSTOMER, MAX_ORDERS_PER_CUSTOMER)

        for _ in range(num_orders):
            # Orders happen after signup
            days_after_signup = random.randint(1, 365)
            order_date = datetime.strptime(
                str(customer_signup), "%Y-%m-%d"
            ) + timedelta(days=days_after_signup)

            amount = round(random.uniform(10.0, 500.0), 2)
            status = random.choice(STATUSES)

            orders.append((order_id, customer_id, order_date.date(), amount, status))
            order_id += 1

    cursor.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?)", orders)

    # Generate as_of_dates (monthly snapshots for 2024)
    as_of_dates = []
    for month in range(1, 13):
        date = datetime(2024, month, 1).date()
        as_of_dates.append((date,))

    cursor.executemany("INSERT INTO as_of_dates VALUES (?)", as_of_dates)

    conn.commit()

    # Print statistics
    cursor.execute("SELECT COUNT(*) FROM customers")
    num_customers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM orders")
    num_orders = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM as_of_dates")
    num_dates = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(order_date), MAX(order_date) FROM orders")
    min_date, max_date = cursor.fetchone()

    cursor.execute("SELECT AVG(amount), MIN(amount), MAX(amount) FROM orders")
    avg_amount, min_amount, max_amount = cursor.fetchone()

    conn.close()

    print("✓ Database created successfully!")
    print("\nStatistics:")
    print(f"  Customers: {num_customers}")
    print(f"  Orders: {num_orders}")
    print(f"  As-of dates: {num_dates}")
    print(f"  Date range: {min_date} to {max_date}")
    print(
        f"  Order amounts: ${min_amount:.2f} - ${max_amount:.2f} (avg: ${avg_amount:.2f})"
    )
    print(f"\nDatabase: {db_path}")


if __name__ == "__main__":
    create_database()
