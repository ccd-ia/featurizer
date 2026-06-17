#!/usr/bin/env python
"""Generate sample retail supply chain data for Example 3 (Deep Nesting).

Loads into PostgreSQL. Run via ``just example 03`` (which starts the throwaway
database first), or set DATABASE_URL / PG* and run directly. See ``examples/_db.py``.
"""

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # examples/ for _db
import _db

# Configuration
SCHEMA = "example_03"
NUM_STORES = 20
NUM_SUPPLIERS = 15
NUM_PRODUCTS = 50

MIN_ORDERS_PER_STORE = 5
MAX_ORDERS_PER_STORE = 30
MIN_ITEMS_PER_ORDER = 1
MAX_ITEMS_PER_ORDER = 10

REGIONS = ["North", "South", "East", "West", "Central"]
ORDER_STATUSES = ["completed", "pending", "cancelled", "shipped"]
CATEGORIES = ["Electronics", "Clothing", "Food", "Home", "Sports", "Books"]
COUNTRIES = ["US", "China", "Germany", "Japan", "India", "Brazil"]

# Seed for reproducibility
random.seed(42)


def create_database():
    """Load multi-level supply chain data into the ``example_03`` schema."""
    conn = _db.connect(SCHEMA)
    cursor = conn.cursor()

    # Create tables in FK-dependency order (PostgreSQL resolves REFERENCES at
    # creation time, so a referenced table must already exist): suppliers ->
    # products -> stores -> orders -> order_items.
    cursor.execute("""
        CREATE TABLE suppliers (
            supplier_id INTEGER PRIMARY KEY,
            country TEXT NOT NULL,
            rating DOUBLE PRECISION NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE products (
            product_id INTEGER PRIMARY KEY,
            supplier_id INTEGER NOT NULL,
            category TEXT NOT NULL,
            base_cost DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (supplier_id) REFERENCES suppliers(supplier_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE stores (
            store_id INTEGER PRIMARY KEY,
            open_date DATE NOT NULL,
            region TEXT NOT NULL,
            size_sqft INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE orders (
            order_id INTEGER PRIMARY KEY,
            store_id INTEGER NOT NULL,
            order_date DATE NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (store_id) REFERENCES stores(store_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE order_items (
            item_id INTEGER PRIMARY KEY,
            order_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price DOUBLE PRECISION NOT NULL,
            FOREIGN KEY (order_id) REFERENCES orders(order_id),
            FOREIGN KEY (product_id) REFERENCES products(product_id)
        )
    """)

    cursor.execute("""
        CREATE TABLE as_of_dates (
            as_of_date DATE PRIMARY KEY
        )
    """)

    # Generate suppliers (depth 3)
    suppliers = []
    for i in range(1, NUM_SUPPLIERS + 1):
        country = random.choice(COUNTRIES)
        rating = round(random.uniform(1.0, 5.0), 1)
        suppliers.append((i, country, rating))

    cursor.executemany("INSERT INTO suppliers VALUES (%s, %s, %s)", suppliers)

    # Generate products (depth 2)
    products = []
    for i in range(1, NUM_PRODUCTS + 1):
        supplier_id = random.randint(1, NUM_SUPPLIERS)
        category = random.choice(CATEGORIES)
        base_cost = round(random.uniform(5.0, 500.0), 2)
        products.append((i, supplier_id, category, base_cost))

    cursor.executemany("INSERT INTO products VALUES (%s, %s, %s, %s)", products)

    # Generate stores (depth 0 - target)
    base_date = datetime(2022, 1, 1)
    stores = []
    for i in range(1, NUM_STORES + 1):
        open_date = base_date + timedelta(days=random.randint(0, 730))
        region = random.choice(REGIONS)
        size_sqft = random.randint(1000, 50000)
        stores.append((i, open_date.date(), region, size_sqft))

    cursor.executemany("INSERT INTO stores VALUES (%s, %s, %s, %s)", stores)

    # Generate orders (depth 1)
    orders = []
    order_id = 1
    for store_id in range(1, NUM_STORES + 1):
        store_open = stores[store_id - 1][1]
        num_orders = random.randint(MIN_ORDERS_PER_STORE, MAX_ORDERS_PER_STORE)

        for _ in range(num_orders):
            days_after_open = random.randint(1, 500)
            order_date = datetime.strptime(str(store_open), "%Y-%m-%d") + timedelta(
                days=days_after_open
            )
            status = random.choice(ORDER_STATUSES)
            orders.append((order_id, store_id, order_date.date(), status))
            order_id += 1

    cursor.executemany("INSERT INTO orders VALUES (%s, %s, %s, %s)", orders)

    # Generate order_items (depth 2)
    items = []
    item_id = 1
    for order_id_val in range(1, order_id):
        num_items = random.randint(MIN_ITEMS_PER_ORDER, MAX_ITEMS_PER_ORDER)
        for _ in range(num_items):
            product_id = random.randint(1, NUM_PRODUCTS)
            quantity = random.randint(1, 20)
            unit_price = round(random.uniform(10.0, 1000.0), 2)
            items.append((item_id, order_id_val, product_id, quantity, unit_price))
            item_id += 1

    cursor.executemany("INSERT INTO order_items VALUES (%s, %s, %s, %s, %s)", items)

    # Generate as_of_dates (quarterly for 2023-2024)
    as_of_dates = []
    for year in [2023, 2024]:
        for month in [1, 4, 7, 10]:
            date = datetime(year, month, 1).date()
            as_of_dates.append((date,))

    cursor.executemany("INSERT INTO as_of_dates VALUES (%s)", as_of_dates)

    conn.commit()

    # Print statistics
    cursor.execute("SELECT COUNT(*) FROM stores")
    num_stores = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM orders")
    num_orders = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM order_items")
    num_items = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM products")
    num_products = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM suppliers")
    num_suppliers = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM as_of_dates")
    num_dates = cursor.fetchone()[0]

    # Calculate some aggregates
    cursor.execute("SELECT AVG(rating) FROM suppliers")
    avg_rating = cursor.fetchone()[0]

    cursor.execute("""
        SELECT COUNT(DISTINCT s.supplier_id)
        FROM stores st
        JOIN orders o ON st.store_id = o.store_id
        JOIN order_items oi ON o.order_id = oi.order_id
        JOIN products p ON oi.product_id = p.product_id
        JOIN suppliers s ON p.supplier_id = s.supplier_id
    """)
    connected_suppliers = cursor.fetchone()[0]

    conn.close()

    print("✓ Data loaded successfully!")
    print("\nStatistics:")
    print(f"  Stores (depth 0): {num_stores}")
    print(f"  Orders (depth 1): {num_orders}")
    print(f"  Order items (depth 2): {num_items}")
    print(f"  Products (depth 2): {num_products}")
    print(f"  Suppliers (depth 3): {num_suppliers}")
    print(f"  As-of dates: {num_dates}")
    print(f"  Average supplier rating: {avg_rating:.1f}")
    print(f"  Suppliers with orders: {connected_suppliers}")
    print(f"\nSchema: {SCHEMA}")


if __name__ == "__main__":
    create_database()
