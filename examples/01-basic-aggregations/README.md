# Example 1: Basic Aggregations

This example demonstrates basic feature aggregation across a parent-child relationship.

## Scenario: E-Commerce

**Entities:**
- **Customers** (parent) - User accounts
- **Orders** (child) - Purchase transactions

**Goal:** Generate features for each customer based on their order history.

## Data Schema

```
customers
├── customer_id (PK)
├── signup_date
├── country
└── age

orders
├── order_id (PK)
├── customer_id (FK)
├── order_date
├── amount
└── status
```

## Generated Features

For each customer, we aggregate:
- Order count (total orders)
- Mean order amount
- Sum of all orders (lifetime value)
- Standard deviation of order amounts
- Time-windowed aggregations (last 7 days, 30 days)

Plus transformations:
- Day of week from signup_date
- Month from signup_date
- Absolute value of order metrics
- Cumulative sums

## Files

- `config.yaml` - Featurizer configuration
- `create_data.py` - Generates sample SQLite database
- `run_example.py` - Runs feature generation
- `data.db` - SQLite database (generated)

## Usage

```bash
# 1. Generate sample data
python create_data.py

# 2. Run feature generation
python run_example.py

# 3. View generated SQL
python run_example.py --show-sql

# 4. Execute and save results
python run_example.py --execute --output features.csv
```

## What You'll Learn

- Basic entity-relationship setup
- Parent-child aggregations
- Temporal indexes for time-windowed features
- How to inspect generated SQL
- Feature naming conventions
