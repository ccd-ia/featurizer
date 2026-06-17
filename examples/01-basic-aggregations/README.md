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
- `create_data.py` - Loads sample data into PostgreSQL (schema `example_01`)
- `run_example.py` - Runs feature generation

## Usage

```bash
# From the repo root: start the throwaway PostgreSQL, then run end to end
just db-up
just example 01            # loads data + executes

# Or step by step (DATABASE_URL / PG* must point at a PostgreSQL):
python create_data.py                    # load schema example_01
python run_example.py                    # feature summary
python run_example.py --show-sql         # inspect SQL (no database needed)
python run_example.py --execute --output features.csv
```

## What You'll Learn

- Basic entity-relationship setup
- Parent-child aggregations
- Temporal indexes for time-windowed features
- How to inspect generated SQL
- Feature naming conventions
