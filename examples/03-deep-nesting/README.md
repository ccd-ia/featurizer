# Example 3: Deep Nesting

This example demonstrates feature generation across multiple levels of entity relationships (depth=3).

## Scenario: Retail Supply Chain

**Entities:**
- **Stores** (target, depth 0) - Retail locations
- **Orders** (depth 1) - Orders placed by stores
- **Products** (depth 2) - Products within orders (via order_items)
- **Suppliers** (depth 3) - Suppliers of products

**Goal:** Generate features for stores based on their entire supply chain, aggregating across multiple levels.

## Data Schema

```
stores
├── store_id (PK)
├── open_date
├── region
└── size_sqft

orders
├── order_id (PK)
├── store_id (FK)
├── order_date
└── status

order_items
├── item_id (PK)
├── order_id (FK)
├── product_id (FK)
├── quantity
└── unit_price

products
├── product_id (PK)
├── supplier_id (FK)
├── category
└── base_cost

suppliers
├── supplier_id (PK)
├── country
└── rating
```

## Relationship Chain

```
Stores (depth 0)
  → Orders (depth 1)
    → OrderItems (depth 2)
      → Products (depth 2)
        → Suppliers (depth 3)
```

## Generated Features

Features aggregate across multiple levels:
- **Depth 1**: Order count, mean order value, cancelled orders
- **Depth 2**: Total product variety, mean quantity per order, product categories
- **Depth 3**: Supplier diversity, mean supplier rating, international supplier count

## Files

- `config.yaml` - Featurizer configuration with max_depth=3
- `create_data.py` - Generates sample SQLite database with multi-level relationships
- `run_example.py` - Runs feature generation with deep nesting
- `data.db` - SQLite database (generated)

## Usage

```bash
# 1. Generate sample data
python create_data.py

# 2. Run feature generation
python run_example.py

# 3. View generated SQL with deep joins
python run_example.py --show-sql

# 4. Execute and save results
python run_example.py --execute --output deep_features.csv
```

## What You'll Learn

- Multi-level relationship configuration
- How max_depth controls feature generation
- Deep aggregation chains (e.g., stores → orders → products → suppliers)
- CTE structure for nested relationships
- Performance considerations for deep hierarchies
