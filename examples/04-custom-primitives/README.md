# Example 4: Custom Primitives

This example demonstrates how to extend Featurizer by registering custom aggregation and transformation primitives.

## Scenario: Financial Analytics

**Entities:**
- **Accounts** (target) - Bank accounts
- **Transactions** (child) - Financial transactions

**Goal:** Create custom financial metrics using domain-specific aggregations and transformations.

## Custom Primitives

### Custom Aggregations
- **Median** - Calculate median transaction amount
- **Percentile95** - Calculate 95th percentile of transaction amounts
- **Range** - Calculate range (max - min) of transaction amounts

### Custom Transformations
- **Log** - Natural logarithm transformation
- **ZScore** - Standardize features to z-scores
- **BinCount** - Discretize continuous values into bins

## Data Schema

```
accounts
├── account_id (PK)
├── open_date
├── account_type
└── credit_limit

transactions
├── transaction_id (PK)
├── account_id (FK)
├── transaction_date
├── amount
└── transaction_type
```

## Implementation

The example shows:
1. How to create custom primitive classes
2. How to register them with the feature system
3. How to use them in feature generation
4. How custom SQL expressions are rendered

## Files

- `config.yaml` - Featurizer configuration
- `custom_primitives.py` - Custom aggregation and transformation implementations
- `create_data.py` - Generates sample SQLite database
- `run_example.py` - Runs feature generation with custom primitives
- `data.db` - SQLite database (generated)

## Usage

```bash
# 1. Generate sample data
python create_data.py

# 2. Run with custom primitives
python run_example.py

# 3. View SQL with custom expressions
python run_example.py --show-sql

# 4. Execute and save results
python run_example.py --execute --output custom_features.csv
```

## What You'll Learn

- How to subclass Aggregation and Transformation
- How to implement to_sql() methods
- How to register custom primitives
- How to verify primitive registration
- Best practices for custom primitive design
