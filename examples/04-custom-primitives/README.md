# Example 4: Custom Primitives

> **Note (2026-03-17):** Since the primitives expansion in commit `6b25339`, several primitives demonstrated here — Median, Percentile95, Range, Log, and ZScore — are now available as built-in primitives (`median`, `p95`, `range`, `ln`, `cross_entity_zscore`). This example is retained to demonstrate the **registration pattern** for creating truly custom primitives. To use the built-in versions, simply reference them by name in your configuration — no custom code needed.

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

- `config.yaml` - Featurizer configuration (selects the custom primitives)
- `custom_primitives.py` - Custom aggregation and transformation implementations
- `create_data.py` - Loads financial data into PostgreSQL (schema `example_04`)
- `run_example.py` - Runs feature generation with custom primitives

## Using Built-in Equivalents

```yaml
# These primitives are now built-in — no custom code required:
# - median     (was: Median)
# - p95        (was: Percentile95)
# - range      (was: Range)
# - ln         (was: Log)
# - cross_entity_zscore  (was: ZScore)
#
# Just request them via get_aggregations() or get_transformers():
from featurizer.primitives.utils import get_aggregations
aggs = get_aggregations(["median", "p95", "range"])
```

## Usage

```bash
# From the repo root: start the throwaway PostgreSQL, then run end to end
just db-up
just example 04            # loads data + executes (registers custom primitives)

# Or step by step (DATABASE_URL / PG* must point at a PostgreSQL):
python create_data.py                    # load schema example_04
python run_example.py                    # feature summary
python run_example.py --show-sql         # inspect SQL (no database needed)
python run_example.py --execute --output custom_features.csv
python run_example.py --list-primitives  # show registered primitives
```

## What You'll Learn

- How to subclass Aggregation and Transformation
- How to implement to_sql() methods
- How to register custom primitives
- How to verify primitive registration
- Best practices for custom primitive design
