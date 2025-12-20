# Featurizer Examples

This directory contains practical, self-contained examples demonstrating key features of the Featurizer library.

## Overview

Each example includes:
- **README.md** - Detailed explanation of the scenario and concepts
- **config.yaml** - Featurizer configuration file
- **create_data.py** - Script to generate sample SQLite database
- **run_example.py** - Script to run feature generation
- **data.db** - Generated SQLite database (created by create_data.py)

All examples use SQLite databases for simplicity and require no external dependencies beyond Python and Featurizer.

## Examples

### [01-basic-aggregations](./01-basic-aggregations/)
**Difficulty:** Beginner
**Scenario:** E-commerce (Customers → Orders)

Demonstrates:
- Basic parent-child relationships
- Aggregations (count, mean, sum, stddev)
- Time-windowed features (P7D, P30D)
- Feature naming conventions

**Start here if you're new to Featurizer.**

### [02-temporal-joins](./02-temporal-joins/)
**Difficulty:** Intermediate
**Scenario:** Healthcare (Patients → Care Plans)

Demonstrates:
- As-of join semantics
- Temporal relationships with grace periods
- Point-in-time feature generation
- LATERAL join SQL generation

**Learn about temporal feature engineering.**

### [03-deep-nesting](./03-deep-nesting/)
**Difficulty:** Intermediate
**Scenario:** Retail Supply Chain (Stores → Orders → Products → Suppliers)

Demonstrates:
- Multi-level relationships (depth=3)
- Feature aggregation across chains
- Complex entity graphs
- CTE structure for nested relationships

**Understand how features propagate through multiple levels.**

### [04-custom-primitives](./04-custom-primitives/)
**Difficulty:** Advanced
**Scenario:** Financial Analytics (Accounts → Transactions)

Demonstrates:
- Creating custom aggregation primitives
- Creating custom transformation primitives
- Registering primitives with the system
- Custom SQL expression generation

**Learn how to extend Featurizer with domain-specific primitives.**

## Quick Start

Each example follows the same workflow:

```bash
# 1. Navigate to an example directory
cd 01-basic-aggregations/

# 2. Generate sample data
python create_data.py

# 3. Run feature generation (shows summary)
python run_example.py

# 4. View generated SQL
python run_example.py --show-sql

# 5. Execute query and save results
python run_example.py --execute --output features.csv
```

## Learning Path

**Recommended order:**

1. **Start with Example 1** to understand basic concepts
2. **Try Example 2** to learn temporal joins
3. **Explore Example 3** to see deep nesting in action
4. **Study Example 4** when you need custom primitives

## Common Patterns

### Configuration Structure

All examples follow this YAML structure:

```yaml
target: entity_alias       # Target entity for features
max_depth: 2               # Maximum relationship depth

intervals:                 # Time windows for aggregations
  - P7D
  - P30D

entities:                  # Entity definitions
  - alias: entity_name
    id: primary_key
    table: table_name
    temporal_ix: timestamp_column  # Optional
    variables:
      column_name:
        type: numeric|categorical

relationships:             # Parent-child relationships
  - parent:
      entity: parent_alias
      key: parent_key
    child:
      entity: child_alias
      key: child_key
    temporal:              # Optional
      mode: as_of
      grace: P7D
```

### Database Requirements

All examples create an `as_of_dates` table required by Featurizer:

```sql
CREATE TABLE as_of_dates (
    as_of_date DATE PRIMARY KEY
);
```

This table defines the time points at which features are calculated.

## Example Outputs

### Feature Counts

Typical feature counts by example:

- **Example 1:** ~40-60 features (basic aggregations)
- **Example 2:** ~30-50 features (temporal joins)
- **Example 3:** ~100-150 features (deep nesting creates many combinations)
- **Example 4:** ~40-60 features (+ custom primitives)

### Generated SQL

All examples generate SQL with:
- Multiple CTEs for feature synthesis and transformation
- Proper temporal filtering
- Window functions for time-based features
- Joins following the relationship graph

Use `--show-sql` to inspect the generated queries and learn SQL patterns.

## Troubleshooting

### "Database not found" error
Run `python create_data.py` first to generate the SQLite database.

### "Module not found" error
Make sure you're running from within the example directory, or that featurizer is installed:
```bash
pip install -e /path/to/featurizer
```

### SQLite vs PostgreSQL differences
These examples use SQLite for simplicity. Some features work differently in PostgreSQL:
- Median/Percentile functions (Example 4) use different syntax
- Date/time functions may differ
- Some advanced aggregations require PostgreSQL-specific functions

To use with PostgreSQL, update the `run_example.py` DATABASE_URL and adjust SQL generation if needed.

## Next Steps

After completing these examples:

1. **Read the main README** in the repository root for architecture details
2. **Review AGENTS.md** for understanding the planner/renderer/executor pipeline
3. **Explore featurizer/primitives/** to see available aggregations and transformations
4. **Try your own datasets** by creating custom config.yaml files

## Contributing Examples

Have an interesting use case? Contributions are welcome!

New examples should:
- Follow the same directory structure
- Include comprehensive README.md
- Use SQLite for portability
- Include sample data generation
- Add error handling in run_example.py
- Document key learning objectives
