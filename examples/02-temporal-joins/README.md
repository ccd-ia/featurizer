# Example 2: Temporal Joins

This example demonstrates temporal join functionality using as-of semantics with grace periods.

## Scenario: Healthcare

**Entities:**
- **Patients** (target) - Patient records
- **Care Plans** (related) - Treatment plans that change over time

**Goal:** For each patient at specific points in time, join the most recent care plan that was active before that time.

## Data Schema

```
patients
├── patient_id (PK)
├── admission_date
├── age
└── severity_level

care_plans
├── plan_id (PK)
├── patient_id (FK)
├── plan_date
├── treatment_type
└── cost
```

## Temporal Join Behavior

The relationship uses `temporal_mode: as_of` with a grace period:
- For each patient at a given as_of_date, find the most recent care_plan where `plan_date <= as_of_date`
- Grace period allows matching plans up to 7 days in the future
- Only one care plan per patient is selected (the latest applicable one)

## Generated Features

With as-of joins, features capture the state of care plans at specific moments:
- Treatment type at each snapshot
- Cost of active plan
- Time-windowed aggregations (plans in last 30/90 days)
- Direct attributes from the most recent plan

## Files

- `config.yaml` - Featurizer configuration with temporal relationship
- `create_data.py` - Loads temporal data into PostgreSQL (schema `example_02`)
- `run_example.py` - Runs feature generation with temporal joins

## Usage

```bash
# From the repo root: start the throwaway PostgreSQL, then run end to end
just db-up
just example 02            # loads data + executes

# Or step by step (DATABASE_URL / PG* must point at a PostgreSQL):
python create_data.py                    # load schema example_02
python run_example.py                    # feature summary
python run_example.py --show-sql         # inspect SQL (no database needed)
python run_example.py --execute --output temporal_features.csv
```

## What You'll Learn

- Temporal relationship configuration (mode: as_of)
- Grace period usage for fuzzy temporal matching
- As-of join SQL generation (LATERAL clauses)
- How temporal_ix drives join logic
- Point-in-time feature generation
