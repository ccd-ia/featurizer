# Example 2: Temporal Joins

This example puts featurizer's two relationship directions side by side: an
aggregation, and an as-of (point-in-time) lookup with a grace period. Which one
a relationship is depends on which side you declare as `parent`.

## Scenario: Healthcare

**Entities:**
- **Patients** (target) - Patient records, one row per admission
- **Care Plans** (aggregated) - Treatment plans, many per patient
- **Risk Assessments** (as-of lookup) - Scores recorded over time, before and after admission

**Goal:** For each patient, summarize their care plans up to each as-of date,
and attach the most recent risk assessment known at admission.

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

risk_assessments
├── assessment_id (PK)
├── patient_id (FK)
├── assessed_on
└── risk_score
```

## Two Relationships, Two Directions

| Relationship | `parent` | `child` | What it renders |
|---|---|---|---|
| Aggregation | `patients` (target) | `care_plans` | Care plans rolled up per patient, bounded by `as_of_date` |
| As-of lookup | `risk_assessments` | `patients` | One `left join lateral … limit 1` per patient row |

**The as-of lookup declares the lookup table as `parent`.** Each patient row
pulls the most recent assessment where:

- `assessed_on <= admission_date` - nothing dated after the admission is read
- `assessed_on >= admission_date - 30 days` - `grace: P30D` is a lookback cap, so
  an older assessment is ignored and the columns are `NULL`
- only the latest matching row is kept (`order by assessed_on desc limit 1`)

The sample data has assessments on both sides of the admission date so both
bounds do visible work: 14 of the 50 patients have an assessment inside the
window, and the other 36 get `NULL`.

**A `temporal:` block on the aggregation relationship would do nothing.** The
planner reads the block only on the as-of direction. `featurizer validate`
warns when it finds one on an aggregation, and names the orientation that
works. The aggregation needs no block: its reads of `care_plans` are already
bounded by `as_of_date`.

## Generated Features

- Time-windowed aggregations over care plans (lifetime, last 30 days, last 90 days)
- `risk_score` from the as-of lookup, and its transformations
- Direct patient attributes (`age`, `severity_level`)

## Files

- `config.yaml` - Featurizer configuration with temporal relationship
- `create_data.py` - Loads the three tables into PostgreSQL (schema `example_02`)
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

- Which side is `parent` for an aggregation, and which for an as-of lookup
- Temporal relationship configuration (`mode: as_of`)
- `grace` as a lookback cap
- As-of join SQL generation (LATERAL clauses)
- How `temporal_ix` drives the join
