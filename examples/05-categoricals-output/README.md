# Example 5: Direct Categoricals, Output & Imputation

This example demonstrates the **consumer-facing** end of Featurizer: turning a
direct categorical attribute into model-ready columns, recovering readable
names, executing against PostgreSQL, and the opt-in imputation contract.

Unlike examples 1–4 (which only inspect config/features/SQL, no database), this
tutorial **executes** against PostgreSQL so you see the actual feature matrix.

## Scenario: Food Inspections

**Entities:**
- **Facilities** (target) — `name` (identifier), `facility_type` (categorical),
  `first_seen`
- **Inspections** (child events) — `score` per inspection

## What it shows

1. **Variable roles** — `facility_type` (`role: categorical`) is one-hot
   encoded; `name` (`role: identifier`) is excluded from the output, loudly.
2. **Fixed-vocabulary, fit-free one-hot** — each value of the declared
   vocabulary becomes a `facilities.facility_type=<value>` 0/1 column.
   Featurizer is split-blind: it never learns the vocabulary from the data.
   The data deliberately includes an **out-of-vocabulary** value and a **NULL** —
   both produce an all-zero one-hot row (never a crash).
3. **Feature manifest** — `feature_manifest` / `manifest_dataframe()` map each
   output column to its full, untruncated label (and one-hot `source_column` /
   `value`). Since v0.5.0 entries also carry lineage (`depth`, `parents`,
   `source_alias`, `interval`) and a generated human `description`, and
   `to_tables(schema)` persists the manifest as `"<schema>"."<stem>_manifest"`
   beside the feature-group tables.
4. **Output formats** — `to_dataframe()` (pandas) and `to_arrow()` (Arrow,
   NULL-faithful, keys as columns).
5. **Imputation** — `impute=True` fills count-like aggregates (`COUNT`) with 0,
   leaves measures (`MEAN`) NULL, and emits a `<feature>__missing` flag.

## Files

- `config.yaml` — Featurizer configuration (roles + a declared vocabulary)
- `create_data.py` — loads sample data into PostgreSQL (schema `example_05`)
- `run_example.py` — runs feature generation (`--show-sql`, `--execute`)
- `tutorial.ipynb` — the full walk-through (**requires PostgreSQL**)

## Running it

This example needs a PostgreSQL (it executes the matrix):

```bash
just db-up         # throwaway postgres:16 (exports DATABASE_URL)
just example 05    # seed + run end to end
just db-down
```

Or with your own database (`DATABASE_URL` / `PG*` set):

```bash
python create_data.py
python run_example.py --execute
python run_example.py --show-sql   # SQL only; still needs the config, not the DB
```

For the notebook, start a database first (`just db-up`) and set `DATABASE_URL`,
then run `tutorial.ipynb`.

## Vocabulary source: declared vs ENUM

This example **declares** the vocabulary in `config.yaml`, which keeps
`Featurizer(...)` database-free. The alternative is to type the column as a
PostgreSQL `ENUM`; featurizer then reads the labels via introspection — pass a
connection (`Featurizer("config.yaml", connection=conn)`), or it opens one from
`DATABASE_URL` / `PG*`. Either way the vocabulary is **fixed**; featurizer never
scans the data for distinct values (that learned, train-only transform belongs
to the consumer). See ADR-0007 and the top-level README.
