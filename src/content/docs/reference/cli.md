---
title: CLI
description: >-
  The featurizer command line — discover primitives and validate
  configurations without writing any Python.
sidebar:
  order: 3
---

featurizer ships a small CLI for the two things you want before writing code:
*what primitives exist?* and *is my config valid?*

```bash
uv run python -m featurizer --help
```

```text
usage: featurizer [-h] {list-primitives,lp,validate} ...

Featurizer - Deep Feature Synthesis for PostgreSQL

positional arguments:
  {list-primitives,lp,validate}
    list-primitives (lp)   List available aggregation and transformation primitives
    validate               Validate a configuration file
```

## `list-primitives` — discover the registry

```bash
uv run python -m featurizer list-primitives [--type agg|transform|all]
                                            [--category] [--show-sql]
```

| flag | effect |
|---|---|
| `--type` / `-t` | `agg`, `transform`, or `all` (default) |
| `--category` / `-c` | group by category instead of a flat list |
| `--show-sql` / `-s` | include an example SQL rendering per primitive |

`lp` is a shorthand alias. Grouped output looks like:

```text
============================================================
AGGREGATION PRIMITIVES (67 available)
============================================================

  Temporal (support interval windows):
    acf_1                     Lag-1 autocorrelation: corr(x_t, x_{t-1})
                                inputs: numeric
    age_in_system             Alias of tenure: days since the first observed event
                                inputs: index
    all                       True if all values are true (boolean AND)
                                inputs: boolean
    …
```

The same information, browsable with search, lives in the
[primitives reference](/featurizer/reference/primitives/) — generated from
this registry at every deploy.

## `validate` — check a config before running anything

```bash
uv run python -m featurizer validate config.yaml
```

```text
Configuration is valid: examples/01-basic-aggregations/config.yaml
```

Validation covers structure (required keys, types), values (ISO-8601
intervals, known variable types), semantics (the target exists, relationship
endpoints exist, as-of joins have a timestamp to order by), and best-practice
warnings (`max_depth > 5`, more than 10 intervals). Unknown primitive names
get a *did you mean?* suggestion from the registry. Errors report their exact
config location — see the
[configuration reference](/featurizer/reference/configuration/) for every key.

The same validation runs automatically inside `Featurizer("config.yaml")`;
the CLI just gives you the answer without leaving the shell.
