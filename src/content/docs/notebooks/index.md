---
title: Tutorial notebooks
description: >-
  Five executed Jupyter notebooks, from basic aggregations to custom
  primitives — rendered here from their committed outputs.
sidebar:
  order: 0
  label: Learning path
---

Five hands-on tutorials, each a Jupyter notebook in the repo under
[`examples/`](https://github.com/ccd-ia/featurizer/tree/master/examples).
The pages here are rendered from the notebooks' **committed, executed
outputs** — what you see is what ran against a real PostgreSQL. Follow them
in order:

| # | Tutorial | Difficulty | Scenario |
|---|---|---|---|
| 1 | [Basic aggregations](/featurizer/notebooks/01-basic-aggregations/) | Beginner | E-commerce: Customers → Orders |
| 2 | [Temporal joins](/featurizer/notebooks/02-temporal-joins/) | Intermediate | Healthcare: Patients → Care plans |
| 3 | [Deep nesting](/featurizer/notebooks/03-deep-nesting/) | Intermediate | Retail supply chain, depth 3 |
| 4 | [Custom primitives](/featurizer/notebooks/04-custom-primitives/) | Advanced | Financial analytics: Accounts → Transactions |
| 5 | [Categoricals, output & imputation](/featurizer/notebooks/05-categoricals-output/) | Advanced | Food inspections: Facilities → Inspections |

**What each one teaches**

1. **Basic aggregations** — parent-child aggregations (`count`, `sum`, `mean`,
   `min`, `max`, `stddev`, `nunique`), time windows (`P7D`, `P30D`), feature
   naming. *Start here* (it is the notebook form of the
   [walkthrough](/featurizer/walkthrough/)).
2. **Temporal joins** — as-of join semantics, grace periods, point-in-time
   generation, the `LEFT JOIN LATERAL` SQL, and rolling stats.
3. **Deep nesting** — multi-level relationships (`max_depth: 3`), feature
   propagation across chains, and how the CTE structure grows.
4. **Custom primitives** — writing and registering your own aggregations
   (`range`, `p95`) and transformations (`log1p`, `zscore`, `bin`), then
   selecting them in `config.yaml`.
5. **Categoricals, output & imputation** — fixed-vocabulary one-hot encoding
   (`role: categorical` / `role: identifier`), the feature manifest, output
   formats (`to_dataframe` / `to_arrow`), and the imputation contract. Unlike
   1–4, this one **executes** against PostgreSQL throughout.

To run them yourself:

```bash
git clone https://github.com/ccd-ia/featurizer.git && cd featurizer
uv sync && just db-up
uv run jupyter lab examples/
```
