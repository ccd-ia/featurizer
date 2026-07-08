"""Developer benchmarking + golden-value tooling for featurizer.

This package is **outside the installable wheel** (``[tool.hatch.build.targets.wheel]
packages = ["featurizer"]``) — it exists to measure the correlated-subquery
aggregator scaling cliff (ADR-0009) and to freeze v0.5.2 aggregator semantics as
golden values before the set-based rewrite (plan:
``specs/correlated-subquery-aggregator-scaling.html``).

Entry points (run from the repo root against an ephemeral PostgreSQL — see the
``bench-aggs`` justfile recipe)::

    uv run python -m benchmarks inventory
    uv run python -m benchmarks capture-golden
    uv run python -m benchmarks bench --scale 1k

It imports only from ``featurizer`` (never from ``tests``); the integration test
``tests/integration/test_preagg_value_equality.py`` imports the *pure* case spec
in :mod:`benchmarks.preagg_cases` so capture and verification cannot drift.
"""
