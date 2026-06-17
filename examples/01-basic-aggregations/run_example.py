#!/usr/bin/env python
"""Run Example 1: Basic Aggregations."""

import argparse
import os
import sys
from pathlib import Path

# Add repo root (featurizer) and examples/ (_db) to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import _db

from featurizer import Featurizer

SCHEMA = "example_01"


def main():
    parser = argparse.ArgumentParser(description="Run Example 1: Basic Aggregations")
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against PostgreSQL"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"

    print("Loading configuration...")
    featurizer = Featurizer(str(config_path))

    # Show statistics
    print("\n📊 Feature Generation Summary")
    print(f"  Target entity: {featurizer.target.alias}")
    print(f"  Max depth: {featurizer.max_depth}")
    print(f"  Intervals: {', '.join(featurizer.intervals)}")
    print(f"  Entities: {len(list(featurizer.entities))}")
    print(f"  Relationships: {len(featurizer.relationships)}")

    target_features = featurizer.features[featurizer.target.alias]
    print(f"  Generated features: {len(target_features)}")

    # Show sample features
    print("\n🔍 Sample Features (first 10):")
    for i, feature in enumerate(sorted(target_features, key=lambda f: f.name)[:10], 1):
        print(f"  {i}. {feature.name}")

    # Show SQL if requested
    if args.show_sql:
        print("\n📝 Generated SQL Query:")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    # Execute if requested
    if args.execute:
        print("\n⚙️  Executing query on PostgreSQL...")

        # Point records/SQLAlchemy at the example_01 schema (psycopg3 + search_path).
        # Reads DATABASE_URL / PG* from the env; exits with guidance if unset.
        os.environ["DATABASE_URL"] = _db.records_url(SCHEMA)

        try:
            df = featurizer.to_dataframe()

            print("✓ Query executed successfully!")
            print(f"\nResults shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())

            # Save to CSV if requested
            if args.output:
                output_path = Path(__file__).parent / args.output
                df.to_csv(output_path)
                print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error executing query: {e}")
            print(
                "  Is the data loaded? Run `python create_data.py` (or "
                "`just example 01`) against a running PostgreSQL (`just db-up`)."
            )
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
