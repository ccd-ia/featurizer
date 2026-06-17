#!/usr/bin/env python
"""Run Example 3: Deep Nesting."""

import argparse
import os
import sys
from pathlib import Path

# Add repo root (featurizer) and examples/ (_db) to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import _db

from featurizer import Featurizer

SCHEMA = "example_03"


def main():
    parser = argparse.ArgumentParser(description="Run Example 3: Deep Nesting")
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against database"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--show-depth", action="store_true", help="Show feature breakdown by depth"
    )
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

    # Show depth breakdown if requested
    if args.show_depth:
        print("\n🔍 Features by Entity:")
        for entity_alias, features in featurizer.features.items():
            print(f"  {entity_alias}: {len(features)} features")
            # Sample a few feature names
            sample = sorted(features, key=lambda f: f.name)[:3]
            for feat in sample:
                print(f"    - {feat.name}")

    # Show sample features
    print("\n🔍 Sample Target Features (first 15):")
    for i, feature in enumerate(sorted(target_features, key=lambda f: f.name)[:15], 1):
        print(f"  {i}. {feature.name}")

    # Show SQL if requested
    if args.show_sql:
        print("\n📝 Generated SQL Query (with deep nesting):")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    # Execute if requested
    if args.execute:
        print(
            f"\n⚙️  Executing query with depth={featurizer.max_depth} on PostgreSQL..."
        )

        # Point records/SQLAlchemy at the example_03 schema (psycopg3 + search_path).
        # Reads DATABASE_URL / PG* from the env; exits with guidance if unset.
        os.environ["DATABASE_URL"] = _db.records_url(SCHEMA)

        try:
            df = featurizer.to_dataframe()

            print("✓ Query executed successfully!")
            print(f"\nResults shape: {df.shape}")
            print(f"\nColumn count: {len(df.columns)}")
            print("\nFirst 5 rows (first 10 columns):")
            print(df.iloc[:, :10].head())

            # Save to CSV if requested
            if args.output:
                output_path = Path(__file__).parent / args.output
                df.to_csv(output_path, index=False)
                print(f"\n✓ Results saved to: {output_path}")

        except Exception as e:
            print(f"\n✗ Error executing query: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
