#!/usr/bin/env python
"""Run Example 5: Direct categoricals, output formats, and imputation."""

import argparse
import os
import sys
from pathlib import Path

# Add repo root (featurizer) and examples/ (_db) to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import _db

from featurizer import Featurizer

SCHEMA = "example_05"


def main():
    parser = argparse.ArgumentParser(
        description="Run Example 5: Direct categoricals, output & imputation"
    )
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against PostgreSQL"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"

    print("Loading configuration...")
    featurizer = Featurizer(str(config_path))

    print("\n📊 Feature Generation Summary")
    print(f"  Target entity: {featurizer.target.alias}")
    print(f"  Intervals: {', '.join(featurizer.intervals)}")

    one_hots = [e for e in featurizer.feature_manifest if e.kind == "one_hot"]
    print(f"  One-hot columns: {[e.column for e in one_hots]}")

    if args.show_sql:
        print("\n📝 Generated SQL Query:")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    if args.execute:
        print("\n⚙️  Executing query on PostgreSQL...")
        os.environ["DATABASE_URL"] = _db.records_url(SCHEMA)
        try:
            df = featurizer.to_dataframe(impute=True)
            print("✓ Query executed successfully!")
            print(f"\nResults shape: {df.shape}")
            one_hot_cols = [e.column for e in one_hots]
            missing_cols = [c for c in df.columns if c.endswith("__missing")]
            print("\nOne-hot columns (first 5 rows):")
            print(df[one_hot_cols].head())
            print(f"\n__missing indicator columns: {missing_cols}")
            if args.output:
                output_path = Path(__file__).parent / args.output
                df.to_csv(output_path)
                print(f"\n✓ Results saved to: {output_path}")
        except Exception as e:
            print(f"\n✗ Error executing query: {e}")
            print(
                "  Is the data loaded? Run `python create_data.py` (or "
                "`just example 05`) against a running PostgreSQL (`just db-up`)."
            )
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
