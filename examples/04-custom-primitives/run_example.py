#!/usr/bin/env python
"""Run Example 4: Custom Primitives."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import featurizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from featurizer import Featurizer

# Import and register custom primitives
from custom_primitives import register_all_custom_primitives


def main():
    parser = argparse.ArgumentParser(description="Run Example 4: Custom Primitives")
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against database"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    parser.add_argument(
        "--list-primitives", action="store_true", help="List all registered primitives"
    )
    args = parser.parse_args()

    # Register custom primitives BEFORE loading configuration
    print("Registering custom primitives...")
    register_all_custom_primitives()

    # List primitives if requested
    if args.list_primitives:
        from featurizer.primitives.utils import list_aggregations, list_transformations

        print("\n📋 Registered Aggregations:")
        for name in sorted(list_aggregations()):
            print(f"  - {name}")

        print("\n📋 Registered Transformations:")
        for name in sorted(list_transformations()):
            print(f"  - {name}")

        if not (args.show_sql or args.execute):
            return

    # Check if database exists
    db_path = Path(__file__).parent / "data.db"
    if not db_path.exists():
        print("Error: Database not found. Run 'python create_data.py' first.")
        sys.exit(1)

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"

    print("\nLoading configuration...")
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

    # Show features using custom primitives
    print("\n🔍 Features using custom primitives:")
    custom_primitives = ["median", "p95", "range", "log", "zscore", "bin"]
    custom_features = [
        f
        for f in target_features
        if any(prim in f.name.lower() for prim in custom_primitives)
    ]

    if custom_features:
        for feat in sorted(custom_features, key=lambda f: f.name)[:20]:
            print(f"  - {feat.name}")
    else:
        print("  (No features using custom primitives found)")
        print(
            "  Note: Custom primitives need to be explicitly used in feature generation"
        )

    # Show sample features
    print("\n🔍 Sample Features (first 10):")
    for i, feature in enumerate(sorted(target_features, key=lambda f: f.name)[:10], 1):
        print(f"  {i}. {feature.name}")

    # Show SQL if requested
    if args.show_sql:
        print("\n📝 Generated SQL Query (with custom primitives):")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    # Execute if requested
    if args.execute:
        print("\n⚙️  Executing query...")

        # Set DATABASE_URL for records library
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

        try:
            df = featurizer.to_dataframe()

            print("✓ Query executed successfully!")
            print(f"\nResults shape: {df.shape}")
            print("\nFirst 5 rows:")
            print(df.head())

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
