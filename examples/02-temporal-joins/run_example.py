#!/usr/bin/env python
"""Run Example 2: Temporal Joins."""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import featurizer
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from featurizer import Featurizer


def main():
    parser = argparse.ArgumentParser(description="Run Example 2: Temporal Joins")
    parser.add_argument(
        "--show-sql", action="store_true", help="Print generated SQL query"
    )
    parser.add_argument(
        "--execute", action="store_true", help="Execute query against database"
    )
    parser.add_argument("--output", type=str, help="Save results to CSV file")
    args = parser.parse_args()

    # Check if database exists
    db_path = Path(__file__).parent / "data.db"
    if not db_path.exists():
        print("Error: Database not found. Run 'python create_data.py' first.")
        sys.exit(1)

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

    # Check for temporal relationships
    temporal_rels = [
        r for r in featurizer.relationships if hasattr(r, "temporal") and r.temporal
    ]
    if temporal_rels:
        print(f"  Temporal relationships: {len(temporal_rels)}")
        for rel in temporal_rels:
            mode = rel.temporal.get("mode", "N/A")
            grace = rel.temporal.get("grace", "none")
            print(
                f"    - {rel.parent.entity.alias} → {rel.child.entity.alias} (mode: {mode}, grace: {grace})"
            )

    target_features = featurizer.features[featurizer.target.alias]
    print(f"  Generated features: {len(target_features)}")

    # Show sample features
    print("\n🔍 Sample Features (first 10):")
    for i, feature in enumerate(sorted(target_features, key=lambda f: f.name)[:10], 1):
        print(f"  {i}. {feature.name}")

    # Show SQL if requested
    if args.show_sql:
        print("\n📝 Generated SQL Query (with temporal joins):")
        print("=" * 80)
        print(featurizer.query)
        print("=" * 80)

    # Execute if requested
    if args.execute:
        print("\n⚙️  Executing query with temporal joins...")

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
            sys.exit(1)

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
