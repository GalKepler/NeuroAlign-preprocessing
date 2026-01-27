"""
Export SQLite cache to Parquet files.

This utility allows you to manually export a partially completed pipeline run
from the SQLite cache to Parquet files, making it readable by the FeatureStore.
"""

import argparse
import logging
import sys
from pathlib import Path

from neuroalign_preprocessing.preprocessing.feature_store import FeatureStore

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export SQLite cache to Parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Export cache from a failed pipeline run
  neuroalign-export-cache data/processed

  # Check cache status first
  neuroalign-export-cache data/processed --status

  # Export and keep cache
  neuroalign-export-cache data/processed --keep-cache
        """,
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory (same as used with neuroalign-preprocess)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit (don't export)",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep SQLite cache after exporting (default: delete)",
    )
    parser.add_argument(
        "--atlas-name",
        default="",
        help="Atlas name for metadata (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)

    # Initialize store
    store = FeatureStore(args.output_dir)

    # Check if cache exists
    if not store.has_cache():
        logger.error(f"No cache found in {args.output_dir}")
        logger.info("Cache is created during pipeline runs and contains partially loaded data")
        return 1

    # Show status
    status = store.cache_status()
    print("\n" + "=" * 60)
    print("CACHE STATUS")
    print("=" * 60)
    print(f"Location: {store.cache_db_path}")
    print(f"Size: {status['size_mb']:.2f} MB")
    print(f"Anatomical sessions: {status['n_sessions_anatomical']}")
    print(f"Diffusion sessions: {status['n_sessions_diffusion']}")
    print(f"\nTables: {len(status['tables'])}")
    for table in status['tables']:
        print(f"  - {table}")
    print("=" * 60)

    if args.status:
        return 0

    # Export cache
    print("\nExporting cache to Parquet files...")
    try:
        long_formats = store.export_cache_to_parquet(atlas_name=args.atlas_name)

        print(f"\n✓ Exported {len(long_formats)} long formats:")
        for fmt in long_formats:
            print(f"  - {fmt}")

        # Generate wide features
        print("\nGenerating wide-format features...")
        wide_features = store.generate_wide_features()
        print(f"✓ Generated {len(wide_features)} wide features")

        # Clear cache unless --keep-cache
        if not args.keep_cache:
            store.clear_cache()
            print("\n✓ Cache cleared")
        else:
            print("\n✓ Cache kept (use --keep-cache=false to remove)")

        print("\nExport complete! You can now use FeatureStore to load the data:")
        print("  from neuroalign_preprocessing.preprocessing import FeatureStore")
        print(f"  store = FeatureStore('{args.output_dir}')")
        print("  features = store.list_features()")

        return 0

    except Exception as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
