"""CLI entry point for neuroalign-preprocessing."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def _env_path(var: str) -> Optional[Path]:
    val = os.getenv(var)
    return Path(os.path.expanduser(val)) if val else None


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description=(
            "Aggregate pre-parcellated CAT12 and QSIParc TSV files into "
            "per-metric long-format Parquet files."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sessions", "-s",
        type=Path,
        default=_env_path("SESSIONS_CSV"),
        help="Path to linked_sessions.csv (env: SESSIONS_CSV)",
    )
    parser.add_argument(
        "--cat12-root",
        type=Path,
        default=_env_path("CAT12_PARCELLATED_ROOT"),
        help="Root of CAT12 parcellated outputs (env: CAT12_PARCELLATED_ROOT)",
    )
    parser.add_argument(
        "--qsiparc-root",
        type=Path,
        default=_env_path("QSIPARC_PATH"),
        help="Root of QSIParc outputs containing qsirecon-* dirs (env: QSIPARC_PATH)",
    )
    parser.add_argument(
        "--atlas",
        default=os.getenv("ATLAS_NAME", "Schaefer2018N400n7Tian2020S3"),
        help="Atlas name (env: ATLAS_NAME)",
    )
    parser.add_argument(
        "--mask",
        default="gm",
        help="Mask label used in parcellated filenames",
    )
    parser.add_argument(
        "--maskthr",
        type=int,
        default=None,
        help="Mask threshold value to include in output filename (e.g., 50)",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/parcellations"),
        help="Output directory",
    )
    parser.add_argument(
        "--tissues",
        nargs="+",
        default=["GM", "WM", "CT"],
        help="CAT12 tissues to aggregate",
    )
    parser.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "gzip", "zstd", "none"],
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.sessions:
        print(
            "Error: --sessions / SESSIONS_CSV is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.cat12_root and not args.qsiparc_root:
        print(
            "Error: at least one of --cat12-root / --qsiparc-root must be set.",
            file=sys.stderr,
        )
        sys.exit(1)

    compression = None if args.compression == "none" else args.compression
    args.output.mkdir(parents=True, exist_ok=True)

    from .aggregate import aggregate_cat12, aggregate_qsiparc

    total_rows = 0

    if args.cat12_root:
        results = aggregate_cat12(
            cat12_root=args.cat12_root,
            sessions_csv=args.sessions,
            output_dir=args.output,
            atlas=args.atlas,
            tissues=tuple(args.tissues),
            mask=args.mask,
            maskthr=args.maskthr,
            compression=compression,
        )
        for tissue, n in results.items():
            print(f"  cat12/{tissue.lower()}.parquet  →  {n:,} rows")
            total_rows += n

    if args.qsiparc_root:
        results = aggregate_qsiparc(
            qsiparc_root=args.qsiparc_root,
            sessions_csv=args.sessions,
            output_dir=args.output,
            atlas=args.atlas,
            mask=args.mask,
            maskthr=args.maskthr,
            compression=compression,
        )
        for key, n in results.items():
            print(f"  qsiparc/{key}.parquet  →  {n:,} rows")
            total_rows += n

    print(f"\nDone. {total_rows:,} total rows written to {args.output}/")
    sys.exit(0)
