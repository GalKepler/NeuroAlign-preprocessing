"""Parse parcellated TSV + JSON sidecar pairs into DataFrames."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def parse_bids_entities(filename: str, keys: Optional[list[str]] = None) -> dict[str, str]:
    """Extract BIDS key-value entities from a filename.

    Parameters
    ----------
    filename:
        BIDS-formatted filename (basename only, or full path — stem is used).
    keys:
        If provided, return only these keys. Keys absent from the filename are
        omitted from the result. If None, all entities are returned.

    Returns
    -------
    dict[str, str]
        Mapping of entity key -> value strings.
    """
    stem = Path(filename).stem
    # Handle double extensions like .nii.gz
    stem = stem.removesuffix(".nii")
    entities: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            k, v = part.split("-", 1)
            entities[k] = v
    if keys is not None:
        return {k: entities[k] for k in keys if k in entities}
    return entities


def read_parcellation(
    tsv_path: Path,
    extra_columns: Optional[dict[str, str]] = None,
) -> Optional[pd.DataFrame]:
    """Read a parcellation TSV and its JSON sidecar, returning a unified DataFrame.

    The JSON sidecar (same path with ``.json`` extension) is read and flattened:
    nested dicts become ``{parent}_{child}`` columns. All sidecar fields are
    appended as constant columns to every row of the TSV.

    Parameters
    ----------
    tsv_path:
        Path to the ``.tsv`` parcellation file.
    extra_columns:
        Additional constant columns to add (e.g. ``{"subject_code": "0001",
        "tissue": "GM"}``).  Applied after sidecar columns so they take
        precedence on name conflicts.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with TSV data + flattened JSON metadata + extra_columns, or
        ``None`` if the TSV file does not exist.
    """
    if not tsv_path.exists():
        return None

    df = pd.read_csv(tsv_path, sep="\t")

    json_path = tsv_path.with_suffix(".json")
    if json_path.exists():
        with open(json_path) as f:
            metadata = json.load(f)
        for key, value in metadata.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    df[f"{key}_{subkey}"] = subvalue
            else:
                df[key] = value
    else:
        logger.warning("No sidecar JSON found for %s", tsv_path)

    if extra_columns:
        for col, val in extra_columns.items():
            df[col] = val

    return df
