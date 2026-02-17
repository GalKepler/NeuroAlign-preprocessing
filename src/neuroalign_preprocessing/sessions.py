"""Session utilities for NeuroAlign preprocessing."""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def sanitize_subject_code(subject_code: str) -> str:
    """Remove special characters and zero-pad to 4 digits."""
    return subject_code.replace("-", "").replace("_", "").replace(" ", "").zfill(4)


def sanitize_session_id(session_id: Union[str, int, float]) -> str:
    """Convert to string, clean, and zero-pad to 12 digits."""
    if isinstance(session_id, float):
        if pd.isna(session_id):
            return ""
        session_str = str(int(session_id))
    else:
        session_str = str(session_id)
    return session_str.replace("-", "").replace("_", "").replace(" ", "").zfill(12)


def load_sessions(csv_path: Union[str, Path]) -> pd.DataFrame:
    """Load and sanitize a linked_sessions CSV file.

    Expects columns SubjectCode and ScanID. Returns a DataFrame with
    ``subject_code`` and ``session_id`` columns (sanitized, de-duplicated).

    Parameters
    ----------
    csv_path:
        Path to the linked_sessions.csv file.
    """
    df = pd.read_csv(csv_path)
    df["subject_code"] = df["SubjectCode"].apply(sanitize_subject_code)
    df["session_id"] = df["ScanID"].apply(sanitize_session_id)
    return df[["subject_code", "session_id"]].drop_duplicates().reset_index(drop=True)
