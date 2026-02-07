from typing import Union
import pandas as pd


def sanitize_subject_code(subject_code: str) -> str:
    """
    Sanitize subject code by removing special characters and padding with zeros.

    Args:
        subject_code (str): Original subject code.
    Returns:
        str: Sanitized subject code.
    """
    sanitized = subject_code.replace("-", "").replace("_", "").replace(" ", "").zfill(4)
    return sanitized


def sanitize_session_id(session_id: Union[str, int, float]) -> str:
    """
    Sanitize session ID by converting to string, removing special characters, and padding with zeros.

    Args:
        session_id (Union[str, int, float]): Original session ID.
    Returns:
        str: Sanitized session ID.
    """
    if isinstance(session_id, float):
        if pd.isna(session_id):
            return ""
        session_str = str(int(session_id))
    else:
        session_str = str(session_id)
    sanitized = session_str.replace("-", "").replace("_", "").replace(" ", "").zfill(12)
    return sanitized
