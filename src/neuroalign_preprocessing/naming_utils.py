from typing import Optional, Dict, Any
from pathlib import Path

def generate_parquet_filename(
    entities: Dict[str, Any],
    modality: str, # "cat12" or "qsiparc"
    mask: Optional[str] = None,
    maskthr: Optional[int] = None,
) -> str:
    """Generates a BIDS-like filename for output Parquet files.

    Parameters
    ----------
    entities : Dict[str, Any]
        A dictionary containing BIDS entities like 'sub', 'ses', 'atlas',
        'tissue', 'model', 'param'.
    modality : str
        The modality of the data, either "cat12" or "qsiparc".
    mask : Optional[str]
        The mask label (e.g., "gm").
    maskthr : Optional[int]
        The mask threshold value (e.g., 50).

    Returns
    -------
    str
        The generated filename with a .parquet extension.
    """
    parts = []

    # Required entities from input
    for key in ["sub", "ses", "atlas"]:
        if key in entities and entities[key] is not None:
            parts.append(f"{key}-{entities[key]}")

    # Fixed entities from examples
    # These values appear to be constant based on the provided examples.
    parts.append("space-MNI152NLin2009cAsym")
    parts.append("res-01")

    # Modality-specific entities
    if modality == "cat12":
        if "tissue" in entities and entities["tissue"] is not None:
            parts.append(f"tissue-{entities['tissue']}")
    elif modality == "qsiparc":
        if "model" in entities and entities["model"] is not None:
            parts.append(f"model-{entities['model']}")
        if "param" in entities and entities["param"] is not None:
            parts.append(f"param-{entities['param']}")

    # Optional mask entities
    if mask:
        parts.append(f"mask-{mask}")
    if maskthr is not None:
        parts.append(f"maskthr-{maskthr}")

    return "_".join(parts) + "_parc.parquet"

