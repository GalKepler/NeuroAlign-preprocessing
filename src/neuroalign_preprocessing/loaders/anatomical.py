"""
Anatomical MRI Data Loader for NeuroAlign
==========================================

Loads pre-parcellated CAT12 anatomical data (GM volume, WM volume, cortical thickness)
from TSV files. Extracts TIV directly from CAT12 XML files (no MATLAB dependency).

Example:
    >>> from neuroalign_preprocessing.loaders import AnatomicalLoader
    >>> loader = AnatomicalLoader(
    ...     cat12_root="/path/to/cat12/derivatives",
    ...     cat12_parcellated_root="/path/to/cat12_parcellated",
    ...     atlas_name="4S456Parcels"
    ... )
    >>> df = loader.load_sessions(sessions_csv="linked_sessions.csv", n_jobs=8)
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AnatomicalPaths:
    """Configuration for anatomical data paths."""

    cat12_root: Path  # Original CAT12 derivatives (for XML/TIV extraction)
    cat12_parcellated_root: Path  # Pre-parcellated TSVs
    atlas_name: str = "4S456Parcels"


def _extract_tiv_from_xml(xml_path: Path) -> Optional[float]:
    """
    Extract vol_TIV from CAT12 XML <subjectmeasures> section.

    The XML contains multiple vol_TIV elements - some with description text
    and some with actual numeric values. This function finds the first
    valid numeric TIV value from the root-level subjectmeasures section.

    Args:
        xml_path: Path to CAT12 XML file (cat_*.xml)

    Returns:
        TIV value in mL, or None if extraction fails
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Find vol_TIV in subjectmeasures sections
        # Note: XML may have multiple subjectmeasures - some nested with descriptions,
        # and a root-level one with actual numeric values
        for subj_measures in root.iter("subjectmeasures"):
            vol_tiv = subj_measures.find("vol_TIV")
            if vol_tiv is not None and vol_tiv.text:
                try:
                    return float(vol_tiv.text.strip())
                except ValueError:
                    # This vol_TIV contains description text, not a number
                    continue

        # Alternative: look directly for vol_TIV anywhere in the tree
        # Skip subjectratings which contains normalized values
        for vol_tiv in root.iter("vol_TIV"):
            if vol_tiv.text:
                # Check if parent is subjectratings (we want subjectmeasures)
                # Since ElementTree doesn't have parent access, just try to parse
                try:
                    return float(vol_tiv.text.strip())
                except ValueError:
                    continue

        logger.debug(f"No numeric vol_TIV found in XML: {xml_path}")
        return None

    except ET.ParseError as e:
        logger.warning(f"Failed to parse XML {xml_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error extracting TIV from {xml_path}: {e}")
        return None


def parse_bids_entities(filename: str) -> Dict[str, str]:
    """
    Parse BIDS filename entities.

    Args:
        filename: BIDS-formatted filename

    Returns:
        Dictionary of entity key-value pairs
    """
    entities = {}
    parts = Path(filename).name.split("_")
    for part in parts:
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
    return entities


class AnatomicalLoader:
    """
    Loader for CAT12-processed anatomical MRI data.

    Loads pre-parcellated TSV files (GM, WM, CT) and extracts TIV from
    CAT12 XML files. Uses ThreadPoolExecutor for efficient I/O-bound
    batch loading.

    Attributes:
        paths: AnatomicalPaths configuration
        n_jobs: Number of parallel workers for loading
    """

    def __init__(
        self,
        cat12_root: Path,
        cat12_parcellated_root: Path,
        atlas_name: str = "4S456Parcels",
        n_jobs: int = 1,
    ):
        """
        Initialize anatomical data loader.

        Args:
            cat12_root: Path to CAT12 derivatives directory (for XML/TIV extraction)
            cat12_parcellated_root: Path to pre-parcellated CAT12 TSVs
            atlas_name: Name of parcellation atlas
            n_jobs: Number of parallel workers (default: 1 for serial processing)
        """
        self.paths = AnatomicalPaths(
            cat12_root=Path(cat12_root),
            cat12_parcellated_root=Path(cat12_parcellated_root),
            atlas_name=atlas_name,
        )
        self.n_jobs = n_jobs

    def get_session_directory(self, subject: str, session: str) -> Optional[Path]:
        """
        Get pre-parcellated TSV directory for a subject/session.

        Args:
            subject: Subject code (without 'sub-' prefix)
            session: Session ID (without 'ses-' prefix)

        Returns:
            Path to session atlas directory or None if not found
        """
        session_dir = (
            self.paths.cat12_parcellated_root
            / "cat12"
            / f"sub-{subject}"
            / f"ses-{session}"
            / "anat"
            / f"atlas-{self.paths.atlas_name}"
        )
        return session_dir if session_dir.exists() else None

    def get_cat12_directory(self, subject: str, session: str) -> Optional[Path]:
        """
        Get CAT12 output directory for a subject/session (for XML/TIV).

        Args:
            subject: Subject code (without 'sub-' prefix)
            session: Session ID (without 'ses-' prefix)

        Returns:
            Path to CAT12 anat directory or None if not found
        """
        cat12_dir = self.paths.cat12_root / f"sub-{subject}" / f"ses-{session}" / "anat"
        return cat12_dir if cat12_dir.exists() else None

    def _get_tiv_for_session(self, subject: str, session: str) -> Optional[float]:
        """
        Extract TIV from CAT12 XML for a session.

        Tries corrected T1w first, then uncorrected, then plain T1w.

        Args:
            subject: Subject code
            session: Session ID

        Returns:
            TIV value in mL, or None if not found
        """
        cat12_dir = self.get_cat12_directory(subject, session)
        if cat12_dir is None:
            return None

        # Try different XML naming patterns in order of preference
        patterns = [
            f"cat_sub-{subject}_ses-{session}_ce-corrected_T1w.xml",
            f"cat_sub-{subject}_ses-{session}_ce-uncorrected_T1w.xml",
            f"cat_sub-{subject}_ses-{session}_T1w.xml",
        ]

        for pattern in patterns:
            xml_path = cat12_dir / pattern
            if xml_path.exists():
                tiv = _extract_tiv_from_xml(xml_path)
                if tiv is not None:
                    return tiv

        # Fallback: glob for any cat_*.xml file
        xml_files = list(cat12_dir.glob("cat_*.xml"))
        if xml_files:
            return _extract_tiv_from_xml(xml_files[0])

        return None

    def load_session(
        self,
        subject: str,
        session: str,
        include_metadata: bool = True,
        include_tiv: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load pre-parcellated TSVs for a single session.

        Args:
            subject: Subject code (without 'sub-' prefix)
            session: Session ID (without 'ses-' prefix)
            include_metadata: Whether to include subject/session columns
            include_tiv: Whether to extract and include TIV

        Returns:
            DataFrame with regional features or None if session not found
        """
        session_dir = self.get_session_directory(subject, session)
        if session_dir is None:
            return None

        dfs = []

        # Tissue types and their corresponding modality/metric names
        tissue_config = {
            "GM": ("gm", "volume"),
            "WM": ("wm", "volume"),
            "CT": ("ct", "thickness"),
        }

        for tissue, (modality, metric) in tissue_config.items():
            tsv_files = list(session_dir.glob(f"*_tissue-{tissue}_parc.tsv"))
            if tsv_files:
                df = pd.read_csv(tsv_files[0], sep="\t")
                df["modality"] = modality
                df["metric"] = metric

                if include_metadata:
                    df["subject_code"] = subject
                    df["session_id"] = session

                dfs.append(df)

        if not dfs:
            return None

        result_df = pd.concat(dfs, ignore_index=True)

        # Add TIV if requested
        if include_tiv:
            tiv = self._get_tiv_for_session(subject, session)
            if tiv is not None:
                result_df["tiv"] = tiv

        return result_df

    def _load_session_worker(
        self,
        row: Dict[str, Any],
        include_tiv: bool,
    ) -> Tuple[str, str, Optional[pd.DataFrame]]:
        """
        Worker function for parallel session loading.

        Args:
            row: Dictionary with session metadata (subject_code, session_id, etc.)
            include_tiv: Whether to extract and include TIV

        Returns:
            Tuple of (subject_code, session_id, DataFrame or None)
        """
        subject = row["subject_code"]
        session = row["session_id"]

        session_data = self.load_session(
            subject=subject,
            session=session,
            include_metadata=True,
            include_tiv=include_tiv,
        )

        if session_data is not None:
            # Add metadata columns from sessions CSV
            for col, val in row.items():
                if col not in session_data.columns:
                    session_data[col] = val

        return subject, session, session_data

    def load_sessions(
        self,
        sessions_csv: Path,
        n_jobs: Optional[int] = None,
        progress: bool = True,
        include_qc: bool = True,  # Kept for API compatibility, not used
        normalize_by_tiv: bool = False,
        tiv_output_dir: Optional[Path] = None,  # Kept for API compatibility, not used
        calculate_tiv: bool = True,
    ) -> pd.DataFrame:
        """
        Load anatomical data for multiple sessions.

        Uses ThreadPoolExecutor for efficient parallel loading when n_jobs > 1.

        Args:
            sessions_csv: Path to CSV with 'subject_code' and 'session_id' columns
            n_jobs: Number of parallel workers (overrides instance setting)
            progress: Whether to show progress bar
            include_qc: Kept for API compatibility (not used with pre-parcellated data)
            normalize_by_tiv: Whether to normalize volumes by TIV
            tiv_output_dir: Kept for API compatibility (not used - TIV from XML)
            calculate_tiv: Whether to extract TIV from XML files

        Returns:
            DataFrame with all sessions' regional features
        """
        sessions = pd.read_csv(sessions_csv, dtype={"subject_code": str, "session_id": str})
        effective_jobs = n_jobs if n_jobs is not None else self.n_jobs

        if effective_jobs == 1:
            df = self._load_sessions_serial(sessions, progress, calculate_tiv)
        else:
            df = self._load_sessions_parallel(sessions, progress, effective_jobs, calculate_tiv)

        # Optionally normalize volume columns by TIV
        if normalize_by_tiv and "tiv" in df.columns:
            volume_mask = df["metric"] == "volume"
            if volume_mask.any():
                if "volume_mm3" in df.columns:
                    df.loc[volume_mask, "volume_mm3_normalized"] = (
                        df.loc[volume_mask, "volume_mm3"] / df.loc[volume_mask, "tiv"]
                    )
                    logger.info(f"Normalized {volume_mask.sum()} volume measurements by TIV")

        return df

    def _load_sessions_serial(
        self,
        sessions: pd.DataFrame,
        progress: bool,
        include_tiv: bool,
    ) -> pd.DataFrame:
        """
        Serial loading.

        Args:
            sessions: DataFrame with subject_code and session_id columns
            progress: Whether to show progress bar
            include_tiv: Whether to extract TIV

        Returns:
            DataFrame with all sessions' regional features
        """
        results: List[pd.DataFrame] = []
        iterator = sessions.iterrows()

        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(sessions), desc="Loading anatomical data")
            except ImportError:
                pass

        for _, row in iterator:
            subject = row["subject_code"]
            session = row["session_id"]

            session_data = self.load_session(
                subject=subject,
                session=session,
                include_metadata=True,
                include_tiv=include_tiv,
            )

            if session_data is not None:
                # Add metadata columns from sessions CSV
                for col in sessions.columns:
                    if col not in session_data.columns:
                        session_data[col] = row[col]
                results.append(session_data)

        if not results:
            raise ValueError("No sessions successfully loaded")

        return pd.concat(results, ignore_index=True, copy=False)

    def _load_sessions_parallel(
        self,
        sessions: pd.DataFrame,
        progress: bool,
        n_jobs: int,
        include_tiv: bool,
    ) -> pd.DataFrame:
        """
        Parallel loading using ThreadPoolExecutor.

        Uses threads (not processes) since TSV loading is I/O-bound.

        Args:
            sessions: DataFrame with subject_code and session_id columns
            progress: Whether to show progress bar
            n_jobs: Number of parallel workers
            include_tiv: Whether to extract TIV

        Returns:
            DataFrame with all sessions' regional features
        """
        results: List[pd.DataFrame] = []
        success_count = 0
        skip_count = 0
        error_count = 0
        failed_sessions: List[Tuple[str, str, str]] = []

        logger.info(f"Loading anatomical data with {n_jobs} parallel workers")
        logger.debug(f"Total sessions to process: {len(sessions)}")

        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(
                    self._load_session_worker,
                    row.to_dict(),
                    include_tiv,
                )
                for _, row in sessions.iterrows()
            ]

            iterator = as_completed(futures)
            if progress:
                try:
                    from tqdm import tqdm

                    iterator = tqdm(iterator, total=len(futures), desc="Loading anatomical data")
                except ImportError:
                    pass

            for fut in iterator:
                try:
                    subject, session, session_data = fut.result()
                    if session_data is not None:
                        results.append(session_data)
                        success_count += 1
                        logger.debug(f"  SUCCESS: sub-{subject}_ses-{session}")
                    else:
                        skip_count += 1
                        failed_sessions.append((subject, session, "no_data"))
                        logger.debug(f"  SKIP: sub-{subject}_ses-{session} - no data found")
                except Exception as e:
                    error_count += 1
                    logger.error(f"Worker exception: {e}", exc_info=True)

        # Log summary
        logger.info(
            f"Anatomical loading complete: "
            f"{success_count} success, {skip_count} skipped, {error_count} errors"
        )

        if failed_sessions:
            logger.debug(f"Failed/skipped sessions ({len(failed_sessions)} total):")
            for subj, sess, reason in failed_sessions[:50]:
                logger.debug(f"  sub-{subj}_ses-{sess}: {reason}")
            if len(failed_sessions) > 50:
                logger.debug(f"  ... and {len(failed_sessions) - 50} more")

        if not results:
            logger.error(
                f"No sessions successfully loaded. "
                f"Attempted {len(sessions)} sessions. "
                f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}"
            )
            raise ValueError("No sessions successfully loaded")

        logger.info(f"Successfully loaded {len(results)} sessions")
        return pd.concat(results, ignore_index=True, copy=False)
