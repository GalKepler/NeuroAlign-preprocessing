"""
Data preparation pipeline for NeuroAlign.

Loads neuroimaging data from multiple modalities and outputs to a FeatureStore
with long format (raw parcellator output) and wide format (per metric) files.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from neuroalign_preprocessing.loaders import AnatomicalLoader, DiffusionLoader

from .config import PipelineConfig
from .feature_store import FeatureStore

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result from data preparation pipeline."""

    store: FeatureStore
    metadata: Dict[str, Any]
    output_path: Path
    long_formats_saved: List[str]
    wide_features_generated: List[str]
    n_new_sessions: int = 0
    n_skipped_sessions: int = 0


class DataPreparationPipeline:
    """
    Main pipeline for preparing NeuroAlign feature matrices.

    Saves data in two formats:
    - Long format: Raw parcellator output with all columns preserved
    - Wide format: One file per metric for efficient modeling
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._sessions_df: Optional[pd.DataFrame] = None

    def _load_sessions_csv(self) -> pd.DataFrame:
        """Load and cache sessions CSV."""
        if self._sessions_df is None:
            self._sessions_df = pd.read_csv(
                self.config.paths.sessions_csv,
                dtype={"subject_code": str, "session_id": str},
            )
            # sanitize subject codes and session IDs
            self._sessions_df["subject_code"] = self._sessions_df["subject_code"].apply(
                self.sanitize_subject_code
            )
            self._sessions_df["session_id"] = self._sessions_df["session_id"].apply(
                self.sanitize_session_id
            )
            logger.info(f"Loaded {len(self._sessions_df)} sessions from CSV")
        return self._sessions_df

    def _validate_sessions_csv(self) -> None:
        """Validate that sessions CSV has required columns."""
        df = self._load_sessions_csv()
        required = ["subject_code", "session_id"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Sessions CSV missing required columns: {missing}")

    def _init_anatomical_loader(self) -> Optional[AnatomicalLoader]:
        """Initialize anatomical loader if paths are configured."""
        if not self.config.modalities.anatomical:
            logger.info("Anatomical modality disabled")
            return None

        if self.config.paths.cat12_root is None:
            logger.warning("Anatomical modality enabled but cat12_root not set")
            return None

        if self.config.paths.cat12_parcellated_root is None:
            logger.warning(
                "Anatomical modality enabled but cat12_parcellated_root not set"
            )
            return None

        return AnatomicalLoader(
            cat12_root=self.config.paths.cat12_root,
            cat12_parcellated_root=self.config.paths.cat12_parcellated_root,
            atlas_name=self.config.atlas_name,
            n_jobs=self.config.n_jobs,
        )

    def _init_diffusion_loader(self) -> Optional[DiffusionLoader]:
        """Initialize diffusion loader if paths are configured."""
        if not self.config.modalities.diffusion:
            logger.info("Diffusion modality disabled")
            return None

        if self.config.paths.qsiparc_path is None:
            logger.warning("Diffusion modality enabled but qsiparc_path not set")
            return None

        if self.config.paths.qsirecon_path is None:
            logger.warning("Diffusion modality enabled but qsirecon_path not set")
            return None

        return DiffusionLoader(
            qsiparc_path=self.config.paths.qsiparc_path,
            qsirecon_path=self.config.paths.qsirecon_path,
            workflows=self.config.modalities.diffusion_workflows,
            atlas_name=self.config.atlas_name,
            n_jobs=self.config.n_jobs,
        )

    def _get_anatomical_modalities(self) -> List[str]:
        """Get list of enabled anatomical modalities."""
        modalities = []
        if self.config.modalities.gray_matter:
            modalities.append("gm")
        if self.config.modalities.white_matter:
            modalities.append("wm")
        if self.config.modalities.cortical_thickness:
            modalities.append("ct")
        return modalities

    def _get_sessions_to_load(
        self, store: FeatureStore
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        """
        Get sessions that need to be loaded based on load status flags.

        Returns separate session lists for anatomical and diffusion to avoid
        redundant processing.

        Returns:
            Tuple of (all_sessions, anatomical_sessions, diffusion_sessions, n_skipped)
        """
        all_sessions = self._load_sessions_csv()

        if self.config.force:
            logger.info("Loading all sessions (force=True)")
            # Reset load status flags
            if store.metadata_path.exists():
                meta = store.load_metadata()
                for col in [
                    "anatomical_gm_loaded",
                    "anatomical_wm_loaded",
                    "anatomical_ct_loaded",
                    "diffusion_loaded",
                ]:
                    if col in meta.columns:
                        meta[col] = False
                meta.to_parquet(
                    store.metadata_path, compression=store.compression, index=False
                )
            return all_sessions, all_sessions, all_sessions, 0

        # Check metadata for load status
        if not store.metadata_path.exists():
            logger.info("No metadata found - loading all sessions")
            return all_sessions, all_sessions, all_sessions, 0

        meta = store.load_metadata()

        # Determine which modalities are enabled
        needs_anatomical_gm = (
            self.config.modalities.anatomical and self.config.modalities.gray_matter
        )
        needs_anatomical_wm = (
            self.config.modalities.anatomical and self.config.modalities.white_matter
        )
        needs_anatomical_ct = (
            self.config.modalities.anatomical
            and self.config.modalities.cortical_thickness
        )
        needs_diffusion = self.config.modalities.diffusion

        # Build filter conditions separately for anatomical and diffusion
        needs_anatomical = pd.Series([False] * len(meta), index=meta.index)

        if needs_anatomical_gm and "anatomical_gm_loaded" in meta.columns:
            needs_anatomical |= ~meta["anatomical_gm_loaded"]
        elif needs_anatomical_gm:
            needs_anatomical |= True

        if needs_anatomical_wm and "anatomical_wm_loaded" in meta.columns:
            needs_anatomical |= ~meta["anatomical_wm_loaded"]
        elif needs_anatomical_wm:
            needs_anatomical |= True

        if needs_anatomical_ct and "anatomical_ct_loaded" in meta.columns:
            needs_anatomical |= ~meta["anatomical_ct_loaded"]
        elif needs_anatomical_ct:
            needs_anatomical |= True

        needs_diffusion_loading = pd.Series([False] * len(meta), index=meta.index)
        if needs_diffusion and "diffusion_loaded" in meta.columns:
            needs_diffusion_loading |= ~meta["diffusion_loaded"]
        elif needs_diffusion:
            needs_diffusion_loading |= True

        # Get sessions for each modality
        anatomical_meta = meta[needs_anatomical].copy()
        diffusion_sessions = meta[needs_diffusion_loading][
            ["subject_code", "session_id"]
        ]

        # For anatomical sessions, determine which specific modalities each needs
        if not anatomical_meta.empty:

            def get_needed_modalities(row):
                """Determine which modalities a session needs to load."""
                needed = []
                if needs_anatomical_gm and not row.get("anatomical_gm_loaded", False):
                    needed.append("gm")
                if needs_anatomical_wm and not row.get("anatomical_wm_loaded", False):
                    needed.append("wm")
                if needs_anatomical_ct and not row.get("anatomical_ct_loaded", False):
                    needed.append("ct")
                return needed if needed else None

            anatomical_meta["_modalities_to_load"] = anatomical_meta.apply(
                get_needed_modalities, axis=1
            )

        anatomical_sessions = anatomical_meta[
            ["subject_code", "session_id", "_modalities_to_load"]
        ]

        # Merge with full sessions CSV to get all columns
        anatomical_sessions = all_sessions.merge(
            anatomical_sessions,
            on=["subject_code", "session_id"],
            how="inner",
        )

        diffusion_sessions = all_sessions.merge(
            diffusion_sessions,
            on=["subject_code", "session_id"],
            how="inner",
        )

        # Count total unique sessions that need any loading
        all_needed = pd.concat(
            [
                anatomical_sessions[["subject_code", "session_id"]],
                diffusion_sessions[["subject_code", "session_id"]],
            ]
        ).drop_duplicates()

        n_skipped = len(all_sessions) - len(all_needed)

        if (
            n_skipped > 0
            or len(anatomical_sessions) < len(all_sessions)
            or len(diffusion_sessions) < len(all_sessions)
        ):
            logger.info(
                f"Incremental mode: {n_skipped} sessions fully loaded, "
                f"{len(anatomical_sessions)} need anatomical, "
                f"{len(diffusion_sessions)} need diffusion"
            )

            # Log per-modality breakdown for anatomical
            if (
                not anatomical_sessions.empty
                and "_modalities_to_load" in anatomical_sessions.columns
            ):
                modality_counts = {}
                for mods in anatomical_sessions["_modalities_to_load"]:
                    if mods:
                        for mod in mods:
                            modality_counts[mod] = modality_counts.get(mod, 0) + 1
                if modality_counts:
                    breakdown = ", ".join(
                        [
                            f"{mod}: {count}"
                            for mod, count in sorted(modality_counts.items())
                        ]
                    )
                    logger.info(f"  Anatomical modalities needed: {breakdown}")

        return all_sessions, anatomical_sessions, diffusion_sessions, n_skipped

    def _write_temp_sessions_csv(self, sessions_df: pd.DataFrame) -> Path:
        """Write a temporary sessions CSV for the loaders."""
        temp_dir = self.config.paths.output_dir / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_csv = temp_dir / "sessions_to_load.csv"
        sessions_df.to_csv(temp_csv, index=False)
        return temp_csv

    def load_anatomical_data(
        self,
        sessions_csv: Optional[Path] = None,
        store: Optional[FeatureStore] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load anatomical data (all modalities combined).

        Args:
            sessions_csv: Path to sessions CSV
            store: FeatureStore for incremental saving (optional)

        Returns:
            Long-format DataFrame with all parcellator columns
        """
        loader = self._init_anatomical_loader()
        if loader is None:
            return None

        csv_path = sessions_csv or self.config.paths.sessions_csv

        # Create callback for incremental saving to SQLite
        def save_session(subject: str, session: str, df: pd.DataFrame) -> None:
            if store is None:
                logger.debug(
                    f"Store is None, skipping cache save for {subject}/{session}"
                )
                return
            if len(df) == 0:
                logger.debug(
                    f"Empty dataframe for {subject}/{session}, skipping cache save"
                )
                return

            # Detect modality from the dataframe
            if "modality" in df.columns:
                modalities = df["modality"].unique()
                logger.debug(
                    f"Saving {subject}/{session} to cache: modalities={modalities}"
                )
                for modality in modalities:
                    mod_df = df[df["modality"] == modality]
                    store.append_anatomical_session(mod_df, subject, session, modality)
            else:
                # If no modality column, save as-is (should not happen)
                logger.warning(
                    f"No modality column in {subject}/{session} - columns: {df.columns.tolist()}"
                )

        logger.info(f"Loading anatomical data (n_jobs={self.config.n_jobs})...")
        try:
            # Use output directory for TIV files (easier to debug than temp)
            tiv_work_dir = self.config.paths.output_dir / ".tiv_work"

            # Load with TIV calculation but WITHOUT normalization
            # This adds 'tiv' column to the data without modifying volume values
            long_df = loader.load_sessions(
                sessions_csv=csv_path,
                n_jobs=self.config.n_jobs,
                progress=self.config.progress,
                normalize_by_tiv=False,  # Don't normalize - store raw values
                calculate_tiv=True,  # But do calculate TIV and add to dataframe
                tiv_output_dir=tiv_work_dir,  # Use known directory for debugging
                session_callback=save_session if store else None,  # Incremental save
            )
        except Exception as e:
            logger.error(f"Failed to load anatomical data: {e}")
            return None

        # Filter to requested modalities
        modalities = self._get_anatomical_modalities()
        if modalities:
            long_df = long_df[long_df["modality"].isin(modalities)]

        logger.info(f"Loaded {len(long_df)} anatomical records")
        return long_df

    def load_diffusion_data(
        self,
        sessions_csv: Optional[Path] = None,
        store: Optional[FeatureStore] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load diffusion data.

        Args:
            sessions_csv: Path to sessions CSV
            store: FeatureStore for incremental saving (optional)

        Returns:
            Long-format DataFrame from DiffusionLoader
        """
        loader = self._init_diffusion_loader()
        if loader is None:
            return None

        csv_path = sessions_csv or self.config.paths.sessions_csv

        # Create callback for incremental saving to SQLite
        def save_session(subject: str, session: str, df: pd.DataFrame) -> None:
            if store is None:
                logger.debug(
                    f"Store is None, skipping cache save for {subject}/{session}"
                )
                return
            if len(df) == 0:
                logger.debug(
                    f"Empty dataframe for {subject}/{session}, skipping cache save"
                )
                return

            # Detect workflow from the dataframe
            if "workflow" in df.columns:
                workflows = df["workflow"].unique()
                logger.debug(
                    f"Saving {subject}/{session} to cache: workflows={workflows}"
                )
                for workflow in workflows:
                    wf_df = df[df["workflow"] == workflow]
                    store.append_diffusion_session(wf_df, subject, session, workflow)
            else:
                # If no workflow column, save all together
                logger.debug(
                    f"Saving {subject}/{session} to cache (no workflow column)"
                )
                store.append_diffusion_session(df, subject, session, workflow=None)

        logger.info(f"Loading diffusion data (n_jobs={self.config.n_jobs})...")
        try:
            long_df = loader.load_sessions(
                sessions_csv=csv_path,
                progress=self.config.progress,
                n_jobs=self.config.n_jobs,
                session_callback=save_session if store else None,  # Incremental save
            )
        except Exception as e:
            logger.error(f"Failed to load diffusion data: {e}")
            return None

        logger.info(f"Loaded {len(long_df)} diffusion records")
        return long_df

    def _extract_tiv(self, anatomical_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Extract TIV from anatomical data if present.

        The anatomical loader adds 'tiv' column when MATLAB/CAT12 is configured.
        """
        if anatomical_df is None or "tiv" not in anatomical_df.columns:
            return None

        # Get unique TIV per session
        tiv_df = (
            anatomical_df[["subject_code", "session_id", "tiv"]]
            .drop_duplicates()
            .dropna(subset=["tiv"])
        )

        if tiv_df.empty:
            return None

        logger.info(f"Extracted TIV for {len(tiv_df)} sessions")
        return tiv_df

    def _prepare_metadata_df(
        self,
        anatomical_df: Optional[pd.DataFrame],
        diffusion_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """Prepare metadata DataFrame with AGE from sessions CSV."""
        session_pairs = []

        if anatomical_df is not None and len(anatomical_df) > 0:
            session_pairs.append(
                anatomical_df[["subject_code", "session_id"]].drop_duplicates()
            )

        if diffusion_df is not None and len(diffusion_df) > 0:
            session_pairs.append(
                diffusion_df[["subject_code", "session_id"]].drop_duplicates()
            )

        if not session_pairs:
            raise ValueError("No data loaded from any modality")

        all_sessions = pd.concat(session_pairs, ignore_index=True).drop_duplicates()

        # Add age from sessions CSV
        sessions = self._load_sessions_csv()
        age_col = self.config.age_column

        age_source_col = None
        for col_name in [age_col, "AGE", "Age@Scan", "age", "Age"]:
            if col_name in sessions.columns:
                age_source_col = col_name
                break

        if age_source_col:
            metadata = all_sessions.merge(
                sessions[
                    ["subject_code", "session_id", age_source_col]
                ].drop_duplicates(),
                on=["subject_code", "session_id"],
                how="left",
            )
            if age_source_col != "AGE":
                metadata = metadata.rename(columns={age_source_col: "AGE"})
        else:
            metadata = all_sessions.copy()
            logger.warning("No age column found in sessions CSV")

        return metadata

    def _compute_metadata(self, store: FeatureStore) -> Dict[str, Any]:
        """Compute metadata about the output."""
        summary = store.summary()

        age_stats = {"min": None, "max": None, "mean": None, "missing": 0}
        if store.metadata_path.exists():
            meta = pd.read_parquet(store.metadata_path)
            if "AGE" in meta.columns:
                age_stats = {
                    "min": (
                        float(meta["AGE"].min())
                        if not meta["AGE"].isna().all()
                        else None
                    ),
                    "max": (
                        float(meta["AGE"].max())
                        if not meta["AGE"].isna().all()
                        else None
                    ),
                    "mean": (
                        float(meta["AGE"].mean())
                        if not meta["AGE"].isna().all()
                        else None
                    ),
                    "missing": int(meta["AGE"].isna().sum()),
                }

        return {
            "n_subjects": summary.get("n_subjects", 0),
            "n_sessions": summary.get("n_sessions", 0),
            "long_formats": summary.get("long_formats", []),
            "n_wide_features": summary.get("n_wide_features", 0),
            "anatomical_features": summary.get("anatomical_features", []),
            "diffusion_features": summary.get("diffusion_features", []),
            "has_tiv": summary.get("has_tiv", False),
            "age_stats": age_stats,
            "atlas_name": self.config.atlas_name,
        }

    def run(self) -> PipelineResult:
        """
        Execute the full data preparation pipeline.

        Steps:
        1. Load anatomical data and save as long format (per modality)
        2. Load diffusion data and save as long format (per workflow)
        3. Extract and save TIV separately
        4. Generate wide-format features from long format
        5. Save metadata

        Returns:
            PipelineResult with store, metadata, and saved formats
        """
        logger.info("Starting data preparation pipeline...")

        self._validate_sessions_csv()

        # Initialize feature store
        store = FeatureStore(
            root_dir=self.config.paths.output_dir,
            compression=self.config.output.compression,
        )

        # Initialize metadata with ALL sessions (tracking load status)
        # Only initialize if metadata doesn't exist or force=True
        all_sessions = self._load_sessions_csv()
        if self.config.force or not store.metadata_path.exists():
            logger.info("Initializing metadata with all sessions")
            store.initialize_metadata(all_sessions, age_col=self.config.age_column)
        else:
            logger.info("Using existing metadata")

        # Determine which sessions to load (separate for anatomical and diffusion)
        _, anatomical_sessions, diffusion_sessions, n_skipped = (
            self._get_sessions_to_load(store)
        )

        if anatomical_sessions.empty and diffusion_sessions.empty:
            logger.info("All sessions already loaded - nothing to do")
            metadata = self._compute_metadata(store)
            return PipelineResult(
                store=store,
                metadata=metadata,
                output_path=self.config.paths.output_dir,
                long_formats_saved=store.list_long_formats(),
                wide_features_generated=store.list_features(),
                n_new_sessions=0,
                n_skipped_sessions=n_skipped,
            )

        # Prepare temp CSVs for each modality if incremental
        temp_anatomical_csv = None
        temp_diffusion_csv = None
        is_incremental = not self.config.force and store.exists()

        if is_incremental:
            if not anatomical_sessions.empty:
                temp_anatomical_csv = self._write_temp_sessions_csv(anatomical_sessions)
            if not diffusion_sessions.empty:
                temp_dir = self.config.paths.output_dir / ".tmp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                temp_diffusion_csv = temp_dir / "sessions_to_load_diffusion.csv"
                diffusion_sessions.to_csv(temp_diffusion_csv, index=False)

        # Calculate total new sessions
        all_needed = (
            pd.concat(
                [
                    anatomical_sessions[["subject_code", "session_id"]],
                    diffusion_sessions[["subject_code", "session_id"]],
                ]
            ).drop_duplicates()
            if not anatomical_sessions.empty or not diffusion_sessions.empty
            else pd.DataFrame()
        )

        n_new_sessions = len(all_needed)

        long_formats_saved = []
        tiv_df = None

        # =====================================================================
        # Load anatomical and diffusion data CONCURRENTLY with incremental saving
        # =====================================================================
        anatomical_df = None
        diffusion_df = None

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            # Only load anatomical if there are sessions that need it
            if self.config.modalities.anatomical and not anatomical_sessions.empty:
                anatomical_csv = temp_anatomical_csv if temp_anatomical_csv else None
                futures["anatomical"] = executor.submit(
                    self.load_anatomical_data,
                    anatomical_csv,
                    store,  # Pass store for incremental saving
                )
            elif self.config.modalities.anatomical:
                logger.info("All anatomical sessions already loaded - skipping")

            # Only load diffusion if there are sessions that need it
            if self.config.modalities.diffusion and not diffusion_sessions.empty:
                diffusion_csv = temp_diffusion_csv if temp_diffusion_csv else None
                futures["diffusion"] = executor.submit(
                    self.load_diffusion_data,
                    diffusion_csv,
                    store,  # Pass store for incremental saving
                )
            elif self.config.modalities.diffusion:
                logger.info("All diffusion sessions already loaded - skipping")

            # Collect results
            for name, future in futures.items():
                try:
                    result = future.result()
                    if name == "anatomical":
                        anatomical_df = result
                    else:
                        diffusion_df = result
                except Exception as e:
                    logger.error(f"Failed to load {name} data: {e}")

        # =====================================================================
        # Export cache to Parquet files
        # =====================================================================
        logger.info("Exporting cached data to Parquet files...")
        long_formats_saved = store.export_cache_to_parquet(
            atlas_name=self.config.atlas_name
        )

        # Extract TIV from anatomical data if available
        if anatomical_df is not None and len(anatomical_df) > 0:
            tiv_df = self._extract_tiv(anatomical_df)

        # =====================================================================
        # Save diffusion data (long format per workflow)
        # =====================================================================

        if diffusion_df is not None and len(diffusion_df) > 0:
            logger.info("Saving diffusion long format data...")

            # Save each workflow separately
            for workflow in diffusion_df["workflow"].unique():
                wf_df = diffusion_df[diffusion_df["workflow"] == workflow].copy()
                name = store.save_diffusion_long(wf_df, workflow=workflow)
                long_formats_saved.append(name)

        # Clean up temp files
        for temp_csv in [temp_anatomical_csv, temp_diffusion_csv]:
            if temp_csv and temp_csv.exists():
                temp_csv.unlink()

        # Clean up temp directory if empty
        temp_dir = self.config.paths.output_dir / ".tmp"
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass  # Directory not empty or other issue

        if not long_formats_saved:
            raise ValueError("No data was saved - check your data paths")

        # =====================================================================
        # Save TIV (as both standalone file and anatomical feature)
        # =====================================================================
        tiv_feature_name = None
        if tiv_df is not None and len(tiv_df) > 0:
            logger.info("Saving TIV data...")
            tiv_feature_name = store.save_tiv(tiv_df)

        # =====================================================================
        # Metadata is already saved incrementally via update_session_load_status()
        # No need to save again here
        # =====================================================================

        # =====================================================================
        # Generate wide-format features
        # =====================================================================
        logger.info("Generating wide-format features...")
        wide_features = store.generate_wide_features()

        # Include TIV in wide features list if it was saved
        if tiv_feature_name and tiv_feature_name not in wide_features:
            wide_features.append(tiv_feature_name)

        # Compute final metadata
        metadata = self._compute_metadata(store)

        # Clear SQLite cache now that everything is exported to Parquet
        store.clear_cache()

        logger.info("Pipeline complete!")

        return PipelineResult(
            store=store,
            metadata=metadata,
            output_path=self.config.paths.output_dir,
            long_formats_saved=long_formats_saved,
            wide_features_generated=wide_features,
            n_new_sessions=n_new_sessions,
            n_skipped_sessions=n_skipped,
        )

    @staticmethod
    def sanitize_subject_code(subject_code: str) -> str:
        """Sanitize subject code by removing problematic characters."""
        return (
            str(subject_code)
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
        ).zfill(4)

    @staticmethod
    def sanitize_session_id(session_id: str) -> str:
        """Sanitize session ID by removing problematic characters."""
        if pd.isna(session_id):
            return
        return (
            str(int(float(session_id)))
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
        )
