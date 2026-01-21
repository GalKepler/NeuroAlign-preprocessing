"""
Adapter for loading preprocessed data from neuroalign-preprocessing.

This module provides a clean interface for NeuroAlign to load
preprocessed data in the standardized format.
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

import pandas as pd


class PreprocessedDataset:
    """
    Load and access preprocessed neuroimaging data.
    
    This class provides a unified interface to data preprocessed by
    the neuroalign-preprocessing pipeline.
    
    Parameters
    ----------
    data_dir : str or Path
        Path to the derivatives/neuroalign-preprocessing directory
        
    Examples
    --------
    >>> dataset = PreprocessedDataset("/path/to/output/derivatives/neuroalign-preprocessing")
    >>> X_anat = dataset.get_features("anatomical", metric="gm", statistic="mean")
    >>> metadata = dataset.get_metadata()
    """
    
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir)
        self.features_dir = self.data_dir / "features"
        
        # Load manifest
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                self.manifest = json.load(f)
        else:
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
        # Load pipeline description
        pipeline_path = self.data_dir / "pipeline_description.json"
        if pipeline_path.exists():
            with open(pipeline_path) as f:
                self.pipeline_info = json.load(f)
        else:
            self.pipeline_info = {}
    
    def get_metadata(self) -> pd.DataFrame:
        """
        Load subject metadata.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with subjects as index and metadata columns (age, sex, site, etc.)
        """
        metadata_path = self.features_dir / "metadata.parquet"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        return pd.read_parquet(metadata_path)
    
    def get_features(
        self,
        modality: Literal["anatomical", "diffusion"],
        metric: Optional[str] = None,
        statistic: Optional[str] = None,
        format: Literal["wide", "long"] = "wide",
        pipeline: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load features for a specific modality.
        
        Parameters
        ----------
        modality : {"anatomical", "diffusion"}
            Which imaging modality to load
        metric : str, optional
            Specific metric to load (e.g., "gm", "wm", "ct" for anatomical;
            "fa", "md" for diffusion). If None, loads all metrics.
        statistic : str, optional
            Statistic to load (e.g., "mean", "std"). If None, loads all statistics.
        format : {"wide", "long"}, default="wide"
            Format of the data
        pipeline : str, optional
            For diffusion data, specify pipeline (e.g., "DIPYDKI", "AMICONODDI")
            
        Returns
        -------
        pd.DataFrame
            Feature data
            
        Examples
        --------
        >>> # Load gray matter mean values (wide format)
        >>> gm_mean = dataset.get_features("anatomical", metric="gm", statistic="mean")
        >>> 
        >>> # Load all anatomical features
        >>> all_anat = dataset.get_features("anatomical")
        >>> 
        >>> # Load diffusion FA from DKI pipeline
        >>> dki_fa = dataset.get_features("diffusion", pipeline="DIPYDKI", metric="fa", statistic="mean")
        """
        modality_dir = self.features_dir / modality / format
        
        if not modality_dir.exists():
            raise FileNotFoundError(f"Modality directory not found: {modality_dir}")
        
        if format == "wide" and metric and statistic:
            # Load specific metric/statistic combination
            if modality == "diffusion" and pipeline:
                filename = f"{pipeline}_{metric}_{statistic}.parquet"
            else:
                filename = f"{metric}_{statistic}.parquet"
            
            file_path = modality_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Feature file not found: {file_path}")
            
            return pd.read_parquet(file_path)
        
        elif format == "long":
            # Load long format data
            if modality == "diffusion" and pipeline:
                filename = f"{pipeline}.parquet"
            else:
                filename = f"{modality}_{metric}.parquet"
            
            file_path = modality_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(f"Feature file not found: {file_path}")
            
            df = pd.read_parquet(file_path)
            
            # Filter by metric and/or statistic if specified
            if metric:
                df = df[df["metric"] == metric]
            if statistic:
                df = df[df["statistic"] == statistic]
            
            return df
        
        else:
            # Load all features for the modality
            all_files = list(modality_dir.glob("*.parquet"))
            if not all_files:
                raise FileNotFoundError(f"No feature files found in {modality_dir}")
            
            # Load and concatenate all parquet files
            dfs = []
            for file_path in all_files:
                df = pd.read_parquet(file_path)
                # Add column prefix to distinguish metrics
                df.columns = [f"{file_path.stem}_{col}" for col in df.columns]
                dfs.append(df)
            
            return pd.concat(dfs, axis=1)
    
    def get_available_metrics(
        self,
        modality: Literal["anatomical", "diffusion"]
    ) -> Dict[str, List[str]]:
        """
        Get available metrics and statistics for a modality.
        
        Parameters
        ----------
        modality : {"anatomical", "diffusion"}
            Which modality to query
            
        Returns
        -------
        dict
            Dictionary with metrics as keys and lists of available statistics as values
        """
        modality_features = self.manifest.get("features", {}).get(modality, {})
        
        metrics = {}
        for format_type in ["wide", "long"]:
            for feature in modality_features.get(format_type, []):
                metric = feature.get("metric")
                statistic = feature.get("statistic")
                
                if metric:
                    if metric not in metrics:
                        metrics[metric] = []
                    if statistic and statistic not in metrics[metric]:
                        metrics[metric].append(statistic)
        
        return metrics
    
    def get_subject_ids(self) -> List[str]:
        """
        Get list of subject IDs in the dataset.
        
        Returns
        -------
        list
            List of subject identifiers
        """
        metadata = self.get_metadata()
        return metadata.index.tolist()
    
    def get_n_subjects(self) -> int:
        """
        Get number of subjects in the dataset.
        
        Returns
        -------
        int
            Number of subjects
        """
        return self.manifest.get("n_subjects", len(self.get_subject_ids()))
    
    def get_pipeline_info(self) -> Dict:
        """
        Get information about the preprocessing pipeline.
        
        Returns
        -------
        dict
            Pipeline metadata including version, configuration, etc.
        """
        return self.pipeline_info
    
    def __repr__(self) -> str:
        n_subjects = self.get_n_subjects()
        return (
            f"PreprocessedDataset("
            f"n_subjects={n_subjects}, "
            f"data_dir='{self.data_dir}')"
        )


class FeatureSelector:
    """
    Helper class for selecting and combining features from preprocessed data.
    
    Examples
    --------
    >>> dataset = PreprocessedDataset("/path/to/data")
    >>> selector = FeatureSelector(dataset)
    >>> 
    >>> # Select specific anatomical features
    >>> X = selector.select(
    ...     anatomical=["gm_mean", "wm_mean", "ct_mean"],
    ...     diffusion=["DIPYDKI_dki_fa_mean", "DIPYDKI_dki_md_mean"]
    ... )
    """
    
    def __init__(self, dataset: PreprocessedDataset):
        self.dataset = dataset
    
    def select(
        self,
        anatomical: Optional[List[str]] = None,
        diffusion: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Select and combine specific features.
        
        Parameters
        ----------
        anatomical : list of str, optional
            List of anatomical features in format "metric_statistic"
        diffusion : list of str, optional
            List of diffusion features in format "pipeline_metric_statistic"
            
        Returns
        -------
        pd.DataFrame
            Combined feature matrix
        """
        dfs = []
        
        if anatomical:
            for feature in anatomical:
                metric, statistic = feature.split("_", 1)
                df = self.dataset.get_features(
                    "anatomical",
                    metric=metric,
                    statistic=statistic
                )
                dfs.append(df)
        
        if diffusion:
            for feature in diffusion:
                parts = feature.split("_")
                pipeline = parts[0]
                metric = "_".join(parts[1:-1])
                statistic = parts[-1]
                
                df = self.dataset.get_features(
                    "diffusion",
                    pipeline=pipeline,
                    metric=metric,
                    statistic=statistic
                )
                dfs.append(df)
        
        if not dfs:
            raise ValueError("Must specify at least one feature")
        
        return pd.concat(dfs, axis=1)
