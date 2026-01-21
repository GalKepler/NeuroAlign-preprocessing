"""Output writer for standardized format."""

import json
from pathlib import Path
import pandas as pd


class OutputWriter:
    """Write preprocessing outputs in standardized format."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.features_dir = self.output_dir / "features"
        
    def write_features(
        self,
        data: pd.DataFrame,
        modality: str,
        metric: str,
        statistic: str,
        format: str = "wide"
    ):
        """Write feature data to parquet file."""
        output_path = self.features_dir / modality / format / f"{metric}_{statistic}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(output_path, engine="pyarrow", compression="snappy")
        
    def write_manifest(self, manifest: dict):
        """Write manifest.json file."""
        with open(self.output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
