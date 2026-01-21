# NeuroAlign Preprocessing Output Schema

This document defines the standardized output format from `neuroalign-preprocessing` that serves as input to `neuroalign`.

## Directory Structure

```
<output_dir>/
├── dataset_description.json              # Dataset-level metadata
├── participants.tsv                      # Participant demographics (BIDS-style)
└── derivatives/
    └── neuroalign-preprocessing/
        ├── pipeline_description.json     # Processing pipeline metadata
        ├── features/
        │   ├── anatomical/
        │   │   ├── wide/                 # Wide format features
        │   │   │   ├── gm_mean.parquet
        │   │   │   ├── gm_std.parquet
        │   │   │   ├── wm_mean.parquet
        │   │   │   ├── ct_mean.parquet
        │   │   │   └── tiv.parquet
        │   │   └── long/                 # Long format features
        │   │       ├── anatomical_gm.parquet
        │   │       ├── anatomical_wm.parquet
        │   │       └── anatomical_ct.parquet
        │   ├── diffusion/
        │   │   ├── wide/
        │   │   │   ├── AMICONODDI_noddi_icvf_mean.parquet
        │   │   │   ├── DIPYDKI_dki_fa_mean.parquet
        │   │   │   └── ... (all diffusion metrics)
        │   │   └── long/
        │   │       ├── AMICONODDI.parquet
        │   │       ├── DIPYDKI.parquet
        │   │       ├── DIPYMAPMRI.parquet
        │   │       └── DSIStudio.parquet
        │   └── metadata.parquet              # Subject-level metadata
        └── manifest.json                     # Index of all output files
```

## File Specifications

### 1. dataset_description.json

```json
{
  "Name": "Dataset Name",
  "BIDSVersion": "1.6.0",
  "DatasetType": "derivative",
  "GeneratedBy": [
    {
      "Name": "neuroalign-preprocessing",
      "Version": "0.1.0",
      "CodeURL": "https://github.com/username/neuroalign-preprocessing"
    }
  ],
  "SourceDatasets": [
    {
      "URL": "path/to/source/dataset",
      "Version": "1.0"
    }
  ]
}
```

### 2. pipeline_description.json

```json
{
  "Name": "neuroalign-preprocessing",
  "Version": "0.1.0",
  "Description": "Preprocessing pipeline for NeuroAlign",
  "ProcessingSteps": [
    {
      "Step": "Data Loading",
      "Description": "Load anatomical and diffusion data",
      "Tools": ["nibabel", "dipy"]
    },
    {
      "Step": "Feature Extraction",
      "Description": "Extract regional statistics from parcellations",
      "Parcellation": "Schaefer 400 parcel 7 networks"
    },
    {
      "Step": "Feature Aggregation",
      "Description": "Compute summary statistics per region",
      "Statistics": ["mean", "std", "median", "mad_median", "robust_mean", "robust_std"]
    }
  ],
  "ProcessingDate": "2025-01-21T12:00:00Z",
  "Config": {
    "anatomical_metrics": ["gm", "wm", "ct"],
    "diffusion_pipelines": ["AMICONODDI", "DIPYDKI", "DIPYMAPMRI", "DSIStudio"],
    "parcellation": "Schaefer400_7Networks",
    "outlier_detection": {
      "method": "IQR",
      "threshold": 3.0
    }
  }
}
```

### 3. participants.tsv

BIDS-compatible TSV file:

```
participant_id	age	sex	site	group	...
sub-001	25.3	M	SiteA	control	...
sub-002	32.1	F	SiteB	patient	...
```

### 4. manifest.json

Index of all generated files with metadata:

```json
{
  "version": "1.0",
  "created": "2025-01-21T12:00:00Z",
  "n_subjects": 150,
  "features": {
    "anatomical": {
      "wide": [
        {
          "file": "features/anatomical/wide/gm_mean.parquet",
          "metric": "gm",
          "statistic": "mean",
          "n_features": 400,
          "description": "Mean gray matter intensity per parcel"
        }
      ],
      "long": [
        {
          "file": "features/anatomical/long/anatomical_gm.parquet",
          "metric": "gm",
          "format": "long",
          "description": "Long format gray matter features"
        }
      ]
    },
    "diffusion": {
      "wide": [...],
      "long": [...]
    }
  },
  "metadata": {
    "file": "features/metadata.parquet",
    "n_subjects": 150,
    "columns": ["participant_id", "age", "sex", "site", "tiv", ...]
  }
}
```

### 5. Parquet File Specifications

#### Wide Format Features
- **Index**: `participant_id` (e.g., "sub-001")
- **Columns**: Feature names (e.g., "parcel_001", "parcel_002", ..., "parcel_400")
- **Example**: `gm_mean.parquet`

```python
import pandas as pd
df = pd.read_parquet("features/anatomical/wide/gm_mean.parquet")
# Index: ['sub-001', 'sub-002', ...]
# Columns: ['parcel_001', 'parcel_002', ..., 'parcel_400']
```

#### Long Format Features
- **Columns**: `participant_id`, `parcel`, `metric`, `value`, `statistic`
- **Example**: `anatomical_gm.parquet`

```python
df = pd.read_parquet("features/anatomical/long/anatomical_gm.parquet")
# Columns: ['participant_id', 'parcel', 'metric', 'value', 'statistic']
# Example row: ['sub-001', 'parcel_001', 'gm', 0.45, 'mean']
```

#### Metadata
- **Index**: `participant_id`
- **Columns**: Demographics + derived features (TIV, QC metrics, etc.)

```python
df = pd.read_parquet("features/metadata.parquet")
# Index: ['sub-001', 'sub-002', ...]
# Columns: ['age', 'sex', 'site', 'tiv', 'qc_pass', ...]
```

## Data Schema Validation

The output can be validated using:

```python
from neuroalign_preprocessing.io import validate_output

# Validate entire output directory
validate_output("/path/to/output")

# Returns validation report:
{
  "valid": True,
  "errors": [],
  "warnings": ["Missing optional field: ..."],
  "summary": {
    "n_subjects": 150,
    "n_anatomical_features": 1200,
    "n_diffusion_features": 5000
  }
}
```

## Usage in NeuroAlign

NeuroAlign loads preprocessed data as:

```python
from neuroalign.data import PreprocessedDataset

# Load preprocessed data
dataset = PreprocessedDataset("/path/to/output/derivatives/neuroalign-preprocessing")

# Access features
X_anat = dataset.get_features("anatomical", format="wide")
X_diff = dataset.get_features("diffusion", format="wide")
metadata = dataset.get_metadata()

# Start NeuroAlign analysis
# ... (regional BAG estimation, alignment, etc.)
```

## Version Compatibility

- **Schema Version**: 1.0
- **Compatible NeuroAlign Versions**: >= 0.1.0
- **BIDS Compatibility**: Follows BIDS Derivatives specification

## Notes

1. All `.parquet` files should use PyArrow engine for compatibility
2. Participant IDs must be consistent across all files
3. Missing data should be represented as `NaN`
4. All file paths are relative to the output directory root
5. Feature names should be stable across versions (breaking changes require major version bump)
