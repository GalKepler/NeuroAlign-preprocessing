# NeuroAlign Preprocessing

**Neuroimaging data preprocessing pipeline for NeuroAlign**

This repository contains the data loading and preprocessing components for the [NeuroAlign](https://github.com/yourusername/neuroalign) framework. It handles:

- Loading anatomical MRI data (gray matter, white matter, cortical thickness)
- Loading diffusion MRI data (NODDI, DKI, MAPMRI, GQI metrics)
- Feature extraction from parcellated brain regions
- Statistical aggregation and quality control
- Standardized output format for downstream analysis

## Why a Separate Repository?

The preprocessing pipeline is maintained separately from the main NeuroAlign analysis framework because:

1. **Different dependencies**: Preprocessing relies on neuroimaging tools (nibabel, dipy) while analysis uses ML libraries
2. **Different update cycles**: Preprocessing is more stable; analysis evolves more rapidly
3. **Modularity**: Users can use their own preprocessing or start with preprocessed data
4. **Clarity**: Each repository has a single, clear purpose

## Installation

### Using uv (recommended)

```bash
uv pip install neuroalign-preprocessing
```

### Using pip

```bash
pip install neuroalign-preprocessing
```

### From source

```bash
git clone https://github.com/yourusername/neuroalign-preprocessing
cd neuroalign-preprocessing
uv pip install -e .
```

## Quick Start

### Command Line

```bash
# Preprocess a BIDS dataset
neuroalign-preprocess \
    --input /path/to/bids/dataset \
    --output /path/to/output \
    --parcellation Schaefer400_7Networks

# With custom configuration
neuroalign-preprocess \
    --input /path/to/bids/dataset \
    --output /path/to/output \
    --config config.yaml
```

### Python API

```python
from neuroalign_preprocessing import PreprocessingPipeline
from neuroalign_preprocessing.config import Config

# Configure pipeline
config = Config(
    parcellation="Schaefer400_7Networks",
    anatomical_metrics=["gm", "wm", "ct"],
    diffusion_pipelines=["AMICONODDI", "DIPYDKI"],
    statistics=["mean", "std", "median"]
)

# Create and run pipeline
pipeline = PreprocessingPipeline(config)
pipeline.run(
    input_dir="/path/to/bids/dataset",
    output_dir="/path/to/output"
)
```

### Load Existing Preprocessed Data

```python
from neuroalign_preprocessing.loaders import AnatomicalLoader, DiffusionLoader

# Load anatomical features
anat_loader = AnatomicalLoader("/path/to/preprocessed/features/anatomical")
X_gm = anat_loader.load_metric("gm", statistic="mean", format="wide")

# Load diffusion features
diff_loader = DiffusionLoader("/path/to/preprocessed/features/diffusion")
X_noddi = diff_loader.load_pipeline("AMICONODDI", metric="icvf", statistic="mean")
```

## Output Format

The preprocessing pipeline generates a standardized output format that can be directly consumed by [NeuroAlign](https://github.com/yourusername/neuroalign):

```
output_dir/
├── dataset_description.json
├── participants.tsv
└── derivatives/
    └── neuroalign-preprocessing/
        ├── pipeline_description.json
        ├── features/
        │   ├── anatomical/
        │   │   ├── wide/
        │   │   └── long/
        │   ├── diffusion/
        │   │   ├── wide/
        │   │   └── long/
        │   └── metadata.parquet
        └── manifest.json
```

See [OUTPUT_SCHEMA.md](docs/OUTPUT_SCHEMA.md) for complete specification.

## Features

### Anatomical Data
- **Metrics**: Gray matter (GM), white matter (WM), cortical thickness (CT)
- **Statistics**: mean, std, median, MAD median, robust mean, robust std
- **Formats**: Wide (subjects × features) and long (tidy) formats

### Diffusion Data
- **Pipelines**:
  - AMICO-NODDI: ICVF, ISOVF, OD
  - DIPY DKI: FA, MD, AD, RD, MK, AK, RK, kurtosis metrics
  - DIPY MAPMRI: RTOP, RTAP, RTPP, QIV, MSD
  - DSI Studio GQI: QA, GFA, ISO
- **Statistics**: Same as anatomical + IQR filtering, z-score filtering
- **Formats**: Wide and long

### Quality Control
- Outlier detection (IQR-based, z-score-based)
- Missing data handling
- QC metrics per subject
- Validation against output schema

## Configuration

Example configuration file (`config.yaml`):

```yaml
parcellation: Schaefer400_7Networks

anatomical:
  metrics: [gm, wm, ct]
  statistics: [mean, std, median, mad_median, robust_mean, robust_std]

diffusion:
  pipelines: [AMICONODDI, DIPYDKI, DIPYMAPMRI, DSIStudio]
  statistics: [mean, std, median, mad_median, robust_mean, robust_std, 
               iqr_filtered_mean, iqr_filtered_std, z_filtered_mean, z_filtered_std]

quality_control:
  outlier_method: IQR
  outlier_threshold: 3.0
  min_valid_parcels: 0.8

output:
  formats: [wide, long]
  compression: snappy
```

## Integration with NeuroAlign

After preprocessing, use the output with [NeuroAlign](https://github.com/yourusername/neuroalign):

```python
from neuroalign.data import PreprocessedDataset
from neuroalign.modeling import BrainAgeModel

# Load preprocessed data
dataset = PreprocessedDataset("/path/to/output/derivatives/neuroalign-preprocessing")

# Extract features
X = dataset.get_features("anatomical", metric="gm", statistic="mean")
metadata = dataset.get_metadata()

# Run NeuroAlign analysis
model = BrainAgeModel()
predictions = model.fit_predict(X, metadata["age"])
```

## Documentation

- [Installation Guide](docs/INSTALLATION.md)
- [Usage Guide](docs/USAGE.md)
- [Output Schema](docs/OUTPUT_SCHEMA.md)
- [API Reference](docs/API.md)

## Development

### Setup Development Environment

```bash
git clone https://github.com/yourusername/neuroalign-preprocessing
cd neuroalign-preprocessing
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
ruff format .

# Lint
ruff check .

# Type checking
mypy src/
```

## Citation

If you use this preprocessing pipeline, please cite:

```bibtex
@software{neuroalign_preprocessing,
  title = {NeuroAlign Preprocessing},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/neuroalign-preprocessing}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Related Projects

- [NeuroAlign](https://github.com/yourusername/neuroalign) - Main analysis framework
- [BIDS](https://bids.neuroimaging.io/) - Brain Imaging Data Structure

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/neuroalign-preprocessing/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/neuroalign-preprocessing/discussions)

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
