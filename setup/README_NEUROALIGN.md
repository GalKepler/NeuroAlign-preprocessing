# NeuroAlign

**Advanced brain age estimation and alignment framework**

NeuroAlign is a framework for estimating regional brain age gaps (BAG) and analyzing inter-individual alignment patterns in neuroimaging data.

## Overview

NeuroAlign focuses on:
- Regional brain age gap (BAG) estimation
- Cross-individual brain alignment analysis
- Embedding-based retrieval of similar brain patterns
- Agent-based interactive analysis

## Installation

### Using uv (recommended)

```bash
uv pip install neuroalign
```

### Using pip

```bash
pip install neuroalign
```

### From source

```bash
git clone https://github.com/yourusername/neuroalign
cd neuroalign
uv pip install -e .
```

## Quick Start

### Prerequisites

NeuroAlign works with preprocessed neuroimaging data. If you have raw BIDS data, first use [neuroalign-preprocessing](https://github.com/yourusername/neuroalign-preprocessing):

```bash
# Install preprocessing pipeline
uv pip install neuroalign-preprocessing

# Preprocess your data
neuroalign-preprocess \
    --input /path/to/bids/dataset \
    --output /path/to/preprocessed \
    --parcellation Schaefer400_7Networks
```

### Load Preprocessed Data

```python
from neuroalign.data import PreprocessedDataset

# Load preprocessed data
dataset = PreprocessedDataset("/path/to/preprocessed/derivatives/neuroalign-preprocessing")

# Get features and metadata
X_anat = dataset.get_features("anatomical", format="wide")
X_diff = dataset.get_features("diffusion", format="wide")
metadata = dataset.get_metadata()
```

### Regional Brain Age Estimation

```python
from neuroalign.modeling import RegionalBrainAge

# Train regional brain age model
model = RegionalBrainAge(
    parcellation="Schaefer400_7Networks",
    bias_correction=True
)

# Fit and predict
model.fit(X_anat, metadata["age"])
regional_bag = model.predict_regional_gaps(X_anat, metadata["age"])

# Results shape: (n_subjects, n_parcels)
print(f"Regional BAG shape: {regional_bag.shape}")
```

### Brain Alignment Analysis

```python
from neuroalign.alignment import AlignmentAnalysis

# Analyze alignment between individuals
alignment = AlignmentAnalysis()
alignment_scores = alignment.compute_pairwise(regional_bag)

# Find most similar individuals
similar_pairs = alignment.find_similar_patterns(
    regional_bag,
    metadata,
    top_k=10
)
```

### Embedding and Retrieval

```python
from neuroalign.embedding import BrainEmbedding
from neuroalign.retrieval import BrainRetrieval

# Create embeddings from regional BAG patterns
embedder = BrainEmbedding(method="autoencoder", latent_dim=64)
embeddings = embedder.fit_transform(regional_bag)

# Query-based retrieval
retrieval = BrainRetrieval(embeddings, metadata)
similar = retrieval.find_similar(
    query_subject="sub-001",
    n_results=5,
    filters={"age_range": (25, 35)}
)
```

### Interactive Agent

```python
from neuroalign.agent import NeuroAlignAgent

# Start interactive analysis
agent = NeuroAlignAgent(dataset)
agent.chat()

# Example queries:
# > "Show me subjects with accelerated aging in frontal regions"
# > "Compare alignment patterns between males and females"
# > "Find subjects similar to sub-001 in terms of regional BAG"
```

## Features

### Regional Brain Age
- Bias-corrected brain age estimation (de Lange & Cole method)
- Regional (parcel-level) brain age gaps
- Anatomical and diffusion-based models
- Cross-validated predictions

### Alignment Analysis
- Pairwise alignment scoring
- Pattern similarity metrics
- Group-level alignment maps
- Clinical interpretation tools

### Embedding & Retrieval
- Dimensionality reduction (PCA, autoencoders, VAE)
- Similarity search
- Metadata-aware filtering
- Efficient nearest neighbor retrieval

### Visualization
- Regional BAG heatmaps
- Alignment matrices
- Interactive brain surface plots
- Statistical summaries

## Architecture

NeuroAlign is designed to work with preprocessed data from [neuroalign-preprocessing](https://github.com/yourusername/neuroalign-preprocessing):

```
Raw Data (BIDS) 
    ↓
[neuroalign-preprocessing]
    ↓
Preprocessed Features
    ↓
[NeuroAlign] ← YOU ARE HERE
    ↓
Analysis Results
```

### Why Separate Repositories?

- **NeuroAlign-Preprocessing**: Handles raw data → standardized features
  - Dependencies: nibabel, dipy, neuroimaging tools
  - Slower update cycle (stable preprocessing)

- **NeuroAlign**: Handles features → insights
  - Dependencies: scikit-learn, PyTorch, analysis tools  
  - Faster iteration (evolving methods)

## Data Format

NeuroAlign expects data in the format produced by neuroalign-preprocessing. If you have your own preprocessing pipeline, ensure your output matches this schema:

```python
# Wide format: subjects × features
X = pd.DataFrame(index=subject_ids, columns=feature_names)

# Metadata: subjects × covariates
metadata = pd.DataFrame(index=subject_ids, columns=["age", "sex", "site", ...])
```

See the [preprocessing output schema](https://github.com/yourusername/neuroalign-preprocessing/blob/main/docs/OUTPUT_SCHEMA.md) for details.

## Examples

See the [examples/](examples/) directory for complete workflows:

- `01_basic_brain_age.py` - Simple regional brain age estimation
- `02_alignment_analysis.py` - Inter-individual alignment
- `03_embedding_retrieval.py` - Similarity search
- `04_clinical_interpretation.py` - Relating BAG to clinical outcomes

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API.md)
- [Tutorials](docs/tutorials/)
- [FAQ](docs/FAQ.md)

## Development

### Setup

```bash
git clone https://github.com/yourusername/neuroalign
cd neuroalign
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Testing

```bash
pytest tests/
```

### Code Quality

```bash
ruff format .
ruff check .
mypy src/
```

## Citation

```bibtex
@article{neuroalign2025,
  title = {NeuroAlign: Regional Brain Age Estimation and Alignment Analysis},
  author = {Your Name},
  journal = {Journal Name},
  year = {2025}
}
```

## Related Projects

- [neuroalign-preprocessing](https://github.com/yourusername/neuroalign-preprocessing) - Data preprocessing pipeline
- [BIDS](https://bids.neuroimaging.io/) - Brain Imaging Data Structure

## License

MIT License - see [LICENSE](LICENSE)

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/neuroalign/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/neuroalign/discussions)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
