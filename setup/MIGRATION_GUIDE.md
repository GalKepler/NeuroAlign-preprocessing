# Migration Guide: Splitting NeuroAlign into Two Repositories

This guide walks you through splitting NeuroAlign into two repositories:
1. **neuroalign-preprocessing** - Data loading and preprocessing
2. **neuroalign** - Analysis and modeling (regional BAG, alignment, etc.)

## Overview

### Why Split?

- **Separation of concerns**: Preprocessing vs. analysis
- **Different dependencies**: Neuroimaging tools vs. ML libraries  
- **Different update cycles**: Stable preprocessing vs. evolving analysis
- **Modularity**: Users can bring their own preprocessed data

### Architecture

```
Raw BIDS Data
     ↓
neuroalign-preprocessing (NEW)
  - Data loaders (anatomical, diffusion)
  - Feature extraction
  - Statistical aggregation
  - Standardized output format
     ↓
Preprocessed Features (defined schema)
     ↓
neuroalign (UPDATED)
  - Regional brain age estimation
  - Alignment analysis
  - Embedding & retrieval
  - Visualization
     ↓
Analysis Results
```

## Step-by-Step Migration

### Step 1: Create the Preprocessing Repository

```bash
# Create new directory
cd ~/projects  # or wherever you keep your projects
mkdir neuroalign-preprocessing
cd neuroalign-preprocessing

# Run migration script
python /path/to/migrate_to_preprocessing.py \
    --source ~/projects/neuroalign \
    --target ~/projects/neuroalign-preprocessing
```

This script will:
- Create directory structure
- Copy loader modules (anatomical, diffusion, questionnaire)
- Copy preprocessing modules (pipeline, feature_store, transformers, config, cli)
- Copy relevant scripts (cat12_tiv_template.m)
- Create pyproject.toml, .gitignore, README
- Create placeholder modules for new functionality

### Step 2: Update Imports in Preprocessing Repo

The migrated code will have old imports. Update them:

```bash
cd ~/projects/neuroalign-preprocessing

# Find and replace imports
find src -name "*.py" -type f -exec sed -i 's/from neuroalign\.data\.loaders/from neuroalign_preprocessing.loaders/g' {} \;
find src -name "*.py" -type f -exec sed -i 's/from neuroalign\.data\.preprocessing/from neuroalign_preprocessing.preprocessing/g' {} \;
find src -name "*.py" -type f -exec sed -i 's/from neuroalign\.data/from neuroalign_preprocessing/g' {} \;
find src -name "*.py" -type f -exec sed -i 's/import neuroalign\.data/import neuroalign_preprocessing/g' {} \;
```

### Step 3: Add Output Writer to Preprocessing

The preprocessing repo needs to output data in a standardized format. Add this functionality:

```bash
cd ~/projects/neuroalign-preprocessing

# The migration script created a placeholder
# Now you need to integrate it with your existing pipeline
```

Update `src/neuroalign_preprocessing/preprocessing/pipeline.py` to use the OutputWriter:

```python
from neuroalign_preprocessing.io.writer import OutputWriter

class PreprocessingPipeline:
    def __init__(self, config, output_dir):
        self.config = config
        self.writer = OutputWriter(output_dir)
    
    def process_subject(self, subject_id, data):
        # ... existing processing code ...
        
        # Write output in standardized format
        self.writer.write_features(
            data=processed_data,
            modality="anatomical",
            metric="gm",
            statistic="mean",
            format="wide"
        )
```

### Step 4: Initialize Git and Push

```bash
cd ~/projects/neuroalign-preprocessing

# Initialize git
git init
git add .
git commit -m "Initial commit: Extract preprocessing from NeuroAlign"

# Create GitHub repository (using gh CLI)
gh repo create neuroalign-preprocessing --public --source=. --remote=origin

# Or manually create on GitHub and add remote
git remote add origin https://github.com/yourusername/neuroalign-preprocessing.git

# Push
git push -u origin main
```

### Step 5: Test the Preprocessing Repo

```bash
cd ~/projects/neuroalign-preprocessing

# Create virtual environment
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
uv pip install -e .

# Run tests (you'll need to create some basic tests first)
pytest tests/

# Test basic import
python -c "from neuroalign_preprocessing.loaders import AnatomicalLoader; print('✓ Import successful')"
```

### Step 6: Update Main NeuroAlign Repository

Now update the original NeuroAlign repository to remove preprocessing code and use the new package.

#### 6a. Remove Preprocessing Code

```bash
cd ~/projects/neuroalign

# Create a new branch for this refactor
git checkout -b refactor/split-preprocessing

# Remove preprocessing code
rm -rf src/neuroalign/data/loaders
rm -rf src/neuroalign/data/preprocessing
rm -f scripts/cat12_tiv_template.m
# Keep src/neuroalign/data/__init__.py but remove preprocessing imports
```

#### 6b. Add Preprocessing as Dependency

Update `pyproject.toml`:

```toml
[project]
name = "neuroalign"
# ... other fields ...
dependencies = [
    "neuroalign-preprocessing @ git+https://github.com/yourusername/neuroalign-preprocessing.git",
    # ... other dependencies ...
]
```

Or after publishing to PyPI:

```toml
dependencies = [
    "neuroalign-preprocessing>=0.1.0",
    # ... other dependencies ...
]
```

#### 6c. Add Data Loading Adapter

Copy the adapter into your NeuroAlign repo:

```bash
cp /path/to/preprocessed_dataset_adapter.py src/neuroalign/data/dataset.py
```

Update `src/neuroalign/data/__init__.py`:

```python
"""Data loading and management for NeuroAlign."""

from neuroalign.data.dataset import PreprocessedDataset, FeatureSelector

__all__ = ["PreprocessedDataset", "FeatureSelector"]
```

#### 6d. Update README

Replace the README with the new version:

```bash
cp /path/to/README_NEUROALIGN.md README.md
```

Edit to add your actual GitHub username and any project-specific details.

### Step 7: Update Existing Code to Use New Structure

Find code that used old loaders and update it:

**Before:**
```python
from neuroalign.data.loaders import AnatomicalLoader
from neuroalign.data.preprocessing import Pipeline

loader = AnatomicalLoader(data_dir)
data = loader.load()
```

**After:**
```python
from neuroalign.data import PreprocessedDataset

# Assume data is already preprocessed using neuroalign-preprocessing
dataset = PreprocessedDataset(preprocessed_dir)
data = dataset.get_features("anatomical", metric="gm", statistic="mean")
```

### Step 8: Test the Updated NeuroAlign

```bash
cd ~/projects/neuroalign

# Install with new dependencies
uv pip install -e .

# Test imports
python -c "from neuroalign.data import PreprocessedDataset; print('✓ Import successful')"

# Run your existing tests
pytest tests/
```

### Step 9: Create Example Workflow

Create an example showing the full workflow:

```bash
cd ~/projects/neuroalign
mkdir -p examples
```

Create `examples/full_workflow.py`:

```python
"""
Example: Complete workflow from raw data to analysis.
"""

# Step 1: Preprocess data (using neuroalign-preprocessing)
from neuroalign_preprocessing import PreprocessingPipeline
from neuroalign_preprocessing.config import Config

config = Config(
    parcellation="Schaefer400_7Networks",
    anatomical_metrics=["gm", "wm", "ct"],
)

pipeline = PreprocessingPipeline(config)
pipeline.run(
    input_dir="/path/to/bids/dataset",
    output_dir="/path/to/preprocessed"
)

# Step 2: Load preprocessed data (using neuroalign)
from neuroalign.data import PreprocessedDataset

dataset = PreprocessedDataset(
    "/path/to/preprocessed/derivatives/neuroalign-preprocessing"
)

# Step 3: Run analysis (using neuroalign)
from neuroalign.modeling import RegionalBrainAge

X = dataset.get_features("anatomical", metric="gm", statistic="mean")
metadata = dataset.get_metadata()

model = RegionalBrainAge()
regional_bag = model.fit_predict(X, metadata["age"])

print(f"Regional BAG shape: {regional_bag.shape}")
```

### Step 10: Update Documentation

Update all documentation to reflect the new structure:

1. **Main README**: Reference preprocessing repo
2. **Installation docs**: Two-step process
3. **Tutorials**: Update to use new data loading
4. **API docs**: Remove preprocessing from NeuroAlign docs

### Step 11: Commit and Push NeuroAlign Changes

```bash
cd ~/projects/neuroalign

git add .
git commit -m "Refactor: Split preprocessing into separate repository"
git push origin refactor/split-preprocessing

# Create pull request
gh pr create --title "Refactor: Split preprocessing into separate repository" \
             --body "This PR removes preprocessing code and adds neuroalign-preprocessing as a dependency"
```

## Post-Migration Checklist

- [ ] Preprocessing repo is created and tests pass
- [ ] Preprocessing repo is pushed to GitHub
- [ ] NeuroAlign dependencies updated
- [ ] Data adapter is working correctly
- [ ] All imports updated
- [ ] Tests pass in both repositories
- [ ] Documentation updated
- [ ] Example workflows created
- [ ] README files updated with correct links

## Common Issues and Solutions

### Issue: Import errors after migration

**Solution**: Make sure you've updated all imports and installed the packages correctly.

```bash
# In preprocessing repo
cd ~/projects/neuroalign-preprocessing
uv pip install -e .

# In main repo
cd ~/projects/neuroalign  
uv pip install -e .
```

### Issue: Can't find preprocessed data

**Solution**: Make sure the preprocessed data follows the output schema:

```python
from neuroalign.data import PreprocessedDataset

# This should point to derivatives/neuroalign-preprocessing, not the root
dataset = PreprocessedDataset(
    "/path/to/output/derivatives/neuroalign-preprocessing"  # ✓ Correct
)

# Not:
dataset = PreprocessedDataset("/path/to/output")  # ✗ Incorrect
```

### Issue: Tests failing in NeuroAlign

**Solution**: Update test fixtures to use preprocessed data format instead of raw loaders.

## Maintenance Going Forward

### When to update preprocessing repo:
- Adding new data loaders
- Changing feature extraction methods
- Updating parcellations
- Fixing data loading bugs

### When to update NeuroAlign repo:
- Adding new analysis methods
- Improving models
- Adding visualization features
- Updating alignment algorithms

### Coordinating changes:
If preprocessing output format changes, you'll need to:
1. Update OUTPUT_SCHEMA.md
2. Bump major version of preprocessing
3. Update adapter in NeuroAlign
4. Update NeuroAlign's dependency version requirement

## Next Steps

1. Consider publishing both packages to PyPI for easier installation
2. Set up CI/CD for both repositories
3. Create comprehensive test suites
4. Write detailed documentation for each repo
5. Create Jupyter notebook tutorials

## Questions?

If you run into issues during migration, check:
- Import paths are all updated
- Both packages are installed correctly
- Data paths point to the right locations
- Output format matches schema specification
