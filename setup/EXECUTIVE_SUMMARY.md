# Repository Split: Executive Summary

## Decision

Split NeuroAlign into two repositories:
- **neuroalign-preprocessing**: Data loading and feature extraction
- **neuroalign**: Analysis and modeling

## Rationale

### Technical Benefits
1. **Separation of Concerns**
   - Preprocessing: BIDS/DICOM → Standardized Features
   - Analysis: Features → Insights

2. **Dependency Management**
   - Preprocessing: nibabel, dipy, neuroimaging-specific tools
   - Analysis: scikit-learn, PyTorch, ML/stats tools
   - No conflicts or bloat

3. **Development Velocity**
   - Preprocessing: Stable, infrequent updates
   - Analysis: Rapid iteration on new methods
   - Independent release cycles

4. **Modularity**
   - Users can use their own preprocessing
   - Or start directly with preprocessed data
   - Clear entry points

### User Benefits
1. **Clearer Purpose**: Each repo does one thing well
2. **Easier Onboarding**: Simpler README, focused docs
3. **Flexible Workflows**: Use parts independently
4. **Better Testing**: Smaller, focused test suites

## What Goes Where

### neuroalign-preprocessing
**Purpose**: Transform raw neuroimaging data into standardized features

**Contents**:
- `loaders/` - Anatomical, diffusion, questionnaire loaders
- `preprocessing/` - Pipeline, feature store, transformers
- `io/` - Output writer, validator
- `scripts/` - CAT12 templates, processing scripts

**Output**: Parquet files in standardized schema

### neuroalign
**Purpose**: Analyze features to generate insights

**Contents**:
- `modeling/` - Regional brain age, statistical models
- `alignment/` - Inter-individual alignment analysis
- `embedding/` - Dimensionality reduction, representation learning
- `retrieval/` - Similarity search, query systems
- `agent/` - Interactive analysis interface
- `visualization/` - Plotting, brain surface visualization

**Input**: Preprocessed features from neuroalign-preprocessing

## Data Flow

```
Raw BIDS Dataset
       ↓
[neuroalign-preprocessing]
  • Load anatomical (GM, WM, CT)
  • Load diffusion (NODDI, DKI, MAPMRI, GQI)
  • Extract regional features
  • Compute statistics
  • Validate and save
       ↓
Standardized Feature Files (.parquet)
  • features/anatomical/wide/*.parquet
  • features/diffusion/wide/*.parquet
  • features/metadata.parquet
  • manifest.json
       ↓
[NeuroAlign]
  • Load via PreprocessedDataset
  • Regional brain age estimation
  • Alignment analysis
  • Embedding & retrieval
  • Visualization
       ↓
Analysis Results & Insights
```

## Interface Between Repos

### Data Schema (Contract)

The preprocessing repo outputs data following a strict schema:

```
derivatives/neuroalign-preprocessing/
├── features/
│   ├── anatomical/wide/
│   │   └── gm_mean.parquet        # subjects × parcels
│   ├── diffusion/wide/
│   │   └── DIPYDKI_dki_fa_mean.parquet
│   └── metadata.parquet            # subjects × demographics
└── manifest.json                   # Index of all files
```

### Python API (Adapter)

NeuroAlign loads data via clean adapter:

```python
from neuroalign.data import PreprocessedDataset

dataset = PreprocessedDataset("/path/to/derivatives/neuroalign-preprocessing")
X = dataset.get_features("anatomical", metric="gm", statistic="mean")
metadata = dataset.get_metadata()
```

## Migration Path

### Phase 1: Create Preprocessing Repo (1-2 days)
1. Run migration script
2. Update imports
3. Add output writer
4. Test and push to GitHub

### Phase 2: Update NeuroAlign Repo (1 day)
1. Remove preprocessing code
2. Add preprocessing as dependency
3. Add data adapter
4. Update README

### Phase 3: Testing & Documentation (1 day)
1. Test both repos independently
2. Test integration
3. Update documentation
4. Create examples

### Phase 4: Release (1 day)
1. Tag v0.1.0 for preprocessing
2. Update NeuroAlign to use released version
3. Update all documentation
4. Announce changes

**Total Estimated Time**: 4-5 days

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing code | High | Keep old branch, gradual migration |
| Data format compatibility | Medium | Strict schema validation, versioning |
| User confusion | Medium | Clear docs, migration guide |
| Dependency conflicts | Low | Careful dependency management |

## Success Metrics

- [ ] Both repos have passing tests
- [ ] Clear separation of concerns
- [ ] Documentation is comprehensive
- [ ] Example workflows work end-to-end
- [ ] Existing NeuroAlign functionality preserved
- [ ] New users can understand the flow

## Files Provided

1. **OUTPUT_SCHEMA.md** - Specification for preprocessed data format
2. **README_PREPROCESSING.md** - README for new preprocessing repo
3. **README_NEUROALIGN.md** - Updated README for main repo
4. **MIGRATION_GUIDE.md** - Step-by-step migration instructions
5. **migrate_to_preprocessing.py** - Automated migration script
6. **preprocessed_dataset_adapter.py** - Data loading adapter for NeuroAlign
7. **REPO_STRUCTURE.md** - Directory structure specification

## Decision: Approved ✓

This split makes architectural sense and will improve both code quality and user experience. The migration is straightforward and the benefits are clear.
