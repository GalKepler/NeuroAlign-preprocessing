# NeuroAlign-Preprocessing Repository Structure

## Directory Layout

```
neuroalign-preprocessing/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── src/
│   └── neuroalign_preprocessing/
│       ├── __init__.py
│       ├── loaders/
│       │   ├── __init__.py
│       │   ├── anatomical.py
│       │   ├── diffusion.py
│       │   ├── questionnaire.py
│       │   └── base.py                    # Base classes for loaders
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   ├── feature_store.py
│       │   ├── transformers.py
│       │   ├── config.py
│       │   └── cli.py
│       ├── io/
│       │   ├── __init__.py
│       │   ├── writer.py                  # Writes standardized output format
│       │   └── validator.py               # Validates output structure
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── scripts/
│   ├── cat12_tiv_template.m
│   └── preprocess_dataset.py              # Main preprocessing script
├── examples/
│   ├── basic_usage.py
│   └── custom_pipeline.py
├── tests/
│   ├── test_loaders.py
│   ├── test_preprocessing.py
│   └── test_output_format.py
├── docs/
│   ├── OUTPUT_SCHEMA.md                   # Specification for output format
│   ├── INSTALLATION.md
│   └── USAGE.md
└── notebooks/
    └── 01_test_data_loading.ipynb
```

## Key Components

### Loaders (`src/neuroalign_preprocessing/loaders/`)
- Load raw neuroimaging data (BIDS, DICOM, etc.)
- Handle anatomical (T1w, GM, WM, CT) data
- Handle diffusion data (various model outputs: NODDI, DKI, MAPMRI, GQI)
- Load questionnaire/phenotypic data

### Preprocessing (`src/neuroalign_preprocessing/preprocessing/`)
- Feature extraction and aggregation
- Statistical transformations (z-scores, robust statistics, filtering)
- Pipeline orchestration
- Feature store management

### I/O (`src/neuroalign_preprocessing/io/`)
- Write standardized output format
- Validate output structure against schema
- Ensure compatibility with NeuroAlign

### Scripts
- End-to-end preprocessing workflows
- MATLAB templates for CAT12 processing

## Output

The preprocessing pipeline outputs data in a standardized format (see OUTPUT_SCHEMA.md) that serves as the input to NeuroAlign.
