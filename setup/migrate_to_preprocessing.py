#!/usr/bin/env python3
"""
Migration script to extract preprocessing components from NeuroAlign
into a separate neuroalign-preprocessing repository.

Usage:
    python migrate_to_preprocessing.py \
        --source /path/to/neuroalign \
        --target /path/to/neuroalign-preprocessing
"""

import argparse
import shutil
from pathlib import Path
import json


def create_directory_structure(target: Path):
    """Create the directory structure for the preprocessing repo."""
    dirs = [
        "src/neuroalign_preprocessing/loaders",
        "src/neuroalign_preprocessing/preprocessing",
        "src/neuroalign_preprocessing/io",
        "src/neuroalign_preprocessing/utils",
        "scripts",
        "examples",
        "tests",
        "docs",
        "notebooks",
    ]
    
    for dir_path in dirs:
        (target / dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created directory structure in {target}")


def copy_loaders(source: Path, target: Path):
    """Copy loader modules from source to target."""
    source_loaders = source / "src/neuroalign/data/loaders"
    target_loaders = target / "src/neuroalign_preprocessing/loaders"
    
    if not source_loaders.exists():
        print(f"⚠ Warning: {source_loaders} not found")
        return
    
    files_to_copy = [
        "anatomical.py",
        "diffusion.py", 
        "questionnaire.py",
        "anatomical_example.py",
        "__init__.py"
    ]
    
    for file in files_to_copy:
        src_file = source_loaders / file
        if src_file.exists():
            shutil.copy2(src_file, target_loaders / file)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠ Skipped {file} (not found)")


def copy_preprocessing(source: Path, target: Path):
    """Copy preprocessing modules from source to target."""
    source_preprocessing = source / "src/neuroalign/data/preprocessing"
    target_preprocessing = target / "src/neuroalign_preprocessing/preprocessing"
    
    if not source_preprocessing.exists():
        print(f"⚠ Warning: {source_preprocessing} not found")
        return
    
    files_to_copy = [
        "pipeline.py",
        "feature_store.py",
        "transformers.py",
        "config.py",
        "cli.py",
        "__init__.py"
    ]
    
    for file in files_to_copy:
        src_file = source_preprocessing / file
        if src_file.exists():
            shutil.copy2(src_file, target_preprocessing / file)
            print(f"  ✓ Copied {file}")
        else:
            print(f"  ⚠ Skipped {file} (not found)")


def copy_scripts(source: Path, target: Path):
    """Copy relevant scripts."""
    source_scripts = source / "scripts"
    target_scripts = target / "scripts"
    
    if not source_scripts.exists():
        print(f"⚠ Warning: {source_scripts} not found")
        return
    
    scripts_to_copy = ["cat12_tiv_template.m"]
    
    for script in scripts_to_copy:
        src_file = source_scripts / script
        if src_file.exists():
            shutil.copy2(src_file, target_scripts / script)
            print(f"  ✓ Copied {script}")


def copy_notebooks(source: Path, target: Path):
    """Copy data loading notebooks."""
    source_notebooks = source / "notebooks"
    target_notebooks = target / "notebooks"
    
    if not source_notebooks.exists():
        return
    
    notebooks_to_copy = ["01_test_data_loading.ipynb"]
    
    for notebook in notebooks_to_copy:
        src_file = source_notebooks / notebook
        if src_file.exists():
            shutil.copy2(src_file, target_notebooks / notebook)
            print(f"  ✓ Copied {notebook}")


def create_pyproject_toml(target: Path):
    """Create pyproject.toml for the preprocessing repo."""
    pyproject_content = '''[project]
name = "neuroalign-preprocessing"
version = "0.1.0"
description = "Neuroimaging data preprocessing pipeline for NeuroAlign"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "pyarrow>=12.0.0",
    "nibabel>=5.0.0",
    "scipy>=1.10.0",
    "tqdm>=4.65.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "ruff>=0.1.0",
    "mypy>=1.0.0",
    "black>=23.0.0",
]

[project.scripts]
neuroalign-preprocess = "neuroalign_preprocessing.preprocessing.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]  # Line too long

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_functions = "test_*"
'''
    
    with open(target / "pyproject.toml", "w") as f:
        f.write(pyproject_content)
    
    print("  ✓ Created pyproject.toml")


def create_init_files(target: Path):
    """Create __init__.py files."""
    init_files = [
        "src/neuroalign_preprocessing/__init__.py",
        "src/neuroalign_preprocessing/loaders/__init__.py",
        "src/neuroalign_preprocessing/preprocessing/__init__.py",
        "src/neuroalign_preprocessing/io/__init__.py",
        "src/neuroalign_preprocessing/utils/__init__.py",
    ]
    
    for init_file in init_files:
        init_path = target / init_file
        if not init_path.exists():
            init_path.touch()


def create_gitignore(target: Path):
    """Create .gitignore file."""
    gitignore_content = '''# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Data
*.nii
*.nii.gz
*.parquet
data/
*.h5
*.hdf5

# Logs
*.log
'''
    
    with open(target / ".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("  ✓ Created .gitignore")


def create_placeholder_modules(target: Path):
    """Create placeholder modules for new functionality."""
    
    # IO Writer
    writer_content = '''"""Output writer for standardized format."""

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
'''
    
    with open(target / "src/neuroalign_preprocessing/io/writer.py", "w") as f:
        f.write(writer_content)
    
    print("  ✓ Created io/writer.py placeholder")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "="*60)
    print("MIGRATION COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("\n1. Review migrated files and update imports:")
    print("   - Change 'from neuroalign.data.loaders' to 'from neuroalign_preprocessing.loaders'")
    print("   - Change 'from neuroalign.data.preprocessing' to 'from neuroalign_preprocessing.preprocessing'")
    
    print("\n2. Initialize git repository:")
    print("   cd /path/to/neuroalign-preprocessing")
    print("   git init")
    print("   git add .")
    print('   git commit -m "Initial commit: Extract preprocessing from NeuroAlign"')
    
    print("\n3. Create GitHub repository and push:")
    print("   gh repo create neuroalign-preprocessing --public")
    print("   git remote add origin https://github.com/yourusername/neuroalign-preprocessing")
    print("   git push -u origin main")
    
    print("\n4. Update original NeuroAlign repository:")
    print("   - Remove preprocessing code")
    print("   - Add neuroalign-preprocessing as dependency")
    print("   - Update README.md")
    print("   - Create adapter for loading preprocessed data")
    
    print("\n5. Test the new setup:")
    print("   cd /path/to/neuroalign-preprocessing")
    print("   uv venv")
    print("   source .venv/bin/activate")
    print("   uv pip install -e .")
    print("   pytest tests/")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate preprocessing components to separate repository"
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to existing NeuroAlign repository"
    )
    parser.add_argument(
        "--target", 
        type=Path,
        required=True,
        help="Path to new neuroalign-preprocessing repository (will be created)"
    )
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting migration...")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}\n")
    
    # Create target directory
    args.target.mkdir(parents=True, exist_ok=True)
    
    # Execute migration steps
    print("1. Creating directory structure...")
    create_directory_structure(args.target)
    
    print("\n2. Copying loader modules...")
    copy_loaders(args.source, args.target)
    
    print("\n3. Copying preprocessing modules...")
    copy_preprocessing(args.source, args.target)
    
    print("\n4. Copying scripts...")
    copy_scripts(args.source, args.target)
    
    print("\n5. Copying notebooks...")
    copy_notebooks(args.source, args.target)
    
    print("\n6. Creating configuration files...")
    create_pyproject_toml(args.target)
    create_gitignore(args.target)
    create_init_files(args.target)
    
    print("\n7. Creating placeholder modules...")
    create_placeholder_modules(args.target)
    
    # Copy documentation
    print("\n8. Copying documentation...")
    shutil.copy2(
        Path(__file__).parent / "OUTPUT_SCHEMA.md",
        args.target / "docs/OUTPUT_SCHEMA.md"
    )
    shutil.copy2(
        Path(__file__).parent / "README_PREPROCESSING.md",
        args.target / "README.md"
    )
    print("  ✓ Copied documentation")
    
    print_next_steps()


if __name__ == "__main__":
    main()
