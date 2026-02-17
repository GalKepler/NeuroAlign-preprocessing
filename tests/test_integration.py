import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from pathlib import Path

# Import the main CLI function
from neuroalign_preprocessing.cli import main

# Import the naming utility
from neuroalign_preprocessing.naming_utils import generate_parquet_filename

@pytest.fixture
def mock_sessions_df():
    """Returns a dummy sessions DataFrame."""
    return pd.DataFrame({
        "SubjectCode": ["sub-01-raw", "sub-02-raw"],
        "ScanID": [202407110849, 202407110850], # Example ScanIDs
    })

@pytest.fixture
def mock_parcellation_df():
    """Returns a dummy parcellation DataFrame."""
    return pd.DataFrame({
        "label": [1, 2, 3],
        "value": [0.1, 0.2, 0.3],
    })

# Helper to run the main CLI with arguments
def run_cli(args):
    with patch('sys.argv', ['cli.py'] + args):
        with pytest.raises(SystemExit) as excinfo:
            main()
    assert excinfo.value.code == 0 # Ensure successful exit

def test_cli_cat12_output_naming(tmp_path, mock_sessions_df, mock_parcellation_df):
    output_dir = tmp_path / "output"
    cat12_root = tmp_path / "cat12_data" # Doesn't need to exist due to mocking

    # Mock load_sessions and read_parcellation
    with (
        patch('pandas.read_csv', return_value=mock_sessions_df),
        patch('neuroalign_preprocessing.aggregate.read_parcellation', side_effect=lambda *args, **kwargs: mock_parcellation_df),
    ):

        args = [
            "--cat12-root", str(cat12_root),
            "--sessions", "dummy_sessions.csv", # This path is mocked
            "--output", str(output_dir),
            "--atlas", "Schaefer2018N400n7Tian2020S3",
            "--tissues", "GM", "WM",
            "--mask", "gm",
            "--maskthr", "50",
            "--compression", "none",
        ]
        run_cli(args)

        # Assert output files
        expected_gm_filename = generate_parquet_filename(
            entities={'atlas': "Schaefer2018N400n7Tian2020S3", 'tissue': "GM"},
            modality='cat12',
            mask='gm',
            maskthr=50
        )
        expected_wm_filename = generate_parquet_filename(
            entities={'atlas': "Schaefer2018N400n7Tian2020S3", 'tissue': "WM"},
            modality='cat12',
            mask='gm',
            maskthr=50
        )

        assert (output_dir / "cat12" / expected_gm_filename).exists()
        assert (output_dir / "cat12" / expected_wm_filename).exists()

def test_cli_qsiparc_output_naming(tmp_path, mock_sessions_df, mock_parcellation_df):
    output_dir = tmp_path / "output"
    qsiparc_root = tmp_path / "qsiparc_data" # Doesn't need to exist due to mocking

    # Mock load_sessions and read_parcellation
    with (
        patch('pandas.read_csv', return_value=mock_sessions_df),
        patch('neuroalign_preprocessing.aggregate.read_parcellation', side_effect=lambda *args, **kwargs: mock_parcellation_df),
    ):

        # The glob pattern in aggregate_qsiparc needs to be mocked
        # It calls `qsiparc_root.glob(pattern)`
        # The mock needs to return a list of dummy paths for the tsv files
        # The entities in the tsv filenames are parsed to get model/param
        def mock_glob(self_path, pattern):
            # Simulate discovery of 'tensor_md' and 'csd_fa'
            if "model-*" in pattern and "param-*" in pattern:
                # Need to return actual Path objects relative to the mocked qsiparc_root
                return [
                    self_path / "qsirecon-DSIStudio" / "sub-01" / "ses-01" / "dwi" / "atlas-Schaefer2018N400n7Tian2020S3" / "sub-01_ses-01_atlas-Schaefer2018N400n7Tian2020S3_space-MNI152NLin2009cAsym_res-01_model-tensor_param-md_mask-gm_parc.tsv",
                    self_path / "qsirecon-DSIStudio" / "sub-01" / "ses-01" / "dwi" / "atlas-Schaefer2018N400n7Tian2020S3" / "sub-01_ses-01_atlas-Schaefer2018N400n7Tian2020S3_space-MNI152NLin2009cAsym_res-01_model-csd_param-fa_mask-gm_parc.tsv",
                ]
            return []

        # Mock iterdir for workflow discovery in aggregate_qsiparc
        mock_qsirecon_dir = MagicMock()
        mock_qsirecon_dir.name = "qsirecon-DSIStudio"
        mock_qsirecon_dir.is_dir.return_value = True
        with (
            patch.object(Path, 'iterdir', return_value=[mock_qsirecon_dir]),
            patch.object(Path, 'glob', side_effect=mock_glob, autospec=True), # Mock Path.glob method
        ):

            args = [
                "--qsiparc-root", str(qsiparc_root),
                "--sessions", "dummy_sessions.csv",
                "--output", str(output_dir),
                "--atlas", "Schaefer2018N400n7Tian2020S3",
                "--mask", "gm",
                "--maskthr", "50",
                "--compression", "none",
            ]
            run_cli(args)

            # Assert output files
            expected_dsistudio_tensor_md_filename = generate_parquet_filename(
                entities={'atlas': "Schaefer2018N400n7Tian2020S3", 'model': "tensor", 'param': "md"},
                modality='qsiparc',
                mask='gm',
                maskthr=50
            )
            expected_dsistudio_csd_fa_filename = generate_parquet_filename(
                entities={'atlas': "Schaefer2018N400n7Tian2020S3", 'model': "csd", 'param': "fa"},
                modality='qsiparc',
                mask='gm',
                maskthr=50
            )

            assert (output_dir / "qsiparc" / "DSIStudio" / expected_dsistudio_tensor_md_filename).exists()
            assert (output_dir / "qsiparc" / "DSIStudio" / expected_dsistudio_csd_fa_filename).exists()